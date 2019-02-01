import logging
import time
from collections import namedtuple
import tensorflow as tf
from copy import deepcopy

import numpy as np

from .dist import MasterClient, WorkerClient
from .es import *


def euclidean_distance(x, y):
    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a ** 2 + b ** 2)


def compute_novelty_vs_archive(archive, novelty_vector, k):
    distances = []
    nov = novelty_vector.astype(np.float)
    for point in archive:
        distances.append(euclidean_distance(point.astype(np.float), nov))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]

    ##
    result = top_k.mean()
    # todo current
    with open('novelty_vs_archive','a+') as f:
        f.write(str(result))
    return result


def get_mean_bc(env, policy, tslimit, num_rollouts=1):
    novelty_vector = []
    for n in range(num_rollouts):
        rew, t, nv = policy.rollout(env, timestep_limit=tslimit)
        novelty_vector.append(nv)
    return np.mean(novelty_vector, axis=0)


def setup_env(exp):
    import gym
    gym.undo_logger_setup()
    config = Config(**exp['config'])
    env = gym.make(exp['env_id'])
    if exp['policy']['type'] == "ESAtariPolicy":
        from .atari_wrappers import wrap_deepmind
        env = wrap_deepmind(env)
    return config, env


def setup_policy(env, exp, single_threaded):
    from . import policies, tf_util
    sess = make_session(single_threaded=single_threaded)
    policy = getattr(policies, exp['policy']['type'])(env.observation_space, env.action_space, **exp['policy']['args'])
    tf_util.initialize()
    return sess, policy


def run_master(master_redis_cfg, log_dir, exp):
    logger.info('run_master: {}'.format(locals()))
    from .optimizers import SGD, Adam
    from . import tabular_logger as tlogger
    config, env = setup_env(exp)
    algo_type = exp['algo_type']
    master = MasterClient(master_redis_cfg)
    noise = SharedNoiseTable()
    rs = np.random.RandomState()
    # todo 看一下这个ref_batch 怎么用的
    ref_batch = get_ref_batch(env, batch_size=128)

    pop_size = int(exp['novelty_search']['population_size'])
    num_rollouts = int(exp['novelty_search']['num_rollouts'])
    theta_dict = {}
    optimizer_dict = {}
    obstat_dict = {}
    curr_parent = 0

    # 计时相关的 看看tslimit怎么用
    if isinstance(config.episode_cutoff_mode, int):
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = config.episode_cutoff_mode, None, None, config.episode_cutoff_mode
        adaptive_tslimit = False
    # adapative改进的
    elif config.episode_cutoff_mode.startswith('adaptive:'):
        _, args = config.episode_cutoff_mode.split(':')
        arg0, arg1, arg2, arg3 = args.split(',')
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = int(arg0), float(arg1), float(arg2), float(
            arg3)
        adaptive_tslimit = True
        logger.info(
            'Starting timestep limit set to {}. When {}% of rollouts hit the limit, it will be increased by {}. The maximum timestep limit is {}'.format(
                tslimit, incr_tslimit_threshold * 100, tslimit_incr_ratio, tslimit_max))

    elif config.episode_cutoff_mode == 'env_default':
        tslimit, incr_tslimit_threshold, tslimit_incr_ratio, tslimit_max = None, None, None, None
        adaptive_tslimit = False
    else:
        raise NotImplementedError(config.episode_cutoff_mode)
    # 初始化
    for p in range(pop_size):
        with tf.Graph().as_default():
            sess, policy = setup_policy(env, exp, single_threaded=False)

            if 'init_from' in exp['policy']:
                logger.info('Initializing weights from {}'.format(exp['policy']['init_from']))
                policy.initialize_from(exp['policy']['init_from'], ob_stat)

            theta = policy.get_trainable_flat()
            optimizer = {'sgd': SGD, 'adam': Adam}[exp['optimizer']['type']](theta, **exp['optimizer']['args'])
            # mujoco需要ob_stat
            if policy.needs_ob_stat:
                ob_stat = RunningStat(env.observation_space.shape, eps=1e-2)
                obstat_dict[p] = ob_stat
            # atari需要ref_batch
            if policy.needs_ref_batch:
                policy.set_ref_batch(ref_batch)

            mean_bc = get_mean_bc(env, policy, tslimit_max, num_rollouts)
            master.add_to_novelty_archive(mean_bc)

            theta_dict[p] = theta
            optimizer_dict[p] = optimizer

    episodes_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    master.declare_experiment(exp)
    # 真正一直在循环的
    while True:
        step_tstart = time.time()

        theta = theta_dict[curr_parent]
        # ???? 为什么 master 也需要set trainable flat
        policy.set_trainable_flat(theta)
        optimizer = optimizer_dict[curr_parent]

        if policy.needs_ob_stat:
            ob_stat = deepcopy(obstat_dict[curr_parent])

        assert theta.dtype == np.float32
        # 新的任务
        curr_task_id = master.declare_task(Task(
            params=theta,
            ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
            ob_std=ob_stat.std if policy.needs_ob_stat else None,
            ref_batch=ref_batch if policy.needs_ref_batch else None,
            timestep_limit=tslimit
        ))
        master.flush_results()
        new_task_checker = False
        # 检查是否为新的任务
        while not new_task_checker:
            # Query master to see if new task declaration registers
            for _ in range(1000):
                temp_task_id, _ = master.pop_result()
                if temp_task_id == curr_task_id:
                    new_task_checker = True;
                    break

            # Re-declare task if original declaration fails to register
            if not new_task_checker:
                master.task_counter -= 1
                curr_task_id = master.declare_task(Task(
                    params=theta,
                    ob_mean=ob_stat.mean if policy.needs_ob_stat else None,
                    ob_std=ob_stat.std if policy.needs_ob_stat else None,
                    ref_batch=ref_batch if policy.needs_ref_batch else None,
                    timestep_limit=tslimit
                ))
        tlogger.log('********** Iteration {} **********'.format(curr_task_id))
        # eval_rets的是什么
        # todo 看起来这一块可以跳过，其实是把所有结果合成一个list在一起: curr_task_results
        # 这个会一直pop master收到的results直到超过episode_per_batch
        # Pop off results for the current task
        curr_task_results, eval_rets, eval_lens, worker_ids = [], [], [], []
        num_results_skipped, num_episodes_popped, num_timesteps_popped, ob_count_this_batch = 0, 0, 0, 0
        # todo 分清楚 batch, episodes, timesteps, result
        """
        result: 是每次worker的一个结果，是一个Result对象
        episode:
        timestep:
        bathc:
        
        """
        # number of episodes has popped
        while num_episodes_popped < config.episodes_per_batch or num_timesteps_popped < config.timesteps_per_batch:
            # Wait for a result
            task_id, result = master.pop_result()
            assert isinstance(task_id, int) and isinstance(result, Result)
            assert (result.eval_return is None) == (result.eval_length is None)
            worker_ids.append(result.worker_id)
            # 这个是eval_return, eval_length 非空的，其实就是之前的，没有改变参数的参数，重新rollout得到的rewards
            # Evaluation: noiseless weights and noiseless actions
            if result.eval_length is not None:
                # This was an eval job
                episodes_so_far += 1
                timesteps_so_far += result.eval_length
                # Store the result only for current tasks
                if task_id == curr_task_id:
                    # todo 看看这个eval_rets是怎么来的，不同函数的差别。 !!!!!!!!!
                    # evaluate returns
                    eval_rets.append(result.eval_return)
                    eval_lens.append(result.eval_length)
            # Evaluation: noise weights and noise actions
            else:
                # 这个新的参数后rewards
                assert (result.noise_inds_n.ndim == 1 and
                        result.returns_n2.shape == result.lengths_n2.shape == (len(result.noise_inds_n), 2))
                assert result.returns_n2.dtype == np.float32
                # Update counts
                # lengths_ns 每个worker只有一个[len_pos, len_neg]，所以lengths_n2.size =2
                result_num_eps = result.lengths_n2.size
                result_num_timesteps = result.lengths_n2.sum()
                episodes_so_far += result_num_eps
                timesteps_so_far += result_num_timesteps
                # Store results only for current tasks
                if task_id == curr_task_id:
                    curr_task_results.append(result)
                    num_episodes_popped += result_num_eps
                    num_timesteps_popped += result_num_timesteps
                    # Update ob stats
                    if policy.needs_ob_stat and result.ob_count > 0:
                        ob_stat.increment(result.ob_sum, result.ob_sumsq, result.ob_count)
                        ob_count_this_batch += result.ob_count
                else:
                    num_results_skipped += 1

        # Compute skip fraction
        # 因为只取config.episodes_per_batch 或者 config.timesteps_per_batch 数量的results
        frac_results_skipped = num_results_skipped / (num_results_skipped + len(curr_task_results))
        if num_results_skipped > 0:
            logger.warning('Skipped {} out of date results ({:.2f}%)'.format(
                num_results_skipped, 100. * frac_results_skipped))

        #### 这下面是如何利用novelty 和 reward来更新参数的
        # Assemble results
        noise_inds_n = np.concatenate([r.noise_inds_n for r in curr_task_results])
        returns_n2 = np.concatenate([r.returns_n2 for r in curr_task_results])
        lengths_n2 = np.concatenate([r.lengths_n2 for r in curr_task_results])
        # 这个returns_n2就是关键的 用来排序的
        signreturns_n2 = np.concatenate([r.signreturns_n2 for r in curr_task_results])

        assert noise_inds_n.shape[0] == returns_n2.shape[0] == lengths_n2.shape[0]
        # Process returns
        if config.return_proc_mode == 'centered_rank':
            proc_returns_n2 = compute_centered_ranks(returns_n2)
        elif config.return_proc_mode == 'sign':
            proc_returns_n2 = signreturns_n2
        elif config.return_proc_mode == 'centered_sign_rank':
            # 这是对novelty进行排序
            proc_returns_n2 = compute_centered_ranks(signreturns_n2)
        else:
            raise NotImplementedError(config.return_proc_mode)
        # todo 看下区别 !!!! current 为了检查我们那个是否合格
        with open('proc_returns_n2_novely.txt', 'a+') as f:
            f.write(str(proc_returns_n2))

        if algo_type == "nsr":
            rew_ranks = compute_centered_ranks(returns_n2)
            # 这是对reward和novelty进行排序 todo 我们以后要改的地方，不是除以2了
            proc_returns_n2 = (rew_ranks + proc_returns_n2) / 2.0
        # todo 看下区别 !!!! current
        with open('proc_returns_n2_both.txt','a+') as f:
            f.write(str(proc_returns_n2))

        # 其实就是个加权求和所有的综合的reward*noise
        # Compute and take step
        g, count = batched_weighted_sum(
            # 看起来是把pos_nov - neg_nov
            proc_returns_n2[:, 0] - proc_returns_n2[:, 1],
            # 获取和policy.num_params同等数量的参数
            (noise.get(idx, policy.num_params) for idx in noise_inds_n),
            batch_size=500
        )
        g /= returns_n2.size
        # todo 计算每一步的g !!!!! current
        # logger.debug("global gradient g:{}".format(str(g)))
        assert g.shape == (policy.num_params,) and g.dtype == np.float32 and count == len(noise_inds_n)
        # 按照梯度更新比率和参数
        update_ratio, theta = optimizer.update(-g + config.l2coeff * theta)
        # 更新了paramters
        policy.set_trainable_flat(theta)

        # Update ob stat (we're never running the policy in the master, but we might be snapshotting the policy)
        if policy.needs_ob_stat:
            policy.set_ob_stat(ob_stat.mean, ob_stat.std)
        # 按照目前的参数，神经网络得到的唯一bc
        mean_bc = get_mean_bc(env, policy, tslimit_max, num_rollouts)
        master.add_to_novelty_archive(mean_bc)

        # todo 以后我们会做的，adaptive来更新参数
        # Update number of steps to take，为了更新权重来的，增加了tslimit
        if adaptive_tslimit and (lengths_n2 == tslimit).mean() >= incr_tslimit_threshold:
            old_tslimit = tslimit
            tslimit = min(int(tslimit_incr_ratio * tslimit), tslimit_max)
            logger.info('Increased timestep limit from {} to {}'.format(old_tslimit, tslimit))
        # TODO 看懂这个表 今天
        step_tend = time.time()
        tlogger.record_tabular("ParentId", curr_parent)
        # # returns就是rewards[rews_pos.sum(), rews_neg.sum()]，size为2
        # todo
        logger.info(returns_n2.size)
        # 这步取均值是.2秒可能多次？？？
        tlogger.record_tabular("EpRewMean", returns_n2.mean())
        tlogger.record_tabular("EpRewStd", returns_n2.std())
        ## 一次rollout的rewards的length
        tlogger.record_tabular("EpLenMean", lengths_n2.mean())
        # evaluate de rewards
        tlogger.record_tabular("EvalEpRewMean", np.nan if not eval_rets else np.mean(eval_rets))
        tlogger.record_tabular("EvalEpRewStd", np.nan if not eval_rets else np.std(eval_rets))
        tlogger.record_tabular("EvalEpLenMean", np.nan if not eval_rets else np.mean(eval_lens))
        tlogger.record_tabular("EvalPopRank", np.nan if not eval_rets else (
            np.searchsorted(np.sort(returns_n2.ravel()), eval_rets).mean() / returns_n2.size))
        tlogger.record_tabular("EvalEpCount", len(eval_rets))
        # 参数的范数
        tlogger.record_tabular("Norm", float(np.square(policy.get_trainable_flat()).sum()))
        # 梯度的范数
        tlogger.record_tabular("GradNorm", float(np.square(g).sum()))
        tlogger.record_tabular("UpdateRatio", float(update_ratio))
        # 一般来说不是2吗？
        tlogger.record_tabular("EpisodesThisIter", lengths_n2.size)
        # 至今为止的episodes
        tlogger.record_tabular("EpisodesSoFar", episodes_so_far)
        tlogger.record_tabular("TimestepsThisIter", lengths_n2.sum())
        tlogger.record_tabular("TimestepsSoFar", timesteps_so_far)

        num_unique_workers = len(set(worker_ids))
        tlogger.record_tabular("UniqueWorkers", num_unique_workers)
        tlogger.record_tabular("UniqueWorkersFrac", num_unique_workers / len(worker_ids))
        tlogger.record_tabular("ResultsSkippedFrac", frac_results_skipped)
        tlogger.record_tabular("ObCount", ob_count_this_batch)

        tlogger.record_tabular("TimeElapsedThisIter", step_tend - step_tstart)
        tlogger.record_tabular("TimeElapsed", step_tend - tstart)
        tlogger.dump_tabular()

        # updating population parameters
        theta_dict[curr_parent] = policy.get_trainable_flat()
        optimizer_dict[curr_parent] = optimizer
        if policy.needs_ob_stat:
            obstat_dict[curr_parent] = ob_stat

        if exp['novelty_search']['selection_method'] == "novelty_prob":
            novelty_probs = []
            archive = master.get_archive()
            for p in range(pop_size):
                policy.set_trainable_flat(theta_dict[p])
                mean_bc = get_mean_bc(env, policy, tslimit_max, num_rollouts)
                nov_p = compute_novelty_vs_archive(archive, mean_bc, exp['novelty_search']['k'])
                novelty_probs.append(nov_p)
            novelty_probs = np.array(novelty_probs) / float(np.sum(novelty_probs))
            # 重新选下代curr_parent
            curr_parent = np.random.choice(range(pop_size), 1, p=novelty_probs)[0]
        elif exp['novelty_search']['selection_method'] == "round_robin":
            curr_parent = (curr_parent + 1) % pop_size
        else:
            raise NotImplementedError(exp['novelty_search']['selection_method'])
        if config.snapshot_freq != 0 and curr_task_id % config.snapshot_freq == 0:
            import os.path as osp
            filename = 'snapshot_iter{:05d}_rew{}.h5'.format(
                curr_task_id,
                np.nan if not eval_rets else int(np.mean(eval_rets))
            )
            assert not osp.exists(filename)
            policy.save(filename)
            tlogger.log('Saved snapshot {}'.format(filename))


def run_worker(master_redis_cfg, relay_redis_cfg, noise, *, min_task_runtime=.2):
    logger.info('run_worker: {}'.format(locals()))
    assert isinstance(noise, SharedNoiseTable)
    worker = WorkerClient(relay_redis_cfg, master_redis_cfg)
    exp = worker.get_experiment()
    config, env = setup_env(exp)
    sess, policy = setup_policy(env, exp, single_threaded=False)
    rs = np.random.RandomState()
    worker_id = rs.randint(2 ** 31)
    previous_task_id = -1

    assert policy.needs_ob_stat == (config.calc_obstat_prob != 0)

    while True:
        task_id, task_data = worker.get_current_task()
        task_tstart = time.time()
        assert isinstance(task_id, int) and isinstance(task_data, Task)

        if policy.needs_ob_stat:
            policy.set_ob_stat(task_data.ob_mean, task_data.ob_std)

        if policy.needs_ref_batch:
            policy.set_ref_batch(task_data.ref_batch)

        # 只开始一次
        if task_id != previous_task_id:
            archive = worker.get_archive()
            previous_task_id = task_id

        if rs.rand() < config.eval_prob:
            # Evaluation: noiseless weights and noiseless actions
            policy.set_trainable_flat(task_data.params)
            eval_rews, eval_length, _ = policy.rollout(env, timestep_limit=task_data.timestep_limit)
            eval_return = eval_rews.sum()
            logger.info('Eval result: task={} return={:.3f} length={}'.format(task_id, eval_return, eval_length))
            # 这个是不改变参数的，和之前一样
            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=None,
                returns_n2=None,
                signreturns_n2=None,
                lengths_n2=None,
                eval_return=eval_return,
                eval_length=eval_length,
                ob_sum=None,
                ob_sumsq=None,
                ob_count=None
            ))
        else:
            # Rollouts with noise
            noise_inds, returns, signreturns, lengths = [], [], [], []
            # todo 看下RunningState干啥
            task_ob_stat = RunningStat(env.observation_space.shape, eps=0.)  # eps=0 because we're incrementing only

            while not noise_inds or time.time() - task_tstart < min_task_runtime:
                noise_idx = noise.sample_index(rs, policy.num_params)
                # 改变参数
                v = config.noise_stdev * noise.get(noise_idx, policy.num_params)
                policy.set_trainable_flat(task_data.params + v)
                # a list of timestemp of reward
                # nov_vec_pos: an array of RAM(253,128)
                rews_pos, len_pos, nov_vec_pos = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)

                policy.set_trainable_flat(task_data.params - v)
                rews_neg, len_neg, nov_vec_neg = rollout_and_update_ob_stat(
                    policy, env, task_data.timestep_limit, rs, task_ob_stat, config.calc_obstat_prob)
                # nov_pos是一个值
                nov_pos = compute_novelty_vs_archive(archive, nov_vec_pos, exp['novelty_search']['k'])
                nov_neg = compute_novelty_vs_archive(archive, nov_vec_neg, exp['novelty_search']['k'])
                # novelty
                signreturns.append([nov_pos, nov_neg])
                noise_inds.append(noise_idx)
                # 一个子worker可以生成很多次rewards直到min_task_runtime，一般就生成一个
                # returns就是rewards[rews_pos.sum(), rews_neg.sum()]，size为2
                returns.append([rews_pos.sum(), rews_neg.sum()])
                lengths.append([len_pos, len_neg])
                # todo current
                logger.debug('length of signreturns {}'.format(len(signreturns)))
                logger.debug(lengths.size)
                logger.debug('length of signreturns {}'.format(len(noise_inds)))

            worker.push_result(task_id, Result(
                worker_id=worker_id,
                noise_inds_n=np.array(noise_inds),
                returns_n2=np.array(returns, dtype=np.float32),
                signreturns_n2=np.array(signreturns, dtype=np.float32),
                lengths_n2=np.array(lengths, dtype=np.int32),
                eval_return=None,
                eval_length=None,
                ob_sum=None if task_ob_stat.count == 0 else task_ob_stat.sum,
                ob_sumsq=None if task_ob_stat.count == 0 else task_ob_stat.sumsq,
                ob_count=task_ob_stat.count
            ))