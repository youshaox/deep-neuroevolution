import logging
import os
import pickle
import time
from collections import deque
from pprint import pformat

import redis

logger = logging.getLogger(__name__)

EXP_KEY = 'es:exp'
TASK_ID_KEY = 'es:task_id'
TASK_DATA_KEY = 'es:task_data'
TASK_CHANNEL = 'es:task_channel'
RESULTS_KEY = 'es:results'
ARCHIVE_KEY = 'es:archive'

def serialize(x):
    return pickle.dumps(x, protocol=-1)


def deserialize(x):
    return pickle.loads(x)


def retry_connect(redis_cfg, tries=300, base_delay=4.):
    for i in range(tries):
        try:
            r = redis.StrictRedis(**redis_cfg)
            r.ping()
            return r
        except redis.ConnectionError as e:
            if i == tries - 1:
                raise
            else:
                delay = base_delay * (1 + (os.getpid() % 10) / 9)
                logger.warning('Could not connect to {}. Retrying after {:.2f} sec ({}/{}). Error: {}'.format(
                    redis_cfg, delay, i + 2, tries, e))
                time.sleep(delay)


def retry_get(pipe, key, tries=300, base_delay=4.):
    for i in range(tries):
        # Try to (m)get
        if isinstance(key, (list, tuple)):
            vals = pipe.mget(key)
            if all(v is not None for v in vals):
                return vals
        else:
            val = pipe.get(key)
            if val is not None:
                return val
        # Sleep and retry if any key wasn't available
        if i != tries - 1:
            delay = base_delay * (1 + (os.getpid() % 10) / 9)
            logger.warning('{} not set. Retrying after {:.2f} sec ({}/{})'.format(key, delay, i + 2, tries))
            time.sleep(delay)
    raise RuntimeError('{} not set'.format(key))


class MasterClient:
    def __init__(self, master_redis_cfg):
        self.task_counter = 0
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[master] Connected to Redis: {}'.format(self.master_redis))

    def declare_experiment(self, exp):
        self.master_redis.set(EXP_KEY, serialize(exp))
        logger.info('[master] Declared experiment {}'.format(pformat(exp)))

    def declare_task(self, task_data):
        task_id = self.task_counter
        self.task_counter += 1
        # logger.debug("I have declared task ")
        # todo !!!!!!! here !!!!!!! 问题就在这里 它的确declare_task 8，但是不知道又没有成功，因为下面应该没有收到task 8不知道有没有成功，因为从redis中取任务，看起来没有task8
        serialized_task_data = serialize(task_data)
        # 查看有没有成功
        logger.debug("[Master] 检查push任务有没有成功")
        publish_result = (self.master_redis.pipeline()
         .mset({TASK_ID_KEY: task_id, TASK_DATA_KEY: serialized_task_data})
         .publish(TASK_CHANNEL, serialize((task_id, serialized_task_data)))
         .execute()) # TODO: can we avoid transferring task data twice and serializing so much?
        logger.debug(type(publish_result))
        logger.debug(publish_result)
        logger.debug('[master] Declared task {}'.format(task_id))
        return task_id

    def pop_result(self):
        task_id, result = deserialize(self.master_redis.blpop(RESULTS_KEY)[1])
        logger.debug('[master] Popped a result for task {}'.format(task_id))
        return task_id, result

    def flush_results(self):
        return max(self.master_redis.pipeline().llen(RESULTS_KEY).ltrim(RESULTS_KEY, -1, -1).execute()[0] -1, 0)

    def add_to_novelty_archive(self, novelty_vector):
        self.master_redis.rpush(ARCHIVE_KEY, serialize(novelty_vector))
        logger.info('[master] Added novelty vector to archive')

    def get_archive(self):
        archive = self.master_redis.lrange(ARCHIVE_KEY, 0, -1)
        return [deserialize(novelty_vector) for novelty_vector in archive]


class RelayClient:
    """
    Receives and stores task broadcasts from the master
    Batches and pushes results from workers to the master
    """

    def __init__(self, master_redis_cfg, relay_redis_cfg):
        self.master_redis = retry_connect(master_redis_cfg)
        logger.info('[relay] Connected to master: {}'.format(self.master_redis))
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[relay] Connected to relay: {}'.format(self.local_redis))
        self.results_published = 0

    def run(self):
        # todo  current 这块没有执行 !!!!!!!!!!!!!!!!!!!!!!
        # Initialization: read exp and latest task from master
        # 设定exp
        self.local_redis.set(EXP_KEY, retry_get(self.master_redis, EXP_KEY))
        self._declare_task_local(*retry_get(self.master_redis, (TASK_ID_KEY, TASK_DATA_KEY)))
        logger.debug("[relay] _declare_task_local 1nd pos")

        # 这上面是最开始做一次的
        # Start subscribing to tasks

        p = self.master_redis.pubsub(ignore_subscribe_messages=True)
        ## todo !!!!!!!!他妈的就是这里，here 我认为问题处在了这里，没有得到新的任务!!!!!!!，我这边是看到task 8的
        test_p = self.master_redis.pubsub(ignore_subscribe_messages=True)
        # todo
        # test_p.subscribe(**{TASK_CHANNEL: lambda msg: print_task_id_from_channel(*deserialize(msg['data']))})
        # 搞懂这句话的意思
        # 学习下如何使用subscribe
        # 考虑加个循环检查下有没有任务

        p = self.master_redis.pubsub(ignore_subscribe_messages=True)
        while True:
            message = p.get_message()
            logger.debug('---------- test  没有监听到新的task ----------')
            if message:
                logger.debug(
                    '---------- test  监听到新的task id: {} ----------'.format(str(deserialize(message['data'][0]))))
                self._declare_task_local(*deserialize(message['data']))
            time.sleep(0.001)  # be nice to the system :)


        p.subscribe(**{TASK_CHANNEL: lambda msg: self._declare_task_local(*deserialize(msg['data']))})

        logger.debug("[relay] _declare_task_local 2nd pos")
        p.run_in_thread(sleep_time=0.001)

        # Loop on RESULTS_KEY and push to master
        batch_sizes, last_print_time = deque(maxlen=20), time.time()  # for logging
        while True:
            results = []
            start_time = curr_time = time.time()
            while curr_time - start_time < 0.001:
                results.append(self.local_redis.blpop(RESULTS_KEY)[1])
                curr_time = time.time()
            self.results_published += len(results)
            self.master_redis.rpush(RESULTS_KEY, *results)
            # Log
            batch_sizes.append(len(results))
            if curr_time - last_print_time > 5.0:
                logger.info('[relay] Average batch size {:.3f} ({} total)'.format(sum(batch_sizes) / len(batch_sizes), self.results_published))
                last_print_time = curr_time

    def flush_results(self):
        number_flushed = max(self.local_redis.pipeline().llen(RESULTS_KEY).ltrim(RESULTS_KEY, -1, -1).execute()[0] -1, 0)
        number_flushed_master = max(self.master_redis.pipeline().llen(RESULTS_KEY).ltrim(RESULTS_KEY, -1, -1).execute()[0] -1, 0)
        logger.warning('[relay] Flushed {} results from worker redis and {} from master'
            .format(number_flushed, number_flushed_master))
    # todo current 没有进行到这一步
    def _declare_task_local(self, task_id, task_data):
        logger.info('[relay] Received task {}'.format(task_id))
        self.results_published = 0
        self.local_redis.mset({TASK_ID_KEY: task_id, TASK_DATA_KEY: task_data})
        self.flush_results()
    # todo
    def _printtask_id(self, task_id, task_data):
        logger.info('********************************[relay] _printtask_id: {}'.format(task_id))


class WorkerClient:
    def __init__(self, relay_redis_cfg, master_redis_cfg):
        self.local_redis = retry_connect(relay_redis_cfg)
        logger.info('[worker] Connected to relay: {}'.format(self.local_redis))
        self.master_redis = retry_connect(master_redis_cfg)
        logger.warning('[worker] Connected to master: {}'.format(self.master_redis))

        self.cached_task_id, self.cached_task_data = None, None

    def get_experiment(self):
        # Grab experiment info
        exp = deserialize(retry_get(self.local_redis, EXP_KEY))
        logger.info('[worker] Experiment: {}'.format(exp))
        return exp

    def get_archive(self):
        archive = self.master_redis.lrange(ARCHIVE_KEY, 0, -1)
        return [deserialize(novelty_vector) for novelty_vector in archive]

    def get_current_task(self):
        # 该步运行完成，但是task_9没有
        with self.local_redis.pipeline() as pipe:
            # 直到获得cached_task_id 和 cached_task_data
            while True:
                try:
                    pipe.watch(TASK_ID_KEY)
                    # todo current !!!!!!!!!!
                    # 说明卡在这里，一直获得的是旧的任务
                    task_id = int(retry_get(pipe, TASK_ID_KEY))
                    # 如果任务id还和cached)task_id一样的化，继续看有没有新的任务
                    if task_id == self.cached_task_id:
                        # 一直在这里呢！！！！！！！
                        logger.debug('[worker] Returning cached task {}'.format(task_id))
                        break
                    pipe.multi()
                    pipe.get(TASK_DATA_KEY)
                    # todo 这步其实只有一开始进行了，
                    logger.info('[worker] Getting new task {}. Cached task was {}'.format(task_id, self.cached_task_id))
                    self.cached_task_id, self.cached_task_data = task_id, deserialize(pipe.execute()[0])
                    break
                except redis.WatchError:
                    continue
        return self.cached_task_id, self.cached_task_data

    def push_result(self, task_id, result):
        self.local_redis.rpush(RESULTS_KEY, serialize((task_id, result)))
        logger.debug('[worker] Pushed result for task {}'.format(task_id))
