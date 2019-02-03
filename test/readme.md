1. returns 算的是累计的求和，所以同novelty一样，所以一次rollout/episode 只返回一个[x,y]

   ```python
   # rewards :: sum of the positive and sum of negative rewards.
   returns_n2 = [rews_pos.sum(), rews_neg.sum()]
   
   # rewards :: novelty of the actions in one episode (policy parameterised either by postive or negative)
   signreturns_n2 = [nov_pos, nov_neg]
   
   ```

2. The workers will calculate **the results based on the mutated parameter** since the random state is fixed multiple times in one task.