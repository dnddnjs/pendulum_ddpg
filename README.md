# pendulum_ddpg
continuous control of pendulum with ddpg
[ddpg paper](https://arxiv.org/abs/1509.02971)

## ddpg
One can have trouble with applying value-based reinforcement learning to continuous action problem. In DQN, most famous value-based RL algorithm, agent choose action according to the epsilon-greedy action selection strategy. But if action is continuous, choosing according to Q-function becomes problem.

DDPG(Deep Deterministic Policy Gradient) is an variant of policy gradient algorithms. It uses actor-critic architecture to solve continuous problem. The output of actor is not the probability of actions but action itself which is deterministic policy.

Most Policy Gradient Algorithms uses "Policy Gradient Theorem". David Silver proved policy gradient theorem can be applied to deterministic policy and called it as deterministic policy gradient theorem. 

The policy gradient of objective function by the "determinitic policy gradient theorem" is like this.
