# pendulum_ddpg
continuous control of pendulum with ddpg
[ddpg paper](https://arxiv.org/abs/1509.02971)
<center><img src="/img/pendulum.gif" width="300" height="300"></center>

## Deep Deterministic Policy Gradient
One can have trouble with applying value-based reinforcement learning to continuous action problem. In DQN, most famous value-based RL algorithm, agent choose action according to the epsilon-greedy action selection strategy. But if action is continuous, choosing according to Q-function becomes problem.

DDPG(Deep Deterministic Policy Gradient) is an variant of policy gradient algorithms. It uses actor-critic architecture to solve continuous problem. The output of actor is not the probability of actions but action itself which is deterministic policy.

Most Policy Gradient Algorithms uses "[Policy Gradient Theorem](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)". David Silver proved policy gradient theorem can be applied to deterministic policy and called it as deterministic policy gradient theorem. 

The policy gradient of objective function by the "determinitic policy gradient theorem". It is gradient of Q-function of the selected action and using chain-rule, one can get the policy gradient of deterministic policy gradient.

## Requirements
1. Python 3.5
2. Tensorflow
3. Keras
4. numpy
5. scipy
6. matplotlib
7. h5py
8. gym

## Example
I used [Pendulum-v0](https://gym.openai.com/envs/Pendulum-v0/) environment of openai gym to test ddpg algorithm. It is simplist continuous action environment. You can train ddpg agent like this
'''shell
python3 pendulum_ddpg.py
'''

