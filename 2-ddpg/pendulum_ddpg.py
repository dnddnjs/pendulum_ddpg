from collections import deque
import numpy as np
from keras.layers import Dense, Input, Add, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import random
import gym


def ou_noise(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)


def normal(shape, scale=0.05, name=None):
    return K.variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)


class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.render = True
        self.load_model = False

        # build networks
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()
        self.actor_updater = self.actor_optimizer()

        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.discount_factor = 0.99
        self.epsilon = 1
        self.epsilon_decay = 1/2000

    def build_actor(self):
        print("building actor network")
        input = Input(shape=[self.state_size])
        h1 = Dense(30, activation='relu')(input)
        h2 = Dense(30, activation='relu')(h1)
        action = Dense(1, activation='tanh')(h2)
        actor = Model(inputs=input, outputs=action)
        return actor

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        actions = self.actor.output
        dqda = tf.gradients(self.critic.output, self.critic.input)
        loss = - actions * dqda[1]

        optimizer = Adam(lr=0.001)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, self.critic.input[0],
                            self.critic.input[1]], [], updates=updates)
        return train

    def build_critic(self):
        print("building critic network")
        state = Input(shape=[self.state_size], name='state_input')
        action = Input(shape=[self.action_size], name='action_input')
        w1 = Dense(30, activation='relu')(state)
        h1 = Dense(30, activation='linear')(w1)
        a1 = Dense(30, activation='linear')(action)
        h2 = Add()([h1, a1])

        h3 = Dense(30, activation='relu')(h2)
        V = Dense(1, activation='linear')(h3)
        critic = Model(inputs=[state, action], outputs=V)
        critic.compile(loss='mse', optimizer=Adam(lr=0.001))
        # model.summary()
        return critic

    def get_action(self, state):
        self.epsilon -= self.epsilon_decay
        noise = np.zeros([self.action_size])
        action = self.actor.predict(state)[0]
        # add noise to the actor's output,(OU noise)
        noise[0] = max(self.epsilon, 0) * ou_noise(action[0], 0.0, 0.60, 0.30)
        real = action + noise
        return real

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        # make minibatch from replay memory
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        # update critic network
        critic_action_input = self.actor_target.predict(next_states)
        target_q_values = self.critic_target.predict(
            [next_states, critic_action_input])

        targets = np.zeros([self.batch_size, 1])
        for i in range(self.batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.discount_factor * target_q_values[i]

        self.critic.train_on_batch([states, actions], targets)

        # update actor network
        a_for_grad = self.actor.predict(states)
        self.actor_updater([states, states, a_for_grad])

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())


if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # make A2C agent
    agent = DDPGAgent(state_size, action_size)

    print('testing sample agent on torcs')
    global_step = 0

    for e in range(2000):
        done = False
        step = 0
        score = 0
        state = env.reset()

        while not done:
            if agent.render:
                env.render()

            step += 1
            global_step += 1
            action = agent.get_action(np.reshape(state, [1, state_size]))
            next_state, reward, done, info = env.step(action)
            reward /= 10
            score += reward

            agent.append_sample(state, action, reward, next_state, done)

            if global_step > 1000:
                agent.train_model()

            state = next_state

            if done:
                agent.update_target_model()
                print('episode: ', e, ' score: ', score, ' step: ', global_step,
                      ' epsilon: ', agent.epsilon)

        # save the model
        if e % 50 == 0:
            agent.actor.save_weights("./save_model/pendulum_actor.h5")
            agent.critic.save_weights("./save_model/pendulum_critic.h5")
