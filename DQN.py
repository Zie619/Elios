import os
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
import datetime
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
log_dir = "logs/last/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'dontcount2--3layer-128'
train_summary_writer = tf.summary.create_file_writer(log_dir)

################
################
def OurModel(input_shape, action_space,layer_num):
    X_input = Input(input_shape)
    X = X_input
    if layer_num == 3:
        X = Dense(32, input_shape=input_shape, activation="tanh", kernel_initializer='he_uniform')(X)
        X = Dense(24, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X)
    if layer_num == 5:
        X = Dense(8, input_shape=input_shape, activation="tanh", kernel_initializer='he_uniform')(X)
        X = Dense(10, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(12, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(10, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(8, activation="relu", kernel_initializer='he_uniform')(X)
    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
    model = Model(inputs=X_input, outputs=X)
    model.compile(loss="mse", optimizer=Adam(lr=0.00021, epsilon=0.01),
                  metrics=["accuracy"])
    return model


class DQNAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        # by default, CartPole-v1 has max episode steps = 500
        self.env._max_episode_steps = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.EPISODES = 10000
        self.memory = deque(maxlen=2000)

        self.gamma = 0.97  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 256
        self.train_start = 1000
        self.layer_num = 3
        self.Save_Path = 'Models'
        self.scores, self.episodes, self.average = [], [], []



        print("-------------DQN------------")
        self.Model_name = os.path.join(self.Save_Path, "DQN_" + self.env_name + ".h5")

        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size,layer_num=self.layer_num)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space=self.action_size,layer_num=self.layer_num)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)


        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        history=self.model.fit(state, target, batch_size=self.batch_size, epochs=1,verbose=0,  )
        loss = history.history['loss'][0]
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=i)



    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)


    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores) / len(self.scores))
        dqn = 'DQN_'
        return str(self.average[-1])[:5]

    def run(self):
        steps_list = []
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every step update target model

                    # every episode, plot the result
                    average = self.PlotModel(i, e)

                    print("episode: {}/{}, score: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i, self.epsilon,
                                                                                    average))
                    with train_summary_writer.as_default():
                        tf.summary.scalar('reward', i, step=e)
                    steps_list.append(i)
                    runningMean = np.mean(steps_list[-100:])
                    print(runningMean)
                    if runningMean > 475:
                        print("you made it!")
                        print(runningMean)
                        break
                self.replay()


    def test(self):
        self.load("cartpole-ddqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                #self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    agent = DQNAgent(env_name)
    agent.run()
    #agent.test()
