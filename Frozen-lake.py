'''
First Assingment - Frozen lake
Names: Eliad Shahar, Ido Rabia
'''

#importing the packages for the assingment:
import gym
import numpy as np
import time
import matplotlib.pyplot as plt


#initilize Q and setting the env:
env=gym.make('FrozenLake-v0')
Q = np.ones((env.observation_space.n, env.action_space.n))

#initizlize hyper-parameters:
learning_rate = 0.9
discount_factor = 0.9
epsilon = 0.6
number_of_episodes = 700
max_steps=100
Q_at_range=[]
reward_per_episode=[]
steps_at_episode=[]
list_for_graphs=[]

#def how to choose an acton using the method of epsilon greedy:
def choose_action(s, epsilon):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[s, :])
    return action

#def the learn function as shown in class:
def learn(state, action, state2, reward):
    predict = Q[state, action]
    Q_max = np.max(Q[state2, :])
    Q[state, action] += + learning_rate*(reward + discount_factor*Q_max-predict)

#The Q-learning algorithm:
    '''
    state2 is the next state for Q[state,action]
    reward - reward returned from the state2 [-1,0,1]
    fall- if the player died or managed to go to the next step ; 1 for go , 0 for dead.
    info- info for debugging
    '''
for k in range(number_of_episodes):
    state = env.reset()  # setting up the state for the episode
    number_of_steps=0
    for t in range(0,max_steps):
        env.render()
        action = choose_action(state, epsilon)
        state2, reward, done, info = env.step(action)
        learn(state, action, state2, reward)
        epsilon=0.9*epsilon
        state = state2
        if done:
            break
        time.sleep(0.1)
    # printing the values - remove later:
    print(k)
    print(t)
    print(reward)
    print(Q)

    # appending k to a list for later use, if fall steps=100 if not save the steps
    if reward == 1:
        steps_at_episode.append(k)
    if reward == 0:
        steps_at_episode.append(100)
    if k == 500 or k == 2000 or k == number_of_episodes:  #append Q at specific k
        print("yay middle")
        Q_at_range.append(Q)
    reward_per_episode.append(reward)
    list_for_graphs.append(k)
#requierd for anwser:
np.add.reduceat(steps_at_episode, np.arange(0, len(steps_at_episode), 100)) # makes a list for the sum of the steps per 100 episodes

#now we need graph plotting:

'''
Hyper-parameters values that were used for the final solution.
- Q-value table after 500 steps, 2000 steps and the final table as a colormap
(with the values).
- Plot of the reward per episode.
- Plot of the average number of steps to the goal over last 100 episodes (plot
every 100 episodes). If agent didn't get to the goal, the number of steps of
the episode will be set to 100. 
'''
# reward vs episode graph:
x = list_for_graphs
y = reward_per_episode
plt.plot(x, y, 'ro')
plt.xlabel('Number of episodes ')
plt.ylabel('reward')
plt.title('Reward per episode')
plt.show()


x = steps_at_episode
y = (0, 5000, 100)
plt.plot(x, y)
plt.xlabel('Number of episodes ')
plt.ylabel('reward')
plt.title('Reward per episode')
plt.show()