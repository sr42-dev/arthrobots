#/usr/bin/env python3
from qlearning import QL, ReplayBuffer
from quadruped_env import QuadrupedEnvironment
import numpy 
import time
import tensorflow as tf
import pickle

# saver = tf.train.Saver()
env = QuadrupedEnvironment()
last_time_steps = numpy.ndarray(0)
qlearn = QL(alpha=0.2, gamma=0.8, epsilon=0.9)

initial_epsilon = qlearn.epsilon
epsilon_discount = 0.9986
start_time = time.time()
total_episodes = 10000
highest_reward = 0

for x in range(total_episodes):

    done = False
    cumulated_reward = 0
    observation, _ = env.reset()
    if qlearn.epsilon > 0.05:
        qlearn.epsilon *= epsilon_discount
    state = observation

    for i in range(1500):

        # Pick an action based on the current state
        action = qlearn.chooseAction(state)

        # Execute the action and get feedback
        observation, reward, done = env.step(action)
        cumulated_reward += reward

        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        nextState = observation

        qlearn.learn(state, action, reward, nextState)
        if not(done):
            state = nextState
        else:
            last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
            break 
    
    with open('qlearn_model.pkl', 'wb') as f:
        pickle.dump(qlearn.q, f)

    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    print("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
