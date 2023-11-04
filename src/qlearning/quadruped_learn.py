#/usr/bin/env python3
from qlearning import QL, ReplayBuffer
from quadruped_env import QuadrupedEnvironment
import numpy 
import time

env = QuadrupedEnvironment()
last_time_steps = numpy.ndarray(0)
qlearn = QL(actions=range(env.action_shape[0]),alpha=0.2, gamma=0.8, epsilon=0.9)

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
    state = ''.join(map(str, observation))

    for i in range(1500):
        # Pick an action based on the current state
        action = qlearn.chooseAction(state)

        # Execute the action and get feedback
        observation, reward, done, info = env.step(action)
        cumulated_reward += reward

        if highest_reward < cumulated_reward:
            highest_reward = cumulated_reward

        nextState = ''.join(map(str, observation))

        qlearn.learn(state, action, reward, nextState)
        if not(done):
            state = nextState
        else:
            last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
            break 

    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    print("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))



'''
def evaluate_training_result(env, agent):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.

    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    total_reward = 0.0
    episodes_to_play = 10
    for i in range(episodes_to_play):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        total_reward += episode_reward
    average_reward = total_reward / episodes_to_play
    return average_reward


def collect_gameplay_experiences(env, agent, buffer):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.

    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    state = env.reset()
    done = False
    while not done:
        action = agent.collect_policy(state)
        next_state, reward, done = env.step(action)
        if done:
            reward = -1.0
        buffer.store_gameplay_experience(state, next_state, reward, action, done)
        state = next_state


def train_model(max_episodes=50000):
    """
    Trains a DQN agent to play the quadruped game by trial and error

    :return: None
    """
    buffer = ReplayBuffer()
    env = QuadrupedEnvironment()
    
    state_shape = env.state_shape
    action_shape = env.action_shape

    agent = QL(state_shape, action_shape)

    for _ in range(100):
        collect_gameplay_experiences(env, agent, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agent, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        avg_reward = evaluate_training_result(env, agent)
        print('Episode : {0}/{1}, Reward : {2}, loss : {3}'.format(episode_cnt, max_episodes, avg_reward, loss[0]))
        if episode_cnt % 20 == 0:
            agent.update_target_network()
    env.close()


train_model()
'''
