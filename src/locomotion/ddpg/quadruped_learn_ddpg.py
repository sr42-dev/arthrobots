#!/usr/bin/env python3
from __future__ import print_function
from quadruped_env import QuadrupedEnvironment
from ddpg import OUNoise, DDPG
import numpy as np
import os
import time

# a script to initiate training of the quadruped robot on any currently published simulation environment

# declaration of the environment
env = QuadrupedEnvironment()    # functions - init, jsp_callback, normalize_js, imu_sub_callback
                                # reset, step,
                                
# configuration of model parameters
state_shape = env.state_shape
action_shape = env.action_shape

# configuration of the model
agent = DDPG(state_shape,action_shape,batch_size=128,gamma=0.995,tau=0.001, actor_lr=0.0001, critic_lr=0.001, use_layer_norm=True)

print('DDPG agent configured')

# training parameters
max_episode = 5
tot_rewards = []

# environment reset
print('env reset')
observation, done = env.reset()

# state and action space configuration
action = agent.act(observation)
observation, reward, done = env.step(action)
noise_sigma = 0.001
save_cutoff = 1
cutoff_count = 0
save_count = 0
curr_highest_eps_reward = -100.0

epoch_count = []

# training loop
for i in range(max_episode):

    epoch_count.append(i)

    # introducing noise
    if i % 100 == 0 and noise_sigma>0.03:
        agent.noise = OUNoise(agent.nb_actions,sigma=noise_sigma)
        noise_sigma /= 2.0
    
    # iteration loop
    step_num = 0
    while done == False:
        step_num += 1
        action = agent.step(observation, reward, done)
        observation, reward, done = env.step(action)
        print('************')
        print('action:', action)
        print('observation', observation)
        print('************')
        print('reward:',reward,'episode:', i, 'step:',step_num,'curr high eps reward:',curr_highest_eps_reward, 'saved:',save_count, 'cutoff count:', cutoff_count)
    
    # take step
    action, eps_reward = agent.step(observation, reward, done)

    # print("Checking for lock file...")
    # while os.path.exists('rewards.lock'):
    #     time.sleep(0.01)
    # print("Lock file not found, continuing...")

    # with open('rewards.lock', 'w'):
    #     pass

    with open('rewards.txt', 'r+') as f:
        existing_rewards = f.read()
        total_reward = sum(float(r) for r in existing_rewards.split()) + eps_reward
        f.seek(0)
        f.write(str(total_reward) + '\n')

    # os.remove('rewards.lock')

    with open('rewards.txt', 'r') as f:
        rewards = [float(line) for line in f if line.strip()]
        averaged_reward = sum(rewards) / len(rewards) / 1

    tot_rewards.append(averaged_reward)

    # cutoff conditions
    if eps_reward > curr_highest_eps_reward:
        cutoff_count += 1
        curr_highest_eps_reward = eps_reward

    if cutoff_count >= save_cutoff:
        save_count += 1
        print('saving_model at episode:',i)
        agent.save_model()
        agent.save_memory()
        cutoff_count = 0

    # reset environment after cutoff
    observation, done = env.reset()

# save rewards
np.save('eps_rewards',tot_rewards)

# plot rewards
import matplotlib.pyplot as plt
plt.plot(epoch_count,tot_rewards)
plt.xlabel('Epochs')
plt.ylabel('Total Rewards')
plt.show()
