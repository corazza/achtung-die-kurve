import gym
import gym_achtung
import pickle
import os
import time
import IPython
import tensorflow as tf
import sys

from agent import RandomAgent
from agent import build_q_network
from agent import ReplayBuffer
from agent import Agent
from const import *
from state_repeater import StateRepeater
import pygame

def main(path):
    tf.enable_eager_execution()
    env = gym.make("AchtungPmf-v1", use_pygame=True, other_human=True)
    env = StateRepeater(env, 4)

    episode_rewards = []
    frame_counter = 0
    input_shape = env.observation_space.shape

    MAIN_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)
    MAIN_TARGET_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)
    MAIN_DQN.load_weights(path+'/dqn.h5')
    MAIN_TARGET_DQN.load_weights(path+'/target_dqn.h5')

    main_agent = Agent(MAIN_DQN, MAIN_TARGET_DQN, env.action_space.n,
                       input_shape, batch_size=BATCH_SIZE, history_length=env.history_length)

    while frame_counter < TOTAL_FRAMES:
        obs, info = env.reset()
        done = False
        repeated_obs = env.repeated_state
        other_repeated_obs = env.other_repeated_state
        episode_rew = 0
        other_episode_rew = 0

        while not done:
            env.render()

            action = main_agent.get_action(frame_counter, repeated_obs, evaluation=True)

            obs, rew, done, info = env.step(action)

            other_obs = info['other_state']
            other_rew = info['other_reward']
            repeated_obs = env.repeated_state
            other_repeated_obs = env.other_repeated_state
            episode_rew += rew
            other_episode_rew += other_rew
            frame_counter += 1

            time.sleep(0.05)

        print("Episode reward", episode_rew)
        episode_rewards.append(episode_rew)

if __name__ == '__main__':
    load_id = int(sys.argv[1])
    path = f'{SAVE_PATH}/save-{str(load_id).zfill(8)}'
    main(path)
