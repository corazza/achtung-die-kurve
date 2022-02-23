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

def main(save_id):
    tf.enable_eager_execution()
    env = gym.make("AchtungPmf-v1", use_pygame=RENDER_TRAINING)
    env = StateRepeater(env, HISTORY_LENGTH)

    episode_rewards = []
    frame_counter = 0
    input_shape = env.observation_space.shape

    MAIN_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)
    MAIN_TARGET_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)

    main_replay_buffer = ReplayBuffer(input_shape, size=MEM_SIZE, use_per=USE_PER, history_length=env.history_length)

    main_agent = Agent(MAIN_DQN, MAIN_TARGET_DQN, env.action_space.n,
                       input_shape, batch_size=BATCH_SIZE, use_per=USE_PER, history_length=env.history_length)

    OTHER_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)
    OTHER_TARGET_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)

    other_replay_buffer = ReplayBuffer(input_shape, size=MEM_SIZE, use_per=USE_PER, history_length=env.history_length)

    other_agent = Agent(OTHER_DQN, OTHER_TARGET_DQN, env.action_space.n,
                       input_shape, batch_size=BATCH_SIZE, use_per=USE_PER, history_length=env.history_length)

    while frame_counter < TOTAL_FRAMES:
        obs, info = env.reset()
        done = False
        repeated_obs = env.repeated_state
        other_repeated_obs = env.other_repeated_state
        episode_rew = 0
        other_episode_rew = 0

        while not done:
            if RENDER_TRAINING:
                env.render()

            action = main_agent.get_action(frame_counter, repeated_obs)
            other_action = other_agent.get_action(frame_counter, other_repeated_obs)

            env.set_action_other(other_action)
            obs, rew, done, info = env.step(action)

            other_obs = info['other_state']
            other_rew = info['other_reward']
            repeated_obs = env.repeated_state
            other_repeated_obs = env.other_repeated_state
            episode_rew += rew
            other_episode_rew += other_rew
            frame_counter += 1

            main_replay_buffer.add_experience(action=action, frame=obs, reward=rew, clip_reward=False, terminal=done)
            other_replay_buffer.add_experience(action=other_action, frame=other_obs, reward=other_rew, clip_reward=False, terminal=done)

            if frame_counter % UPDATE_FREQ == 0 and main_replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                loss, _ = main_agent.learn(main_replay_buffer, BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                frame_number=frame_counter,
                                            priority_scale=PRIORITY_SCALE)
                loss, _ = other_agent.learn(other_replay_buffer, BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                frame_number=frame_counter,
                                            priority_scale=PRIORITY_SCALE)

            if frame_counter % TARGET_UPDATE_FREQ == 0 and frame_counter > MIN_REPLAY_BUFFER_SIZE:
                main_agent.update_target_network()
                other_agent.update_target_network()
                # IPython.embed()

        percent_done = int(float(frame_counter)/TOTAL_FRAMES * 100)
        print(f'episode_reward={episode_rew}, other_reward={other_episode_rew}, step={frame_counter}/{TOTAL_FRAMES} ({percent_done}%), eps={main_agent.calc_epsilon(frame_counter)}')
        episode_rewards.append(episode_rew)

    main_agent.save(path, episode_rewards=episode_rewards)

if __name__ == '__main__':
    save_id = int(sys.argv[1])
    path = f'{SAVE_PATH}/save-{str(save_id).zfill(8)}'
    main(path)
