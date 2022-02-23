import gym
import gym_achtung
import pickle
import os
import time
import IPython

from agent import RandomAgent
from agent import build_q_network
from agent import ReplayBuffer
from agent import Agent
from const import *
from state_repeater import StateRepeater
import tensorflow as tf
import numpy as np

render = False


def main():
    tf.enable_eager_execution()

    # IPython.embed()
    writer = tf.contrib.summary.create_file_writer(TENSORBOARD_DIR)

    env = gym.make("AchtungPmf-v1", use_pygame=render)
    env = StateRepeater(env, 4)
    input_shape = env.observation_space.shape

    MAIN_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)
    MAIN_TARGET_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)

    main_replay_buffer = ReplayBuffer(input_shape, size=MEM_SIZE, use_per=USE_PER, history_length=env.history_length)

    main_agent = Agent(MAIN_DQN, MAIN_TARGET_DQN, main_replay_buffer, env.action_space.n,
                       input_shape, batch_size=BATCH_SIZE, use_per=USE_PER, history_length=env.history_length)

    frame_number = 0
    rewards = []
    loss_list = []
    model_gen = []

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                # Training
                epoch_frame = 0

                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    obs, terminal = env.reset(), False
                    repeated_obs = env.repeated_state
                    life_lost = False
                    terminal = False
                    episode_reward_sum = 0

                    while not terminal:
                        action = main_agent.get_action(frame_number, repeated_obs)
                        obs, rew, terminal, _ = env.step(action)
                        repeated_obs = env.repeated_state
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += rew
                        main_agent.add_experience(action=action, frame=obs, reward=rew, clip_reward=False, terminal=terminal)

                        if frame_number % UPDATE_FREQ == 0 and main_replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = main_agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                            frame_number=frame_number,
                                                        priority_scale=PRIORITY_SCALE)

                        if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            main_agent.update_target_network()

                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.contrib.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.contrib.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print(
                            f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames for tensorboard
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0

                for _ in range(EVAL_LENGTH):
                    if terminal:
                        obs, terminal = env.reset(), False
                        repeated_obs = env.repeated_state
                        life_lost = False
                        episode_reward_sum = 0

                    action = main_agent.get_action(frame_number, repeated_obs, evaluation=True)

                    obs, rew, terminal, _ = env.step(action)
                    repeated_obs = env.repeated_state
                    evaluate_frame_number += 1
                    episode_reward_sum += rew

                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum

                # Print score and write to tensorboard
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.contrib.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # Save model
                if len(rewards) > 300 and SAVE_PATH is not None:
                    main_agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                                           rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            main_agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number,
                                   rewards=rewards, loss_list=loss_list)
            print('Saved.')

if __name__ == '__main__':
    main()
