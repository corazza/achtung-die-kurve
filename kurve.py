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


def main():
    # env = gym.make("AchtungDieKurveAgainstBot-v1")
    env = gym.make("AchtungPmf-v1")
    env = StateRepeater(env, 4)

    #env = Monitor(env, directory= '/Monitor', force=True)
    render = True

    meanRewards = []
    frame_counter = 0
    input_shape = env.observation_space.shape

    # Build the initial DQN and its target network
    MAIN_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)
    MAIN_TARGET_DQN = build_q_network(env.action_space.n, input_shape, env.history_length, LEARNING_RATE)

    main_replay_buffer = ReplayBuffer(input_shape, size=MEM_SIZE, use_per=USE_PER, history_length=env.history_length)

    main_agent = Agent(MAIN_DQN, MAIN_TARGET_DQN, main_replay_buffer, env.action_space.n,
                       input_shape, batch_size=BATCH_SIZE, use_per=USE_PER, history_length=env.history_length)

    while frame_counter < TOTAL_FRAMES:
        obs, done = env.reset(), False
        repeated_obs = env.repeated_state
        episode_rew = 0
        rew = 0
        while not done:
            if render:
                env.render()

            # action = main_agent.get_action(frame_counter, repeated_obs)
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            repeated_obs = env.repeated_state
            time.sleep(0.1)

            episode_rew += rew
            frame_counter += 1

            main_agent.add_experience(action=action, frame=obs, reward=rew, clip_reward=False, terminal=done)

            if frame_counter % UPDATE_FREQ == 0 and \
                    main_replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                loss, _ = main_agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR,
                                                frame_number=frame_counter,
                                                priority_scale=PRIORITY_SCALE)

            if frame_counter % TARGET_UPDATE_FREQ == 0 and frame_counter > MIN_REPLAY_BUFFER_SIZE:
                main_agent.update_target_network()

        print("Episode reward", episode_rew)
        meanRewards.append(episode_rew)

if __name__ == '__main__':
    main()
