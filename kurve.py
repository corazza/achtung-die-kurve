import gym
import gym_achtung
import pickle
import os
import time

from agent import RandomAgent

def main():
    env = gym.make("AchtungDieKurveAgainstBot-v1")
    agent = RandomAgent(env.action_space)

    #env = Monitor(env, directory= '/Monitor', force=True)
    render = True

    meanRewards = []
    numberOfEpisodes = 100
    eval = 0

    while eval < numberOfEpisodes:
        eval += 1
        obs, done = env.reset(), False
        episode_rew = 0
        rew = 0
        while not done:
            if render:
                env.render()

            action = agent.act(obs, rew, done)
            obs, rew, done, _ = env.step(action)
            time.sleep(0.1)

            episode_rew += rew

        print("Episode reward", episode_rew)
        meanRewards.append(episode_rew)

if __name__ == '__main__':
    main()
