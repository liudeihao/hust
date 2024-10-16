from time import sleep

import gym

env = gym.make('CartPole-v0')
state = env.reset()

for t in range(1000):
    env.render()
    print(state)
    sleep(0.05)

    action = env.action_space.sample()
    state, reward, done, info, _ = env.step(action)

    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
