import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import random
from collections import deque, defaultdict

from matplotlib import pyplot as plt

import game.wrapped_flappy_bird as game

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations

OBSERVE = 100
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1

REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
FRAME_PER_ACTION = 1

output_dir = 'output'


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=8, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(576, 256)
        self.fc2 = nn.Linear(256, ACTIONS)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DQN:
    def __init__(self):
        self.discount = 0.9
        self.Q = Net()
        self.target_Q = Net()
        self.target_Q.load_state_dict(self.Q.state_dict())

    def get_action(self, state):
        qvals = self.Q(state)

        return qvals.argmax()

    def compute_loss(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        a_indices = torch.argmax(a_batch, dim=1, keepdim=True)
        # 使用动作索引来聚集Q值
        qvals = self.Q(s_batch).gather(1, a_indices)

        # qvals = self.Q(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        next_qvals, _ = self.target_Q(next_s_batch).detach().max(dim=1)
        expected_qvals = r_batch + self.discount * (1 - d_batch) * next_qvals
        loss = F.mse_loss(qvals.squeeze(), expected_qvals)
        return loss

    def update_target(self):
        self.target_Q.load_state_dict(self.Q.state_dict())


def soft_update(target, source, tau=0.01):
    """
    update target by target = tau * source + (1 - tau) * target.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


@dataclass
class ReplayBuffer:
    maxsize: int
    size: int = 0
    state: list = field(default_factory=list)
    action: list = field(default_factory=list)
    next_state: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    done: list = field(default_factory=list)

    def push(self, state, action, reward, done, next_state):
        if self.size < self.maxsize:
            self.state.append(state)
            self.action.append(action)
            self.reward.append(reward)
            self.done.append(done)
            self.next_state.append(next_state)
        else:
            position = self.size % self.maxsize
            self.state[position] = state
            self.action[position] = action
            self.reward[position] = reward
            self.done[position] = done
            self.next_state[position] = next_state
        self.size += 1

    def sample(self, batch_size):
        total_number = self.size if self.size < self.maxsize else self.maxsize
        indices = np.random.randint(total_number, size=batch_size)
        state = [self.state[i] for i in indices]
        action = [self.action[i] for i in indices]
        reward = [self.reward[i] for i in indices]
        done = [self.done[i] for i in indices]
        next_state = [self.next_state[i] for i in indices]

        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.int64)
        next_state = np.array(next_state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        next_state = torch.from_numpy(next_state)
        reward = torch.from_numpy(reward)
        done = torch.from_numpy(done)

        return state, action, reward, done, next_state


def preprocess(observation):
    # 预处理观测数据。
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)

    # 调整shape为(1, 80, 80)
    return np.reshape(observation, (1, 80, 80))


def train_dqn(agent):
    game_state = game.GameState()
    replay_buffer = ReplayBuffer(REPLAY_MEMORY)

    optimizer = torch.optim.Adam(agent.Q.parameters())
    optimizer.zero_grad()

    state = init_state(game_state)

    epsilon = INITIAL_EPSILON

    episode_reward = 0
    episode_length = 0
    max_episode_reward = -float("inf")

    max_step = 1000000

    log = defaultdict(list)
    log["loss"].append(0)

    episode = 0

    for i in range(max_step):

        # 每隔FRAME_PER_ACTION帧选择一次动作
        action = np.zeros([ACTIONS])
        action[0] = 1
        if i % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                # 随机选择动作
                action_index = random.randrange(ACTIONS)
            else:
                # 选择使Q值最大的动作
                best_action_index = agent.get_action(torch.from_numpy(state))
                action_index = torch.argmax(best_action_index).item()
            action[action_index] = 1

        # 执行动作
        next_snapshot, reward, terminated = game_state.frame_step(action)
        # 获取下一状态
        x_t1 = preprocess(next_snapshot)
        x_t1_expanded = x_t1[np.newaxis, :, :]
        state_appended = np.concatenate((state, x_t1_expanded), axis=1)
        next_state = state_appended[:, 1:, :]
        # 存储经验
        replay_buffer.push(state.squeeze(), action, reward, terminated, next_state.squeeze())
        state = next_state

        episode_reward += reward
        episode_length += 1

        if terminated is True:
            log["episode_reward"].append(episode_reward)
            log["episode_length"].append(episode_length)
            episode = episode + 1

            print(f"episode={episode}, i={i}, reward={episode_reward:.1f}, length={episode_length}, max_reward={max_episode_reward:.1f}, loss={log['loss'][-1]:.1e}, epsilon={epsilon:.3f}")

            # 如果得分更高，保存模型。
            if episode_reward > max_episode_reward:
                save_path = os.path.join(output_dir, "model.bin")
                torch.save(agent.Q.state_dict(), save_path)
                max_episode_reward = episode_reward

            episode_reward = 0

        # 训练网络
        if i > OBSERVE:
            bs, ba, br, bt, bns = replay_buffer.sample(BATCH)

            # 计算损失
            loss = agent.compute_loss(bs, ba, br, bt, bns)

            # 梯度下降
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            log["loss"].append(loss.item())

            # 更新target网络
            soft_update(agent.target_Q, agent.Q)

        if epsilon > FINAL_EPSILON and i > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    # 3. 画图。
    plt.plot(log["loss"])
    plt.yscale("log")
    plt.savefig(f"{output_dir}/loss.png", bbox_inches="tight")
    plt.close()

    plt.plot(np.cumsum(log["episode_length"]), log["episode_reward"])
    plt.savefig(f"{output_dir}/episode_reward.png", bbox_inches="tight")
    plt.close()

def init_state(game_state):
    # 初始化环境
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # x是当前的画面截图
    snapshot, _, _ = game_state.frame_step(do_nothing)
    x_t = preprocess(snapshot)
    # state是最近的四张截图，作为状态
    state = np.stack((x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, x_t, ), axis=1, dtype=np.float32)
    return state

def eval_dqn(agent):
    game_state = game.GameState()

    model_path = os.path.join(output_dir, "model.bin")
    agent.Q.load_state_dict(torch.load(model_path))

    state = init_state(game_state)

    log = defaultdict(list)
    log["loss"].append(0)

    episode = 0
    episode_reward = 0
    episode_length = 0
    max_episode_reward = -float("inf")

    for i in range(1000):

        # 每隔FRAME_PER_ACTION帧选择一次动作
        action = np.zeros([ACTIONS])
        action[0] = 1
        if i % FRAME_PER_ACTION == 0:
            # 选择使Q值最大的动作
            best_action_index = agent.get_action(torch.from_numpy(state))
            action_index = torch.argmax(best_action_index).item()
            action[action_index] = 1

        # 执行动作
        next_snapshot, reward, terminated = game_state.frame_step(action)
        # 获取下一状态
        x_t1 = preprocess(next_snapshot)
        x_t1_expanded = x_t1[np.newaxis, :, :]
        state_appended = np.concatenate((state, x_t1_expanded), axis=1)
        next_state = state_appended[:, 1:, :]
        # 存储经验
        state = next_state

        episode_reward += reward
        episode_length += 1

        if terminated is True:
            log["episode_reward"].append(episode_reward)
            log["episode_length"].append(episode_length)
            episode = episode + 1

            print(f"episode={episode}, i={i}, reward={episode_reward:.0f}, length={episode_length}, max_reward={max_episode_reward}, loss={log['loss'][-1]:.1e}")
            episode_reward = 0


def main():
    agent = DQN()

    train_dqn(agent)


if __name__ == "__main__":
    main()
