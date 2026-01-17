import numpy as np
import gymnasium as gym
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
from gymnasium import spaces
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.viz.figures_visualize import draw_learning_curve
from config.Config import CONFIG
from src.uti.utilities import clear_folder, get_path_variables
from src.ani.animation_visualize import animate_position_2d, animate_position_2d_img
from src.data.run_context import RunContext, RunRewards

# 检查环境中是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义Q神经网络
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# 样本重放
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)
    
# 动作选择
def epsilon_greedy_dqn(epsilon, q_net, s_onehot, n_actions):
    # 随机选择
    if np.random.rand() < epsilon:
        # Sample a random action
        return env.action_space.sample(env)
    # 根据神经网络正向推理得到贪婪动作
    with torch.no_grad():
        x = torch.tensor(
            s_onehot, dtype=torch.float32, device=device
        ).unsqueeze(0)
        q_values = q_net(x)
        return int(torch.argmax(q_values, dim=1).item())

# 定义物理环境    
class MatrixWorld(gym.Env):
    def __init__(self):
        print("MatrixWorld __int__ called")
        # 定义世界空间大小
        self.width = CONFIG["environment"]["width"]
        self.height = CONFIG["environment"]["height"]
        # 定义世界空间最小坐标
        self.min_x = 0
        self.min_y = 0
        self.min_pos = [self.min_x, self.min_y]
        # 定义世界空间最大坐标
        self.max_x = self.width - 1
        self.max_y = self.height - 1
        self.max_pos = [self.max_x, self.max_y]    
        # 智能体在世界中的位置状态信息
        self.pos = self.min_pos
        # 动作空间：两个动作 0:up / 1:down / 2:left / 3:right
        self.action_space = spaces.Discrete(4)
        # 观测空间：采用多重离散方式
        self.observation_space = spaces.MultiDiscrete([self.width, self.height])
        # 定义训练步数约束
        self.min_step = CONFIG["environment"]["min_steps"]
        self.max_step = CONFIG["environment"]["max_steps"]
        # 初始化奖励
        self.rrwds = None

    def set_rewards(self, rrwds: RunRewards):
        if not isinstance(rrwds, RunRewards):
            raise TypeError("rrwds must be RunRewards")
        self.rrwds = rrwds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 一局开始：把智能体放回起点
        self.pos = self.min_pos.copy()
        # 将初始化位置转换到观测空间
        obs = np.array(self.pos, dtype=np.int64)
        # 初始化训练步数累计值
        self.steps = 0
        # 额外信息反馈出口，非Agent相关学习信息，可用于调试
        info = {}
        
        return obs, info

    def step(self, action): #0:up / 1:down / 2:left / 3:right
        # 取出 x, y 现有坐标
        x, y = self.pos
        old_x = x
        old_y = y
        # 如果向上，y + 1
        if (action == 0):
            y += 1
        # 如果向下，y - 1
        elif (action == 1):
            y -= 1
        # 如果左，x - 1
        elif (action == 2):
            x -= 1
        # 如果向右，x + 1
        elif (action == 3):
            x += 1
        # 边界裁剪，如果Y当前位置超出0或4，拉回到0-4空间内
        y = int(np.clip(y, self.min_y, self.max_y))
        # 边界裁剪，如果X当前位置超出0或4，拉回到0-4空间内
        x = int(np.clip(x, self.min_x, self.max_x))
        # 最新坐标位置传回
        self.pos = [x, y]
        # 撞墙判断
        hit_wall = False
        if x == old_x and y == old_y:
            hit_wall = True
        # 抵达终点时标记任务结束，最大化奖励
        if self.pos == self.max_pos:
            reward = self.rrwds.max_pos_reward
            terminated = True
        # 未抵达中间时标记任务继续，惩罚    
        else:
            reward = self.rrwds.step_reward
            terminated = False
            if hit_wall and self.rrwds.hit_wall_enable:
                reward += self.rrwds.hit_wall_reward
        # 将位置信息转换到观测空间
        obs = np.array(self.pos, dtype=np.int64)
        # 当步数远超预期时，中断训练
        self.steps += 1 
        if self.steps >= self.max_step:
            truncated = True 
        else:
            truncated = False 
        info = {}

        return obs, reward, terminated, truncated, info   
    

# 环境观测转换为状态
def obs_to_state(obs, width):
    x, y = obs
    return y * width + x

# 状态转换为one-hot
def state_to_onehot(s: int, num_states: int) -> np.ndarray:
    v = np.zeros(num_states, dtype=np.float32)
    v[s] = 1.0
    return v

# 每轮训练状态初始化
def train_episode_initilize(env):
    # 每轮训练初始化观测环境
    obs, info = env.reset()
    # 每轮训练初始化状态
    s = obs_to_state(obs, env.width)
    # 每轮训练初始化结束判断
    done = False
    # 每轮训练初始化步数统计
    step_count = 0

    return obs, info, s, done, step_count    

def evaluate_performance(env, q_net, n_states):
    # 每轮训练环境与变量初始化
    obs, info, s, done, eval_steps = train_episode_initilize(env)
    # 更新positions记录
    x, y = obs
    positions = []
    positions.append((int(x), int(y)))
    # 更新actions记录
    actions = []
    while not done:
        # 根据神经网络正向推理得到贪婪动作
        with torch.no_grad():
            x = torch.tensor(
                state_to_onehot(s, n_states), dtype=torch.float32, device=device
            ).unsqueeze(0)
            q_values = q_net(x)
        a = int(torch.argmax(q_values, dim=1).item())
        # 取得最优状态对应动作
        actions.append(a)            
        # 执行动作获得反馈
        next_obs, reward, terminated, truncated, info = env.step(a)
        # 判断是否中止或结束
        done = terminated or truncated
        # 最新状态转换
        s_next = obs_to_state(next_obs, env.width)
        # 状态推进
        s = s_next
        # 步数累计
        eval_steps += 1
        # 下一步坐标存入positions
        x, y = next_obs
        positions.append((int(x), int(y)))
        
    return eval_steps, positions, actions


def train_dqn(env, rctx, q_net, replay_buffer, optimizer):

    # 计算状态空间
    n_states = env.width * env.height
    # 计算动作空间
    n_actions = env.action_space.n
    # 初始化训练次数
    episodes = CONFIG["training"]["episodes"]
    # 初始化学习率 alpha
    alpha = CONFIG["algorithm"]["alpha"]
    # 初始化折扣因子 gamma，表示未来奖励这算在现在值多少
    gamma = CONFIG["algorithm"]["gamma"]
    # 初始化DQN学习率
    batch_size = CONFIG["algorithm"]["batch_size"]
    # 初始化随机因子相关参数
    epsilon = CONFIG["exploration"]["epsilon_start"]    
    epsilon_decay = CONFIG["exploration"]["epsilon_decay"]
    epsilon_min = CONFIG["exploration"]["epsilon_min"]        
    # 学习效果统计
    # episode_steps = []
    # 创建tensorBoard日志，位于runs/
    log_dir=CONFIG["paths"]["log_dir"]
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = Path(log_dir) / f"{rctx.execute_stem}_{run_time}"
    writer = SummaryWriter(log_dir)
    # 初始化动画相关设定
    save_gif = CONFIG["animation"]["save_gif"]
    save_gif_ep = CONFIG["animation"]["save_gif_ep"]
    save_gif_ep_start = CONFIG["animation"]["save_git_ep_start"]
    agent_img_dir = CONFIG["animation"]["agent_img_dir"]
    des_img_dir = CONFIG["animation"]["des_img_dir"]
    end_img_path = CONFIG["animation"]["end_img_path"]
    ending_time = CONFIG["animation"]["ending_time"]
    # 初始化rewards相关设定
    max_pos_reward = CONFIG["rewards"]["max_pos_reward"]
    step_reward = CONFIG["rewards"]["step_reward"]
    hit_wall_enable = CONFIG["rewards"]["hit_wall_enable"]
    hit_wall_reward = CONFIG["rewards"]["hit_wall_reward"]
    rrwds = RunRewards(
        max_pos_reward = max_pos_reward,
        step_reward = step_reward,
        hit_wall_enable = hit_wall_enable,
        hit_wall_reward = hit_wall_reward
    )
    env.set_rewards(rrwds)
    # 进入训练，训练次数=episodes
    for ep in range(episodes):
        # 每轮训练初始化观测环境
        print("The episode >>>>>>>>> ", ep)
        obs, info, s, done, step_count = train_episode_initilize(env)

        # 开始执行直到抵达终点或任务中断
        while not done:
            # 选择一个动作
            s_onehot = state_to_onehot(s, n_states)
            a = epsilon_greedy_dqn(epsilon, q_net, s_onehot, n_actions) 
            # 执行后得到反馈
            next_obs, reward, terminated, truncated, info = env.step(a)
            # 将观测空间传回新状态位置置换
            s_next = obs_to_state(next_obs, env.width)
            # 是否抵达终点或被打断
            done = terminated or truncated
            # 将本次样本记录到回放样本库
            replay_buffer.add(s, a, reward, s_next, done)
            # 当样本数量足够时，触发神经网络学习
            if len(replay_buffer) >= batch_size:
                # 在回放样本库采样
                batch = replay_buffer.sample(batch_size)
                # 采样数据重构为 当前状态，状态对应动作，奖励，下一状态，完成状态
                S, A, R, S2, D = zip(*batch)
                # 转换{当前状态}到神经网络可加载数据形式
                S_np = np.array([state_to_onehot(s, n_states) for s in S],
                                 dtype=np.float32
                                 )
                S = torch.from_numpy(S_np).to(device)
                # 转换{下一状态}到神经网络可加载数据形式
                S2_np = np.array([state_to_onehot(s2, n_states) for s2 in S2],
                                 dtype=np.float32
                                 )
                S2 = torch.from_numpy(S2_np).to(device)                
                # 转换{动作}到神经网络可加载数据形式
                A = torch.tensor(A, dtype=torch.int64, device=device).unsqueeze(1)                
                # 转换{奖励}到神经网络可加载数据形式
                R = torch.tensor(R, dtype=torch.float32, device=device)
                # 转换{是否完成}到神经网络可加载数据形式
                D = torch.tensor(D, dtype=torch.float32, device=device)
                # 以当前状态作为输入，神经网络正向推理得到所有分数，从分数中取得实际执行的动作得分
                q_sa = q_net(S).gather(1, A).squeeze(1)
                # 用下一时刻作为输入，神经网络正向推理得到所有得分，从分数中取得得分最大的动作
                with torch.no_grad():
                    q_next_max = q_net(S2).max(dim=1).values
                    y = R + gamma * (1.0 - D) * q_next_max
                # 通过当前时刻动作得分与下一时刻动作得分计算loss，逼近
                loss = F.smooth_l1_loss(q_sa, y)
                # 清空梯度
                optimizer.zero_grad()
                # 反向传递
                loss.backward()
                # 优化参数
                optimizer.step()

            # 状态推进
            s = s_next
            # 步数累计
            step_count += 1
            if (done): print("This episode is completed.")

        # 随机概率衰减
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay            
        # 将每轮ep步数加入tensorBoard
        writer.add_scalar("Episode/Steps", step_count, ep)
        # 将每轮ep随机率加入tensorBoard
        writer.add_scalar("Episode/Epsilon", epsilon, ep)
        # 评估学习效果，不学习不更新Q表
        eval_steps, positions, actions = evaluate_performance(env, q_net, n_states)
        # 将每轮ep学习效果加入tensorBoard
        writer.add_scalar("Eval/Steps", eval_steps, ep)
        # 满足条件触发动画制作
        if save_gif and ep >= save_gif_ep_start and ep % save_gif_ep == 0: 
            run_time = datetime.now().strftime("%Y%m%d%H%M%S")
            ani_path = log_dir / f"{rctx.execute_stem}_{ep}_{run_time}.gif"
            # ani = animate_position_2d(env, positions, actions, ani_path)  # 无定制化动画
            ani = animate_position_2d_img(env, positions, actions, ani_path, agent_img_dir, des_img_dir, end_img_path, terminated, ending_time) # 定制化动画版本
            
    # 关闭tensorBoard文件写入
    writer.close()    

if __name__ == "__main__":

    # 初始化相关路径变量
    project_root, execute_filename, execute_stem = get_path_variables()
    rctx = RunContext(
        project_root = project_root,
        execute_file = execute_filename,
        execute_stem = execute_stem    
    )
    # 初始化环境对象
    env = MatrixWorld()
    # 初始化神经网络对象
    q_net = QNet(env.width * env.height, env.action_space.n)
    # 初始化样本重放对象
    replay_buffer = ReplayBuffer()
    # 初始化优化器对象
    optimizer = optim.Adam(q_net.parameters(), lr=CONFIG["algorithm"]["lr"])

    # 开始强化学习训练
    train_dqn(env, rctx, q_net, replay_buffer, optimizer)

    print("done")
