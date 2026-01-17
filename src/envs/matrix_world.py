import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config.Config import CONFIG
from src.data.run_context import RunRewards

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
        # 定义世界起点
        self.start_pos = self.min_pos    
        # 定义世界终点
        self.goal_pos = self.max_pos
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
        self.pos = self.start_pos.copy()
        # 随机设定终点位置
        while True:
            gx = self.np_random.integers(self.min_x, self.max_x + 1)
            gy = self.np_random.integers(self.min_y, self.max_y + 1)
            if [gx, gy] != self.pos:
                break
        self.goal_pos = [int(gx), int(gy)]
        # 将初始化位置与终点位置转换到观测空间
        obs = np.concatenate([self.pos, self.goal_pos]).astype(np.int64)
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
        obs = np.concatenate([self.pos, self.goal_pos]).astype(np.int64)
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