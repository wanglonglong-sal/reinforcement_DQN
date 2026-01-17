import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from config.Config import CONFIG
from src.envs.matrix_world import obs_to_state, state_to_onehot
from src.data.run_context import RunRewards
from src.agents.policies import epsilon_greedy_dqn, epsilon_greedy_dqn_ran
from src.ani.animation_visualize import animate_position_2d_img

# 以obs转换模型输入数据
def obs_to_net(obs, env, device):
    # 转换当前坐标，随机终点坐标
    x, y, gx, gy = obs
    # 计算相对距离并且进行归一化处理
    dx = (gx - x) / env.max_x
    dy = (gy - y) / env.max_y
    # 返回可输入模型的数据形式
    return torch.tensor([dx, dy], dtype=torch.float32, device=device)

# 每轮训练状态初始化
def train_episode_initilize(env):
    # 每轮训练初始化观测环境
    obs, info = env.reset()
    # 每轮训练初始化结束判断
    done = False
    # 每轮训练初始化步数统计
    step_count = 0

    return obs, info, done, step_count    

def evaluate_performance(env, epsilon, q_net, n_states, device):
    # 每轮训练环境与变量初始化
    obs, info, done, eval_steps = train_episode_initilize(env)
    # 更新positions记录
    x, y, gx, gy = obs
    positions = []
    positions.append((int(x), int(y)))
    # 更新actions记录
    actions = []
    while not done:
        # 根据神经网络正向推理得到贪婪动作
        obs_tensor = obs_to_net(obs, env, device)
        a = epsilon_greedy_dqn_ran(env, epsilon, q_net, obs_tensor, device) 
        # 取得最优状态对应动作
        actions.append(a)            
        # 执行动作获得反馈
        next_obs, reward, terminated, truncated, info = env.step(a)
        # 判断是否抵达终点或超时中断
        done = terminated or truncated
        # 下一步坐标存入positions
        x, y, gx, gy = next_obs
        positions.append((int(x), int(y)))  
        # 步数累计
        eval_steps += 1              
        # 状态推进
        obs = next_obs
        
    return eval_steps, positions, actions, terminated


def train_dqn_ran(env, rctx, q_net, target_net, replay_buffer, optimizer, device):

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
    # 初始化DQN目标网络更新频率
    target_update_freq = CONFIG["algorithm"]["target_update_freq"]
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
    # 全局步数计数器-不重置
    global_step = 0
    # 进入训练，训练次数=episodes
    for ep in range(episodes):
        # 每轮训练初始化观测环境
        print("The episode >>>>>>>>> ", ep)
        obs, info, done, step_count = train_episode_initilize(env)
        # 定义网络学习计数器
        q_net_learn_count = 0
        # 开始执行直到抵达终点或任务中断
        while not done:
            # 选择一个动作
            # s_onehot = state_to_onehot(s, n_states)
            obs_tensor = obs_to_net(obs, env, device)
            a = epsilon_greedy_dqn_ran(env, epsilon, q_net, obs_tensor, device) 
            # 执行后得到反馈
            next_obs, reward, terminated, truncated, info = env.step(a)
            # 是否抵达终点或被打断
            done = terminated or truncated
            # 将本次样本记录到回放样本库
            replay_buffer.add(obs, a, reward, next_obs, done)
            # 当样本数量足够时，触发神经网络学习
            if len(replay_buffer) >= batch_size:
                # 在回放样本库采样
                batch = replay_buffer.sample(batch_size)
                # 采样数据重构为 当前状态，状态对应动作，奖励，下一状态，完成状态
                O, A, R, O2, D = zip(*batch)
                # 转换{当前观测信息}到神经网络可加载数据形式
                O_np = np.array([obs_to_net(o, env, device) for o in O],
                                 dtype=np.float32
                                 )
                O = torch.from_numpy(O_np).to(device)
                # 转换{下一观测信息}到神经网络可加载数据形式
                O2_np = np.array([obs_to_net(o2, env, device) for o2 in O2],
                                 dtype=np.float32
                                 )
                O2 = torch.from_numpy(O2_np).to(device)                
                # 转换{动作}到神经网络可加载数据形式
                A = torch.tensor(A, dtype=torch.int64, device=device).unsqueeze(1)                
                # 转换{奖励}到神经网络可加载数据形式
                R = torch.tensor(R, dtype=torch.float32, device=device)
                # 转换{是否完成}到神经网络可加载数据形式
                D = torch.tensor(D, dtype=torch.float32, device=device)
                # 以当前状态作为输入，神经网络正向推理得到所有分数，从分数中取得实际执行的动作得分
                q_sa = q_net(O).gather(1, A).squeeze(1)
                # 用下一时刻作为输入，神经网络正向推理得到所有得分，从分数中取得得分最大的动作
                with torch.no_grad():
                    target_next_max = target_net(O2).max(dim=1).values
                    y = R + gamma * (1.0 - D) * target_next_max
                # 通过当前时刻动作得分与下一时刻动作得分计算loss，逼近
                loss = F.smooth_l1_loss(q_sa, y)
                # 清空梯度
                optimizer.zero_grad()
                # 反向传递
                loss.backward()
                # 优化参数
                optimizer.step()
                # 更新计数器
                q_net_learn_count += 1
                # 目标网络参数更新
                if q_net_learn_count > target_update_freq:
                    target_net.load_state_dict(q_net.state_dict())
                    q_net_learn_count = 0
                # 按step将TD Error记录tensorBoard
                td_error = torch.abs(q_sa - y).mean().item()
                writer.add_scalar("Train/TD_Error", td_error, global_step)
                writer.add_scalar("Train/Q_mean", q_sa.mean().item(), global_step)
                writer.add_scalar("Train/Q_max", q_sa.max().item(), global_step)
                writer.add_scalar("Train/Loss", loss.item(), global_step)

            # 状态推进
            obs = next_obs
            # 步数累计
            global_step += 1
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
        eval_steps, positions, actions, eval_terminated = evaluate_performance(env, epsilon, q_net, n_states, device)
        # 将每轮ep学习效果加入tensorBoard
        writer.add_scalar("Eval/Steps", eval_steps, ep)
        writer.add_scalar("Eval/Success", eval_terminated, ep)
        # 满足条件触发动画制作
        if save_gif and ep >= save_gif_ep_start and ep % save_gif_ep == 0: 
            run_time = datetime.now().strftime("%Y%m%d%H%M%S")
            ani_path = log_dir / f"{rctx.execute_stem}_{ep}_{run_time}.gif"
            # ani = animate_position_2d(env, positions, actions, ani_path)  # 无定制化动画
            ani = animate_position_2d_img(env, positions, actions, ani_path, agent_img_dir, des_img_dir, end_img_path, terminated, ending_time) # 定制化动画版本
            
    # 关闭tensorBoard文件写入
    writer.close()    