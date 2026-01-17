import torch
import numpy as np

# 动作选择
def epsilon_greedy_dqn(env, epsilon, q_net, s_onehot, device):
    # 随机选择
    if np.random.rand() < epsilon:
        # Sample a random action
        return env.action_space.sample()
    # 根据神经网络正向推理得到贪婪动作
    with torch.no_grad():
        if torch.is_tensor(s_onehot):
            x = s_onehot.to(device)
        else:
            x = torch.tensor(s_onehot, dtype=torch.float32, device=device)
        x = x.unsqueeze(0)
        q_values = q_net(x)
        return int(torch.argmax(q_values, dim=1).item())
    
# 动作选择
def epsilon_greedy_dqn_ran(env, epsilon, q_net, obs_np, device):
    # 随机选择
    if np.random.rand() < epsilon:
        # Sample a random action
        return env.action_space.sample()
    # 根据神经网络正向推理得到贪婪动作
    with torch.no_grad():
        x = torch.from_numpy(obs_np).to(device).unsqueeze(0) 
        q_values = q_net(x)
        return int(torch.argmax(q_values, dim=1).item())
