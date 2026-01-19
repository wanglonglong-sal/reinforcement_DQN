import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
import torch.optim as optim
from copy import deepcopy
from config.Config import CONFIG
from src.uti.utilities import get_path_variables
from src.data.run_context import RunContext
from src.envs.matrix_world import MatrixWorld
from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_net import QNetRan
from src.trainers.dqn_trainer_dual import train_dqn_ran

# 检查环境中是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    q_net = QNetRan(6, env.action_space.n).to(device)
    # 创建target_net，结构和参数保持与q_net一致
    target_net = deepcopy(q_net).to(device)
    target_net.load_state_dict(q_net.state_dict())
    # 目标网络切换到评估模式
    target_net.eval()
    # 初始化样本重放对象
    replay_buffer = ReplayBuffer()
    # 初始化优化器对象
    optimizer = optim.Adam(q_net.parameters(), lr=CONFIG["algorithm"]["lr"])

    # 开始强化学习训练
    train_dqn_ran(env, rctx, q_net, target_net, replay_buffer, optimizer, device)

    print("done")
