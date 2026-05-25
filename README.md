# DQN Grid World 强化学习 Demo

[English Version](README_EN.md)

一个基于 PyTorch 与 Gymnasium 的 2D Grid World 强化学习 Demo，用于观察智能体在离散环境中的状态转移、奖励反馈、探索策略和路径学习过程。项目实现了 DQN、随机目标点导航、经验回放、目标网络、Double DQN、TensorBoard 训练记录和 GIF 路径可视化。

## 演示与结果

训练过程中会定期在 `runs/` 目录下保存评估轨迹 GIF。下面展示的是随机目标点环境下的 Double DQN 评估结果，机器人从起点出发，根据 Q 网络输出的动作逐步移动到目标旗帜位置。

| 演示 | 预览 | 内容 |
| --- | --- | --- |
| Double DQN rollout 420 | [GIF](imgs/2Dworld_DQN_ran_dual_420_20260525141818.gif) | 第 420 轮附近的路径评估结果 |
| Double DQN rollout 480 | [GIF](imgs/2Dworld_DQN_ran_dual_480_20260525141829.gif) | 第 480 轮附近的路径评估结果 |

### Double DQN Rollout

![Double DQN rollout 420](imgs/2Dworld_DQN_ran_dual_420_20260525141818.gif)

该演示使用随机目标点环境。每个 episode 中，环境会生成一个目标点，智能体根据当前位置、目标位置和相对距离特征选择动作。

![Double DQN rollout 480](imgs/2Dworld_DQN_ran_dual_480_20260525141829.gif)

## 功能特性

- 自定义 2D Grid World 环境，包含 agent、goal、离散动作空间和终止条件。
- 支持固定目标点和随机目标点两类导航任务。
- 基于 PyTorch 实现 DQN 训练流程，包括 Q 网络、经验回放和 epsilon-greedy 探索。
- 使用 target network 提升训练稳定性。
- 在随机目标点任务中实现 Double DQN 目标估计。
- 支持 reward shaping，包括抵达目标奖励、步数惩罚、撞墙惩罚和重复移动惩罚。
- 使用 TensorBoard 记录 `Loss`、`TD Error`、`Q Mean`、`Q Max`、`Episode Steps` 和 `Eval Success`。
- 支持 checkpoint 保存、加载、续训和迁移初始化。
- 支持将评估路径保存为 GIF，便于观察策略学习效果。

## 技术栈

- Python 3.10
- PyTorch
- Gymnasium
- NumPy
- TensorBoard
- Matplotlib
- Pillow

## 项目结构

```text
.
|-- config/
|   `-- Config.py
|-- imgs/
|   |-- G.gif
|   |-- T.gif
|   `-- Ending.gif
|-- src/
|   |-- agents/
|   |   |-- dqn_net.py
|   |   |-- policies.py
|   |   `-- replay_buffer.py
|   |-- ani/
|   |   `-- animation_visualize.py
|   |-- ckp/
|   |   `-- ckp_functions.py
|   |-- data/
|   |   `-- run_context.py
|   |-- envs/
|   |   `-- matrix_world.py
|   |-- main/
|   |   |-- 2Dworld_DQN.py
|   |   |-- 2Dworld_DQN_ran.py
|   |   `-- 2Dworld_DQN_ran_dual.py
|   |-- trainers/
|   |   |-- dqn_trainer.py
|   |   |-- dqn_trainer_random.py
|   |   `-- dqn_trainer_dual.py
|   |-- uti/
|   `-- viz/
|-- ckps/
|-- runs/
|-- requirements.txt
|-- cpu_env_rl.yml
`-- gpu_env_rl.yml
```

### 主要逻辑

- `config/Config.py`：训练轮数、学习率、环境尺寸、奖励函数、GIF 保存和 checkpoint 参数。
- `src/envs/matrix_world.py`：Grid World 环境定义，包括状态、动作、奖励和 episode 结束条件。
- `src/agents/dqn_net.py`：Q 网络结构。
- `src/agents/replay_buffer.py`：经验回放池。
- `src/agents/policies.py`：epsilon-greedy 动作选择。
- `src/trainers/dqn_trainer.py`：固定目标点 DQN 训练。
- `src/trainers/dqn_trainer_random.py`：随机目标点 DQN 训练。
- `src/trainers/dqn_trainer_dual.py`：随机目标点 Double DQN 训练。
- `src/ckp/ckp_functions.py`：模型参数保存与加载。
- `src/ani/animation_visualize.py`：训练评估路径的 GIF 可视化。

## 运行方法

### 1. 创建 Conda 环境

```powershell
conda create -n env_rl python=3.10 -y
conda activate env_rl
```

### 2. 安装依赖

```powershell
pip install -r .\requirements.txt
```

CPU 版本 PyTorch：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

GPU 版本 PyTorch：

```powershell
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

也可以通过环境文件重建：

```powershell
conda env create -f cpu_env_rl.yml
conda activate env_rl
```

或：

```powershell
conda env create -f gpu_env_rl.yml
conda activate env_rl
```

### 3. 运行训练

进入项目根目录：

```powershell
cd D:\Workspace\reinforce\DQN
```

运行随机目标点 Double DQN：

```powershell
python -m src.main.2Dworld_DQN_ran_dual
```

其他入口：

```powershell
python -m src.main.2Dworld_DQN
python -m src.main.2Dworld_DQN_ran
```

### 4. 查看 TensorBoard

```powershell
tensorboard --logdir runs
```

然后在浏览器打开：

```text
http://localhost:6006/
```

## 配置说明

主要参数位于：

```text
config/Config.py
```

### 训练参数

```python
"training": {
    "episodes": 2000,
    "train_record_fre": 1000,
    "eval_performance_fre": 100,
    "train_mode": 2,
    "load_resume_file_path": "ckps\\best_1615.pt",
    "pt_save_enabled": True
}
```

- `episodes`：训练 episode 数。
- `train_record_fre`：训练指标记录频率，基于全局 step。
- `eval_performance_fre`：评估频率，基于 episode。
- `train_mode`：
  - `0`：全新训练。
  - `1`：相同环境下恢复训练，会加载模型、优化器、epsilon 和 global step。
  - `2`：迁移初始化，只加载 Q 网络和 target network 参数。
- `pt_save_enabled`：是否保存 checkpoint。

### 环境参数

```python
"environment": {
    "width": 20,
    "height": 20,
    "max_steps": 500
}
```

随机目标点版本会在每个 episode 中生成新的 goal。模型输入为：

```text
[x, y, gx, gy, dx, dy]
```

其中 `x, y` 为当前位置，`gx, gy` 为目标位置，`dx, dy` 为归一化后的相对距离。

### 奖励设置

```python
"rewards": {
    "goal_pos_reward": 10,
    "step_reward": -0.1,
    "hit_wall_enable": True,
    "hit_wall_reward": -1,
    "repeat_position_enable": True,
    "repeat_position_reward": -0.5
}
```

奖励函数鼓励智能体尽快到达目标点，同时减少撞墙和原地来回移动。

### GIF 保存

```python
"animation": {
    "save_gif": True,
    "save_git_ep_start": 1800,
    "save_gif_ep": 45,
    "agent_img_dir": "imgs\\G.gif",
    "des_img_dir": "imgs\\T.gif",
    "end_img_path": "imgs\\Ending.gif",
    "ending_time": 10
}
```

当前 Double DQN 训练器中，GIF 保存是在评估逻辑中触发的，因此实际保存频率主要由 `eval_performance_fre` 控制。保存路径位于当前训练对应的 `runs/` 子目录。

## Checkpoint 说明

checkpoint 默认保存在：

```text
ckps/
```

保存内容包括：

- `q_net`
- `target_net`
- `optimizer`
- `episode`
- `epsilon`
- `global_step`

当前代码按固定频率保存 checkpoint，并没有自动判断 best model。`best_1010.pt` 和 `best_1615.pt` 更适合作为手动挑选出的模型文件。

## 注意事项

- 当前项目主要用于强化学习算法理解和可视化展示，不是完整强化学习框架。
- 随机目标点版本使用固定 6 维特征输入，因此可以从 `10x10` 迁移到 `20x20` 继续训练；但如果使用 one-hot 状态输入，环境尺寸变化会导致网络输入维度不匹配。
- `save_git_ep_start` 字段名保留了当前代码中的拼写，修改时不要写成 `save_gif_ep_start`，否则训练器读取不到。
- `ani_dir` 当前没有实际使用，GIF 会保存在 `runs/` 下的训练日志目录中。
- 如果要在 GitHub 展示，建议清理 `__pycache__`、临时日志和过多 checkpoint，只保留必要结果 GIF 与精选模型。
