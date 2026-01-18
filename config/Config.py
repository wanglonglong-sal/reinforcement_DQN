CONFIG = {
    "training":{
        "episodes":2000,              # 训练轮数 
        "train_record_fre":1000,      # 训练数据记录频次，基于全局步数
        "eval_performance_fre": 20,  # 表现评估频次，基于ep轮次
    },
    "algorithm":{
        "alpha" : 0.1,   # 学习率
        "gamma" : 0.99,  # 未来折扣因子
        "batch_size": 32,                   # 样本批次数量 - DQN采样用   
        "lr": 1e-3,                         # 神经网络学习率 - DQN
        "target_update_freq": 200           # 目标网络参数更新频率 - DQN
    },
    "exploration":{
        "epsilon_start":0.3,    # 初始随机率
        "epsilon_decay":0.9,    # 衰减率
        "epsilon_min":0.01      # 最小随机率

    },
    "environment":{
        "width":12,       # 2D空间宽度 
        "height":12,      # 2D空间高度
        "min_steps":1,    # 最大步数
        "max_steps":500  # 最大步数
    },
    "paths":{
        "log_dir":"runs",    # log目录
        "ani_dir":"anis"     # 动画目录 - 未使用
    },
    "animation":{
        "save_gif":True,                # 是否保存动画
        "save_git_ep_start":1800,        # 保存动画起始轮次
        "save_gif_ep":45,               # 保存动画轮次间隔 
        "agent_img_dir":"imgs\\G.gif",  # Agent图片地址
        "des_img_dir":"imgs\\T.gif",    # 终点图片地址    
        "end_img_path":"imgs\\Ending.gif", # 胜利画面地址   
        "ending_time":10                # 胜利画面持续时长  

    },
    "rewards":{
        "goal_pos_reward":10,       # 抵达终点奖励
        "step_reward":-0.1,       # 每步惩罚
        "hit_wall_enable":True,    # 撞墙惩罚开关
        "hit_wall_reward":-1,    # 撞墙惩罚
        "repeat_position_enable":True,  # 横跳惩罚开关
        "repeat_position_reward":-0.5   # 横跳惩罚
    }
}