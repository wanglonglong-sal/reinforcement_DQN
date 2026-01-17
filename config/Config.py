CONFIG = {
    "training":{
        "episodes":500,  # 训练轮数 
        "log_interval":20  # TensorBoard 记录间隔（按 step）
    },
    "algorithm":{
        "alpha" : 0.1,  # 学习率
        "gamma" : 0.99,  # 未来折扣因子
        "batch_size": 32,                   # 样本批次数量 - DQN采样用   
        "lr": 1e-3,                         # 神经网络学习率 - DQN
        "target_update_freq": 100           # 目标网络参数更新频率 - DQN
    },
    "exploration":{
        "epsilon_start":0.3,    # 初始随机率
        "epsilon_decay":0.9,    # 衰减率
        "epsilon_min":0.01      # 最小随机率

    },
    "environment":{
        "width":15,       # 2D空间宽度 
        "height":15,      # 2D空间高度
        "min_steps":1,   # 最大步数
        "max_steps":1000 # 最大步数
    },
    "paths":{
        "log_dir":"runs",
        "ani_dir":"anis"
    },
    "animation":{
        "save_gif":True,         # 是否保存动画
        "save_git_ep_start":200, # 保存动画起始轮次
        "save_gif_ep":90,       # 保存动画轮次间隔 
        "agent_img_dir":"imgs\\G.gif",  # Agent图片地址
        "des_img_dir":"imgs\\T.gif",    # 终点图片地址    
        "end_img_path":"imgs\\Ending.gif", # 胜利画面地址   
        "ending_time":10  # 胜利画面持续时长  

    },
    "rewards":{
        "max_pos_reward":1,
        "step_reward":-0.01,
        "hit_wall_enable":True,
        "hit_wall_reward":-0.5
    }
}
