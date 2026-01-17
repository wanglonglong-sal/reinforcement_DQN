Python: 3.10 (local)

Create conda enviroment
- conda create -n env_rl python=3.10 -y
- conda activate env_rl

Install packages
- pip install -r .\requirements.txt

Install torch, cpu or gpu version
- gpu: conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Run python program
- python -m src.main.2Dworld_DQN_ran
- python -m src.main.2Dworld_DQN

Run tensorBoard
- tensorboard --logdir runs
- http:127.0.0.1:10060

Supervise gpu performance
- nvidia-smi -l 1

