import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import sys

# 确保能找到项目根目录
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.actor import Actor

def train():
    # 1. 初始化设备、模型与优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练运行在设备: {device}")
    
    model = GuandanModel(hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 2. 初始化环境打工人 (Actor)
    actor = Actor(model=model, epsilon=0.1)
    
    # 3. 初始化经验回放池与统计变量
    memory = []
    batch_size = 128
    updates_per_episode = 5  # 每打完一局，模型更新的次数 (加速数据利用率)
    
    recent_rewards = []      # 用于记录最近 20 局的得分，观察趋势
    
    print("🎬 开始蒙特卡洛 (DMC) 强化学习训练...")
    print("="*60)
    
    # 设置一个较大的训练局数，你可以随时 Ctrl+C 中断
    for episode in range(1, 1001):
        # A. 收集数据
        episode_data, final_reward = actor.play_episode()
        
        # B. 为数据贴上这局的最终得分 (Target) 并存入记忆池
        for data in episode_data:
            data['target'] = final_reward
            memory.append(data)
            
        # 记录最近的得分用于计算滑动平均值
        recent_rewards.append(final_reward)
        if len(recent_rewards) > 20:
            recent_rewards.pop(0)
        avg_reward = sum(recent_rewards) / len(recent_rewards)
            
        print(f"🔄 回合 [{episode:03d}] 结束 | 本局得分: {final_reward:+.2f} | 近20局均分: {avg_reward:+.2f} | 记忆库大小: {len(memory)}")
        
        # C. 记忆库数据足够后，开始执行梯度下降更新
        if len(memory) >= batch_size:
            model.train() # 切换到训练模式
            total_loss = 0.0
            
            # 增加数据复用率：一局产生几十条数据，我们连续采样更新多次
            for _ in range(updates_per_episode):
                batch = random.sample(memory, batch_size)
                
                # 将 numpy 数组转为 GPU Tensor
                b_query = torch.tensor(np.array([item['query'] for item in batch])).to(device)
                b_context = torch.tensor(np.array([item['context'] for item in batch])).to(device)
                b_history = torch.tensor(np.array([item['history'] for item in batch])).to(device)
                b_mask = torch.tensor(np.array([item['history_mask'] for item in batch])).to(device)
                
                # Target 维度对齐为 [Batch_Size, 1]
                b_target = torch.tensor(np.array([item['target'] for item in batch]), dtype=torch.float32).to(device).unsqueeze(1)
                
                # 前向传播打分
                preds = model(b_query, b_context, b_history, b_mask)
                
                # 计算均方误差 (MSE)
                loss = F.mse_loss(preds, b_target)
                
                # 反向传播，更新参数
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪：保护大模型训练不崩盘的重要技巧
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / updates_per_episode
            print(f"   📉 [Train] 当前平均 Loss: {avg_loss:.4f}")
            
        # D. 清理过老的记忆 (维持在最近的 20000 步决策，约 500 局)
        if len(memory) > 20000:
            memory = memory[-20000:]

    print("🎉 训练运行完毕！")

if __name__ == '__main__':
    train()