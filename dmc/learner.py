import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import sys
import copy

# 确保能找到项目根目录
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.actor import Actor

def evaluate(model, eval_episodes=20):
    """摸底考试：关闭探索，纯靠模型实力对战"""
    eval_actor = Actor(model=model, epsilon=0.0)
    total_score = 0.0
    wins = 0
    
    for _ in range(eval_episodes):
        _, score = eval_actor.play_episode()
        total_score += score
        if score > 0: wins += 1
        
    return total_score / eval_episodes, wins / eval_episodes

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 训练运行在设备: {device}")
    
    save_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    model = GuandanModel(hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    
    # ==========================================
    # [新增] 断点续练 (Checkpoint Loading) 核心逻辑
    # ==========================================
    start_episode = 1
    best_eval_score = -999.0
    
    checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    old_model_path = os.path.join(save_dir, 'latest_model.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 发现完整断点存档，正在加载: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_episode = checkpoint['episode'] + 1
        best_eval_score = checkpoint.get('best_eval_score', -999.0)
        print(f"✅ 断点加载成功！将从第 {start_episode} 局无缝继续训练...")
        
    elif os.path.exists(old_model_path):
        # 兼容你之前只保存了权重的模型
        print(f"🔄 发现历史模型权重，正在加载: {old_model_path}")
        model.load_state_dict(torch.load(old_model_path, map_location=device))
        print("✅ 权重加载成功！(提示: 这是老版本存档，局数和探索率将从 1 重新开始计算)")
    else:
        print("✨ 未发现历史存档，从头开始初始化全新模型。")
    # ==========================================

    actor = Actor(model=model, epsilon=0.5) 
    memory = []
    
    TOTAL_EPISODES = 50000   
    BATCH_SIZE = 256         
    UPDATES_PER_EP = 5
    EVAL_FREQ = 200          
    
    recent_train_rewards = []
    
    print("\n🎬 开始大规模蒙特卡洛 (DMC) 强化学习...")
    print("="*60)
    
    # 循环从 start_episode 开始，确保断点续练时局数是连续的
    for episode in range(start_episode, TOTAL_EPISODES + 1):
        
        # Epsilon 线性退火：根据真实的局数衰减瞎打概率
        actor.epsilon = max(0.02, 0.5 - episode * (0.48 / 20000.0))
        
        # A. 收集数据
        episode_data, final_reward = actor.play_episode()
        for data in episode_data:
            data['target'] = final_reward
            memory.append(data)
            
        recent_train_rewards.append(final_reward)
        if len(recent_train_rewards) > 50: recent_train_rewards.pop(0)
            
        if episode % 10 == 0:
            avg_reward = sum(recent_train_rewards) / len(recent_train_rewards)
            print(f"🔄 训练进度 [{episode}/{TOTAL_EPISODES}] | 探索率: {actor.epsilon:.3f} | 近50局均分: {avg_reward:+.2f} | 记忆库: {len(memory)}")
        
        # B. 梯度更新
        if len(memory) >= BATCH_SIZE:
            model.train()
            total_loss = 0.0
            
            for _ in range(UPDATES_PER_EP):
                batch = random.sample(memory, BATCH_SIZE)
                
                b_query = torch.tensor(np.array([item['query'] for item in batch])).to(device)
                b_context = torch.tensor(np.array([item['context'] for item in batch])).to(device)
                b_history = torch.tensor(np.array([item['history'] for item in batch])).to(device)
                b_mask = torch.tensor(np.array([item['history_mask'] for item in batch])).to(device)
                b_target = torch.tensor(np.array([item['target'] for item in batch]), dtype=torch.float32).to(device).unsqueeze(1)
                
                preds = model(b_query, b_context, b_history, b_mask)
                loss = F.mse_loss(preds, b_target)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()

        # C. 清理记忆
        if len(memory) > 30000:
            memory = memory[-30000:]

        # ==========================================
        # 定期摸底考试与断点存档
        # ==========================================
        if episode % EVAL_FREQ == 0:
            print("\n" + "🌟"*20)
            print(f"[{episode}局] 暂停训练，开启纯实力摸底考试 (Epsilon=0)...")
            model.eval() 
            eval_score, eval_win_rate = evaluate(model, eval_episodes=30)
            
            print(f"🎯 评估结果 -> 真实均分: {eval_score:+.2f} | 胜率: {eval_win_rate*100:.1f}%")
            
            # 【核心修改】：保存完整的 Checkpoint，包含局数、优化器状态等
            checkpoint = {
                'episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_eval_score': best_eval_score
            }
            torch.save(checkpoint, os.path.join(save_dir, 'latest_checkpoint.pth'))
            
            # 保存最优模型（依然只存纯权重，方便以后写对战脚本直接 load_state_dict 调用）
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"💾 突破历史最高分！已保存最优权重至 {best_path}")
                
            print(f"📊 当前学习率: {scheduler.get_last_lr()[0]:.6f}")
            print("🌟"*20 + "\n")

if __name__ == '__main__':
    train()