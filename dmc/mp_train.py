import os
os.environ["OMP_NUM_THREADS"] = "1" # 防死锁

import sys
import time
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.mp_actor import create_buffers, act_worker, UNROLL_LENGTH
# 引入环境供 GPU 亲自做摸底考试使用
from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

def evaluate(model, device, eval_episodes=20):
    """独立的摸底考试函数：关闭探索，纯实力对战"""
    env_wrapper = GuandanEnvWrapper()
    bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}
    
    total_score = 0.0
    wins = 0
    
    model.eval() # 开启评估模式
    for _ in range(eval_episodes):
        rand_level = random.randint(2, 14)
        obs = env_wrapper.reset(current_level=2)
        while True:
            cur_player = obs['player_id']
            legal_actions = obs['legal_actions']
            
            if not legal_actions: legal_actions = [[]]
            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = env_wrapper.step([])
                if done: break
                continue

            if cur_player == 0:
                x_batch = obs['x_batch']
                # 放入 GPU 加速算牌
                q_t = torch.FloatTensor(x_batch['query']).to(device)
                c_t = torch.FloatTensor(x_batch['context']).to(device)
                h_t = torch.FloatTensor(x_batch['history']).to(device)
                m_t = torch.FloatTensor(x_batch['history_mask']).to(device)
                
                with torch.no_grad():
                    scores = model(q_t, c_t, h_t, m_t).squeeze(-1)
                
                # 【极其重要】：摸底考试绝对不探索，纯靠网络最高分决策
                action_idx = torch.argmax(scores).item()
                best_action = legal_actions[action_idx]
                
                obs, reward, done, result_info = env_wrapper.step(best_action)
            else:
                action = bots[cur_player].act(obs['infoset'])
                obs, reward, done, result_info = env_wrapper.step(action)
                
            if done: 
                break
                
        # 结算本局
        result = result_info['result']
        team_score = float(result['level_up']) if result['winner'] == 'A' else -float(result['level_up'])
        penalty = len(env_wrapper.env.players_hand[0]) * 0.05
        scaled_reward = (team_score - penalty) / 3.0 
        
        total_score += scaled_reward
        if team_score > 0: wins += 1
        
    return total_score / eval_episodes, wins / eval_episodes


def train():
    mp.set_start_method('spawn', force=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 总司令塔启动！Learner 运行在: {device}")
    
    learner_model = GuandanModel(hidden_dim=256).to(device)
    shared_actor_model = GuandanModel(hidden_dim=256).to('cpu')
    shared_actor_model.load_state_dict(learner_model.state_dict())
    shared_actor_model.share_memory() 
    shared_actor_model.eval()

    optimizer = optim.Adam(learner_model.parameters(), lr=1e-4)
    
    NUM_ACTORS = 12       
    NUM_BUFFERS = 300     
    BATCH_SIZE = 64      
    TOTAL_FRAMES = 10000000
    
    # 每收集 20000 帧（大约耗时 40 秒），进行一次纯实力摸底考试
    EVAL_FREQ_FRAMES = 20000  
    best_eval_score = -999.0
    os.makedirs("checkpoints", exist_ok=True)

    buffers = create_buffers(NUM_BUFFERS)
    ctx = mp.get_context('spawn')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()
    
    for i in range(NUM_BUFFERS):
        free_queue.put(i)

    actor_processes = []
    for i in range(NUM_ACTORS):
        # 【核心黑科技】：分布式探索策略
        # 让 12 个 Actor 的瞎打概率从 0.01 均匀分布到 0.5
        actor_eps = 0.01 + (0.5 - 0.01) * (i / max(1, NUM_ACTORS - 1))
        
        p = ctx.Process(
            target=act_worker,
            args=(i, free_queue, full_queue, shared_actor_model, buffers, actor_eps)
        )
        p.start()
        actor_processes.append(p)

    frames_processed = 0
    last_eval_frame = 0
    start_time = time.time()
    
    print("\n" + "="*60)
    print(f"🔥 数据洪流已开启! Actor数量: {NUM_ACTORS} | 批量大小: {BATCH_SIZE}x{UNROLL_LENGTH}")
    print("="*60 + "\n")

    try:
        while frames_processed < TOTAL_FRAMES:
            indices = [full_queue.get() for _ in range(BATCH_SIZE)]
            
            b_query = torch.stack([buffers['query'][m] for m in indices], dim=0).to(device)       
            b_context = torch.stack([buffers['context'][m] for m in indices], dim=0).to(device)   
            b_history = torch.stack([buffers['history'][m] for m in indices], dim=0).to(device)   
            b_mask = torch.stack([buffers['history_mask'][m] for m in indices], dim=0).to(device) 
            b_target = torch.stack([buffers['target'][m] for m in indices], dim=0).to(device)     
            
            b_query = b_query.view(-1, 216)
            b_context = b_context.view(-1, 112)
            b_history = b_history.view(-1, 40, 112)
            b_mask = b_mask.view(-1, 40)
            b_target = b_target.view(-1, 1) 
            
            learner_model.train()
            preds = learner_model(b_query, b_context, b_history, b_mask)
            loss = F.mse_loss(preds, b_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            shared_actor_model.load_state_dict(learner_model.state_dict())
            
            for m in indices:
                free_queue.put(m)
                
            frames_processed += BATCH_SIZE * UNROLL_LENGTH
            
            if (frames_processed // (BATCH_SIZE * UNROLL_LENGTH)) % 10 == 0:
                fps = frames_processed / (time.time() - start_time)
                print(f"⚡ [Training] 帧数: {frames_processed} | Loss: {loss.item():.4f} | 吞吐量: {fps:.1f} frames/sec")

            # ==========================================
            # 🌟 定期摸底考试与最好模型保存
            # ==========================================
            if frames_processed - last_eval_frame >= EVAL_FREQ_FRAMES:
                last_eval_frame = frames_processed
                print("\n" + "🌟"*20)
                print(f"[{frames_processed} 帧] 暂停吞吐，开启纯实力摸底考试 (Epsilon=0)...")
                
                # 连打 20 局评估胜率
                eval_score, eval_win_rate = evaluate(learner_model, device, eval_episodes=20)
                
                print(f"🎯 评估结果 -> 真实均分: {eval_score:+.2f} | 胜率: {eval_win_rate*100:.1f}%")
                
                # 保存最新模型（用于断点续练）
                torch.save(learner_model.state_dict(), "checkpoints/v3_mp_latest.pth")
                
                # 破纪录则保存最好模型
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_path = "checkpoints/v3_mp_best.pth"
                    torch.save(learner_model.state_dict(), best_path)
                    print(f"💾 突破历史最高分！已保存最优权重至 {best_path}")
                print("🌟"*20 + "\n")

    except KeyboardInterrupt:
        print("\n⚠️ 收到停止指令，正在安全关闭...")
    finally:
        for _ in actor_processes:
            free_queue.put(None) 
        for p in actor_processes:
            p.join(timeout=3)
            if p.is_alive(): p.terminate()
            
        print("✅ 训练安全结束！")

if __name__ == '__main__':
    train()