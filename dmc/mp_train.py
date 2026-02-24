import os
os.environ["OMP_NUM_THREADS"] = "1" 

import sys, time, random, torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path: sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.mp_actor import create_buffers, act_worker, UNROLL_LENGTH
from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

def evaluate(model, device, eval_episodes=20):
    env_wrapper = GuandanEnvWrapper()
    bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}
    total_score, wins = 0.0, 0
    model.eval() 
    for _ in range(eval_episodes):
        # 考试也要随机测逢人配的理解度
        obs = env_wrapper.reset(current_level=random.randint(2, 14))
        while True:
            cur_player, legal_actions = obs['player_id'], obs['legal_actions']
            if not legal_actions: legal_actions = [[]]
            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = env_wrapper.step([])
                if done: break
                continue
            if cur_player == 0:
                with torch.no_grad():
                    scores = model(
                        torch.FloatTensor(obs['x_batch']['query']).to(device),
                        torch.FloatTensor(obs['x_batch']['context']).to(device),
                        torch.FloatTensor(obs['x_batch']['history']).to(device),
                        torch.FloatTensor(obs['x_batch']['history_mask']).to(device)
                    ).squeeze(-1)
                obs, reward, done, result_info = env_wrapper.step(legal_actions[torch.argmax(scores).item()])
            else:
                obs, reward, done, result_info = env_wrapper.step(bots[cur_player].act(obs['infoset']))
            if done: break
        
        result = result_info['result']
        team_score = float(result['level_up']) if result['winner'] == 'A' else -float(result['level_up'])
        total_score += (team_score - len(env_wrapper.env.players_hand[0]) * 0.05) / 3.0 
        if team_score > 0: wins += 1
    return total_score / eval_episodes, wins / eval_episodes

def train():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 总司令塔启动！Learner 运行在: {device}")
    
    # 【加载新版 512 维的大模型】
    learner_model = GuandanModel(hidden_dim=512).to(device)
    shared_actor_model = GuandanModel(hidden_dim=512).to('cpu')
    shared_actor_model.load_state_dict(learner_model.state_dict())
    shared_actor_model.share_memory() 
    shared_actor_model.eval()

    optimizer = optim.Adam(learner_model.parameters(), lr=1e-4)
    
    NUM_ACTORS, NUM_BUFFERS, BATCH_SIZE = 16, 300, 128      
    TOTAL_FRAMES, EVAL_FREQ_FRAMES = 10000000, 20000  
    best_eval_score = -999.0
    os.makedirs("checkpoints", exist_ok=True)

    buffers = create_buffers(NUM_BUFFERS)
    ctx = mp.get_context('spawn')
    free_queue, full_queue = ctx.SimpleQueue(), ctx.SimpleQueue()
    for i in range(NUM_BUFFERS): free_queue.put(i)

    actor_processes = []
    for i in range(NUM_ACTORS):
        actor_eps = 0.01 + (0.5 - 0.01) * (i / max(1, NUM_ACTORS - 1))
        p = ctx.Process(target=act_worker, args=(i, free_queue, full_queue, shared_actor_model, buffers, actor_eps))
        p.start()
        actor_processes.append(p)

    frames_processed, last_eval_frame, start_time = 0, 0, time.time()
    print("\n" + "="*60 + f"\n🔥 数据洪流已开启! Actor数量: {NUM_ACTORS} | 批量大小: {BATCH_SIZE}x{UNROLL_LENGTH}\n" + "="*60 + "\n")

    try:
        while frames_processed < TOTAL_FRAMES:
            indices = [full_queue.get() for _ in range(BATCH_SIZE)]
            
            b_query = torch.stack([buffers['query'][m] for m in indices], dim=0).to(device)       
            b_context = torch.stack([buffers['context'][m] for m in indices], dim=0).to(device)   
            b_history = torch.stack([buffers['history'][m] for m in indices], dim=0).to(device)   
            b_mask = torch.stack([buffers['history_mask'][m] for m in indices], dim=0).to(device) 
            b_target = torch.stack([buffers['target'][m] for m in indices], dim=0).to(device)     
            
            # 【核心排雷】：这里的解包维度必须跟着 MAX_HISTORY 一起改成 64！
            b_query = b_query.view(-1, 216)
            b_context = b_context.view(-1, 112)
            b_history = b_history.view(-1, 64, 112)
            b_mask = b_mask.view(-1, 64)
            b_target = b_target.view(-1, 1) 
            
            learner_model.train()
            preds = learner_model(b_query, b_context, b_history, b_mask)
            loss = F.mse_loss(preds, b_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            shared_actor_model.load_state_dict(learner_model.state_dict())
            for m in indices: free_queue.put(m)
                
            frames_processed += BATCH_SIZE * UNROLL_LENGTH
            if (frames_processed // (BATCH_SIZE * UNROLL_LENGTH)) % 10 == 0:
                print(f"⚡ [Training] 帧数: {frames_processed} | Loss: {loss.item():.4f} | 吞吐量: {frames_processed / (time.time() - start_time):.1f} frames/sec")

            if frames_processed - last_eval_frame >= EVAL_FREQ_FRAMES:
                last_eval_frame = frames_processed
                print("\n" + "🌟"*20 + f"\n[{frames_processed} 帧] 暂停吞吐，开启纯实力摸底考试 (Epsilon=0)...")
                eval_score, eval_win_rate = evaluate(learner_model, device, eval_episodes=20)
                print(f"🎯 评估结果 -> 真实均分: {eval_score:+.2f} | 胜率: {eval_win_rate*100:.1f}%")
                
                torch.save(learner_model.state_dict(), "checkpoints/v4_moe_latest.pth")
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    torch.save(learner_model.state_dict(), "checkpoints/v4_moe_best.pth")
                    print(f"💾 突破历史最高分！已保存最优权重至 checkpoints/v4_moe_best.pth")
                print("🌟"*20 + "\n")

    except KeyboardInterrupt: print("\n⚠️ 收到停止指令，正在安全关闭...")
    finally:
        for _ in actor_processes: free_queue.put(None) 
        for p in actor_processes:
            p.join(timeout=3)
            if p.is_alive(): p.terminate()
        print("✅ 训练安全结束！")

if __name__ == '__main__':
    train()