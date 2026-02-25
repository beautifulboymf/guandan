import os
os.environ["OMP_NUM_THREADS"] = "1" 

import sys, time, random, torch, glob, math
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path: sys.path.insert(0, project_root)

from dmc.models import GuandanModel
from dmc.unified_actor import create_buffers, act_worker_unified, UNROLL_LENGTH
from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

def evaluate(model, device, phase, eval_episodes=20):
    """
    智能双模质检仪：
    如果 Phase 0：100% 打 Bot，检测基础规则掌握度。
    如果 Phase 1：80% 打历史自己，20% 打 Bot，检测高阶博弈能力。
    """
    env_wrapper = GuandanEnvWrapper()
    bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}
    
    history_model = None
    if phase == 1:
        history_files = glob.glob("history_models/*.pth")
        if history_files:
            history_model = GuandanModel(hidden_dim=512).to(device)
            # strict=False 防止历史模型加载失败
            history_model.load_state_dict(torch.load(random.choice(history_files), map_location=device, weights_only=True), strict=False)
            history_model.eval()

    total_score, wins = 0.0, 0
    model.eval() 
    
    num_history_games = int(eval_episodes * 0.5) if history_model else 0
    
    for ep in range(eval_episodes):
        obs = env_wrapper.reset(current_level=random.randint(2, 14))
        opponent_type = 'history' if ep < num_history_games else 'bot'

        while True:
            cur_player, legal_actions = obs['player_id'], obs['legal_actions']
            if not legal_actions: legal_actions = [[]]
            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = env_wrapper.step([])
                if done: break
                continue
            
            if cur_player in [0, 2]:
                with torch.no_grad():
                    scores = model(
                        torch.FloatTensor(obs['x_batch']['query']).to(device),
                        torch.FloatTensor(obs['x_batch']['context']).to(device),
                        torch.FloatTensor(obs['x_batch']['history']).to(device),
                        torch.FloatTensor(obs['x_batch']['history_mask']).to(device)
                    ).squeeze(-1)
                best_action = legal_actions[torch.argmax(scores).item()]
            else:
                if opponent_type == 'bot':
                    best_action = bots[cur_player].act(obs['infoset'])
                else:
                    with torch.no_grad():
                        scores = history_model(
                            torch.FloatTensor(obs['x_batch']['query']).to(device),
                            torch.FloatTensor(obs['x_batch']['context']).to(device),
                            torch.FloatTensor(obs['x_batch']['history']).to(device),
                            torch.FloatTensor(obs['x_batch']['history_mask']).to(device)
                        ).squeeze(-1)
                    best_action = legal_actions[torch.argmax(scores).item()]

            obs, reward, done, result_info = env_wrapper.step(best_action)
            if done: break
        
        result = result_info['result']
        team_score = float(result['level_up']) if result['winner'] == 'A' else -float(result['level_up'])
        total_score += team_score / 3.0 # 对齐纯稀疏奖励
        if team_score > 0: wins += 1
        
    return total_score / eval_episodes, wins / eval_episodes

def train():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌍 创世主宰启动！Unified Learner 运行在: {device}")
    
    learner_model = GuandanModel(hidden_dim=512).to(device)
    
    # 提前定义优化器，方便后续加载状态
    INITIAL_LR = 1e-4 
    MIN_LR = 5e-6
    optimizer = optim.Adam(learner_model.parameters(), lr=INITIAL_LR)
    
    shared_phase = mp.Value('i', 0) 
    frames_processed = 0
    promotion_streak = 0
    best_eval_score = -999.0
    
    latest_ckpt_path = "checkpoints/v4_unified_checkpoint.pth"
    best_model_path = "checkpoints/v4_unified_best.pth"
    
    # ==========================================
    # 【完美断点续练逻辑】
    # ==========================================
    if os.path.exists(latest_ckpt_path):
        print(f"🔄 检测到完整断点存档，正在恢复训练现场: {latest_ckpt_path} ...")
        # 注意：这里 weights_only=False，因为我们要读取整个字典
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        
        learner_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        frames_processed = checkpoint.get('frames_processed', 0)
        shared_phase.value = checkpoint.get('phase', 0)
        promotion_streak = checkpoint.get('promotion_streak', 0)
        best_eval_score = checkpoint.get('best_eval_score', -999.0)
        
        print(f"✅ 成功恢复至第 {frames_processed} 帧！当前阶段: Phase {shared_phase.value}")
        
    elif os.path.exists(best_model_path):
        print(f"🔄 检测到前世记忆(仅权重)，正在注入灵魂: {best_model_path} ...")
        learner_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True), strict=False)
        
        print("📏 正在进行开局战力测算 (纯Bot局 10局)...")
        init_score, init_win = evaluate(learner_model, device, phase=0, eval_episodes=10)
        if init_win >= 0.60:
            print(f"👑 测算胜率 {init_win*100:.1f}%，战力已臻化境！直接开启 [Phase 1: 诸神之战]！")
            shared_phase.value = 1
        else:
            print(f"🍼 测算胜率 {init_win*100:.1f}%，仍需巩固基础。进入 [Phase 0: 新手村]！")
    else:
        print("🆕 未检测到模型，从零宇宙大爆炸开始 [Phase 0: 新手村]...")
        
    shared_actor_model = GuandanModel(hidden_dim=512).to('cpu')
    shared_actor_model.load_state_dict(learner_model.state_dict())
    shared_actor_model.share_memory() 
    shared_actor_model.eval()

    NUM_ACTORS, NUM_BUFFERS, BATCH_SIZE = 16, 300, 128      
    TOTAL_FRAMES = 10000000 
    
    # 动态调度器
    INITIAL_LR = 1e-4 
    MIN_LR = 5e-6
    optimizer = optim.Adam(learner_model.parameters(), lr=INITIAL_LR)
    
    EPS_START = 0.5 if shared_phase.value == 0 else 0.3
    EPS_END = 0.05
    EPS_DECAY_FRAMES = TOTAL_FRAMES // 2 
    shared_max_eps = mp.Value('d', EPS_START) 
    
    EVAL_FREQ_FRAMES = 20000      
    HISTORY_SAVE_FREQ = 20000    
    MAX_HISTORY_MODELS = 100       
    
    best_eval_score = -999.0
    promotion_streak = 0 # 晋级进度条
    phase_1_start_frame = 0 if shared_phase.value == 1 else None

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("history_models", exist_ok=True) 

    buffers = create_buffers(NUM_BUFFERS)
    ctx = mp.get_context('spawn')
    free_queue, full_queue = ctx.SimpleQueue(), ctx.SimpleQueue()
    for i in range(NUM_BUFFERS): free_queue.put(i)

    actor_processes = []
    for i in range(NUM_ACTORS):
        p = ctx.Process(target=act_worker_unified, args=(i, free_queue, full_queue, shared_actor_model, buffers, shared_max_eps, shared_phase, NUM_ACTORS))
        p.start()
        actor_processes.append(p)

    # 【完美修复】：绝对不能给 frames_processed 清零！
    last_eval_frame, last_history_frame = frames_processed, frames_processed
    start_time = time.time()
    
    # 【新增】：用于计算纯粹的瞬时训练速度，解决 FPS 越跑越低的问题
    last_log_time = time.time()
    last_log_frames = frames_processed
    
    print("\n" + "="*60 + f"\n🔥 洪流已开启! Actor: {NUM_ACTORS} | Batch: {BATCH_SIZE}x{UNROLL_LENGTH}\n" + "="*60 + "\n")

    try:
        while frames_processed < TOTAL_FRAMES:
            # 1. 执行动态调度
            if frames_processed < EPS_DECAY_FRAMES:
                fraction = frames_processed / EPS_DECAY_FRAMES
                shared_max_eps.value = EPS_START - fraction * (EPS_START - EPS_END)
            else:
                shared_max_eps.value = EPS_END
                
            current_lr = MIN_LR + 0.5 * (INITIAL_LR - MIN_LR) * (1 + math.cos(math.pi * frames_processed / TOTAL_FRAMES))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 2. 获取数据
            indices = [full_queue.get() for _ in range(BATCH_SIZE)]
            
            # 【完美细节5：接收 136 维新 Context 和 324 维隐秘标签】
            b_query = torch.stack([buffers['query'][m] for m in indices], dim=0).to(device).view(-1, 216)
            b_context = torch.stack([buffers['context'][m] for m in indices], dim=0).to(device).view(-1, 136)
            b_history = torch.stack([buffers['history'][m] for m in indices], dim=0).to(device).view(-1, 128, 112)
            b_mask = torch.stack([buffers['history_mask'][m] for m in indices], dim=0).to(device).view(-1, 128)
            b_target = torch.stack([buffers['target'][m] for m in indices], dim=0).to(device).view(-1, 1)
            b_hidden_labels = torch.stack([buffers['hidden_labels'][m] for m in indices], dim=0).to(device).view(-1, 324)
            
            # 3. 混合 Loss 训练 (Mind-Reading Task)
            learner_model.train()
            optimizer.zero_grad()
            
            # 打开 return_preds 阀门，呼叫上帝视角预测头
            q_values, hidden_preds = learner_model(b_query, b_context, b_history, b_mask, return_preds=True)
            
            rl_loss = F.mse_loss(q_values, b_target)
            
            if shared_phase.value == 1:
                # 【高阶核心：读心术 Loss】
                # 利用 BCE_With_Logits 直接计算交叉熵，迫使网络骨架具备超强推理能力
                sl_loss = F.binary_cross_entropy_with_logits(hidden_preds, b_hidden_labels)
                loss = rl_loss + 0.5 * sl_loss
            else:
                loss = rl_loss
                sl_loss = torch.tensor(0.0)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learner_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if (frames_processed // (BATCH_SIZE * UNROLL_LENGTH)) % 10 == 0:
                shared_actor_model.load_state_dict(learner_model.state_dict())
            for m in indices: free_queue.put(m)
                
            frames_processed += BATCH_SIZE * UNROLL_LENGTH
            
            if (frames_processed // (BATCH_SIZE * UNROLL_LENGTH)) % 1 == 0:
                # 【完美修复】：只计算最近这一个 Batch 的瞬时速度！绝不会越来越慢！
                current_time = time.time()
                fps = (frames_processed - last_log_frames) / max(1e-5, (current_time - last_log_time))
                last_log_time = current_time
                last_log_frames = frames_processed
                
                phase_tag = "[新手村]" if shared_phase.value == 0 else "[诸神战]"
                print(f"⚡ {phase_tag} 帧数:{frames_processed} | Q-Loss:{rl_loss.item():.4f} | SL-Loss:{sl_loss.item():.4f} | Eps:{shared_max_eps.value:.3f} | fps:{fps:.1f}")

            # 4. 质检与晋级机制 (Auto Curriculum)
            if frames_processed - last_eval_frame >= EVAL_FREQ_FRAMES:
                last_eval_frame = frames_processed
                print("\n" + "🌟"*20 + f"\n[{frames_processed} 帧] 抽取质检员进行段位考核...")
                eval_score, eval_win_rate = evaluate(learner_model, device, shared_phase.value, eval_episodes=20)
                print(f"🎯 考核结果 -> 真实均分: {eval_score:+.2f} | 胜率: {eval_win_rate*100:.1f}%")
                
                checkpoint = {
                    'frames_processed': frames_processed,
                    'model_state_dict': learner_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'phase': shared_phase.value,
                    'promotion_streak': promotion_streak,
                    'best_eval_score': best_eval_score
                }
                torch.save(checkpoint, "checkpoints/v4_unified_checkpoint.pth")
                
                # 在 Phase 0 中，连续碾压 Bot 即可触发飞升！
                # 在 Phase 0 中，连续碾压 Bot 即可触发飞升！
                if shared_phase.value == 0:
                    if eval_win_rate >= 0.5: 
                        promotion_streak += 1
                        print(f"🔥 晋级条积攒中: {promotion_streak}/2 (当前胜率: {eval_win_rate*100:.1f}%)")
                        if promotion_streak >= 2:
                            print("\n" + "🚨"*20 + "\n🔥 突破境界！AI 已参透基础法则，全服切换至 [Phase 1: 诸神之战]！\n" + "🚨"*20 + "\n")
                            shared_phase.value = 1
                            phase_1_start_frame = frames_processed
                            EPS_START = 0.3
                            best_eval_score = -999.0 # 重置最高分基准
                    else:
                        promotion_streak = 0
                else:
                    # Phase 1: 正常记录最高分
                    if eval_score >= best_eval_score:
                        best_eval_score = eval_score
                        torch.save(learner_model.state_dict(), "checkpoints/v4_unified_best.pth")
                        print(f"💾 [霸榜] 保持最高战力！已更新至 v4_unified_best.pth")
                
                print("🌟"*20 + "\n")

            if shared_phase.value == 1 and frames_processed - last_history_frame >= HISTORY_SAVE_FREQ:
                last_history_frame = frames_processed
                history_path = f"history_models/model_frame_{frames_processed}.pth"
                torch.save(learner_model.state_dict(), history_path)
                
                history_files = sorted(glob.glob("history_models/*.pth"), key=os.path.getctime)
                if len(history_files) > MAX_HISTORY_MODELS:
                    os.remove(history_files[0])

    except KeyboardInterrupt: print("\n⚠️ 收到停止指令，正在安全关闭...")
    finally:
        for _ in actor_processes: free_queue.put(None) 
        for p in actor_processes:
            p.join(timeout=3)
            if p.is_alive(): p.terminate()
        print("✅ 训练安全结束！")

if __name__ == '__main__':
    train()