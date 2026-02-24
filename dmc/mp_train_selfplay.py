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
from dmc.mp_actor_selfplay import create_buffers, act_worker_selfplay, UNROLL_LENGTH
from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

def evaluate(model, device, eval_episodes=20):
    """
    【8:2 综合评估局】：
    前 80% (16局) 对战历史自我，测试高阶博弈的进化程度；
    后 20% (4局)  对战规则 Bot，作为常识底线的锚点测试。
    """
    env_wrapper = GuandanEnvWrapper()
    bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}
    
    # 尝试从博物馆里请一位“历史前任”出来作为考官
    history_files = glob.glob("history_models/*.pth")
    history_model = None
    if history_files:
        history_model = GuandanModel(hidden_dim=512).to(device)
        chosen_file = random.choice(history_files)
        # weights_only=True 消除安全警告
        history_model.load_state_dict(torch.load(chosen_file, map_location=device, weights_only=True))
        history_model.eval()

    total_score, wins = 0.0, 0
    model.eval() 
    
    # 【核心逻辑】：计算历史局和Bot局的分割点
    # 如果一开始还没有历史模型，就全部用 Bot 测；有了之后就按 8:2 测
    num_history_games = int(eval_episodes * 0.8) if history_model else 0
    
    for ep in range(eval_episodes):
        obs = env_wrapper.reset(current_level=random.randint(2, 14))
        
        # 决定这局的考官身份
        if ep < num_history_games:
            opponent_type = 'history'
        else:
            opponent_type = 'bot'

        while True:
            cur_player, legal_actions = obs['player_id'], obs['legal_actions']
            if not legal_actions: legal_actions = [[]]
            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = env_wrapper.step([])
                if done: break
                continue
            
            # ==========================================
            # 2v2 公平对决：0号和2号是同一阵营，全部使用最新模型！
            # ==========================================
            if cur_player in [0, 2]:
                with torch.no_grad():
                    scores = model(
                        torch.FloatTensor(obs['x_batch']['query']).to(device),
                        torch.FloatTensor(obs['x_batch']['context']).to(device),
                        torch.FloatTensor(obs['x_batch']['history']).to(device),
                        torch.FloatTensor(obs['x_batch']['history_mask']).to(device)
                    ).squeeze(-1)
                best_action = legal_actions[torch.argmax(scores).item()]
            
            # ==========================================
            # 1号和3号是对立阵营，由考官身份决定用什么大脑
            # ==========================================
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
        
        # 统计得分
        result = result_info['result']
        team_score = float(result['level_up']) if result['winner'] == 'A' else -float(result['level_up'])
        total_score += (team_score - len(env_wrapper.env.players_hand[0]) * 0.05) / 3.0 
        if team_score > 0: wins += 1
        
    return total_score / eval_episodes, wins / eval_episodes

def train():
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Self-Play] 联盟总司令塔启动！运行在: {device}")
    
    learner_model = GuandanModel(hidden_dim=512).to(device)
    
    best_model_path = "checkpoints/v4_moe_best.pth"
    if os.path.exists(best_model_path):
        print(f"🔄 检测到历史最强满级号，正在注入灵魂: {best_model_path} ...")
        learner_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:
        print("🆕 未检测到基础模型，将从零开始自我博弈 (不推荐)...")
        
    shared_actor_model = GuandanModel(hidden_dim=512).to('cpu')
    shared_actor_model.load_state_dict(learner_model.state_dict())
    shared_actor_model.share_memory() 
    shared_actor_model.eval()

    # ==========================================
    # ⚙️ 动态调度配置区 (Hyperparameters)
    # ==========================================
    NUM_ACTORS, NUM_BUFFERS, BATCH_SIZE = 16, 300, 128      
    TOTAL_FRAMES = 10000000 
    
    # 【LR 调度】：初始低调，余弦平滑衰减
    INITIAL_LR = 1e-4 
    MIN_LR = 5e-6
    optimizer = optim.Adam(learner_model.parameters(), lr=INITIAL_LR)
    
    # 【Epsilon 调度】：前半程降至 0.05，后半程恒定
    EPS_START = 0.3
    EPS_END = 0.05
    EPS_DECAY_FRAMES = TOTAL_FRAMES // 2 
    shared_max_eps = mp.Value('d', EPS_START) # 跨进程共享变量！
    
    EVAL_FREQ_FRAMES = 20000      
    HISTORY_SAVE_FREQ = 20000    
    MAX_HISTORY_MODELS = 100       
    
    best_eval_score = -999.0
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("history_models", exist_ok=True) 

    buffers = create_buffers(NUM_BUFFERS)
    ctx = mp.get_context('spawn')
    free_queue, full_queue = ctx.SimpleQueue(), ctx.SimpleQueue()
    for i in range(NUM_BUFFERS): free_queue.put(i)

    actor_processes = []
    for i in range(NUM_ACTORS):
        # 传入 shared_max_eps 内存指针，供 Actor 实时读取
        p = ctx.Process(target=act_worker_selfplay, args=(i, free_queue, full_queue, shared_actor_model, buffers, shared_max_eps, NUM_ACTORS))
        p.start()
        actor_processes.append(p)

    frames_processed, last_eval_frame, last_history_frame = 0, 0, 0
    start_time = time.time()
    
    print("\n" + "="*60 + f"\n🔥 自我博弈联盟已开启! Actor: {NUM_ACTORS} | Batch: {BATCH_SIZE}x{UNROLL_LENGTH}\n" + "="*60 + "\n")

    try:
        while frames_processed < TOTAL_FRAMES:
            # ==========================================
            # 📈 1. 执行动态调度 (LR & Epsilon)
            # ==========================================
            # 更新 Epsilon (折线衰减)
            if frames_processed < EPS_DECAY_FRAMES:
                fraction = frames_processed / EPS_DECAY_FRAMES
                shared_max_eps.value = EPS_START - fraction * (EPS_START - EPS_END)
            else:
                shared_max_eps.value = EPS_END
                
            # 更新 LR (余弦退火衰减)
            current_lr = MIN_LR + 0.5 * (INITIAL_LR - MIN_LR) * (1 + math.cos(math.pi * frames_processed / TOTAL_FRAMES))
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # ==========================================
            # 🧠 2. 数据获取与模型更新
            # ==========================================
            indices = [full_queue.get() for _ in range(BATCH_SIZE)]
            
            b_query = torch.stack([buffers['query'][m] for m in indices], dim=0).to(device).view(-1, 216)
            b_context = torch.stack([buffers['context'][m] for m in indices], dim=0).to(device).view(-1, 112)
            b_history = torch.stack([buffers['history'][m] for m in indices], dim=0).to(device).view(-1, 64, 112)
            b_mask = torch.stack([buffers['history_mask'][m] for m in indices], dim=0).to(device).view(-1, 64)
            b_target = torch.stack([buffers['target'][m] for m in indices], dim=0).to(device).view(-1, 1)
            
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
            
            # 打印包含当前 LR 和 Eps 的日志
            if (frames_processed // (BATCH_SIZE * UNROLL_LENGTH)) % 10 == 0:
                fps = frames_processed / (time.time() - start_time)
                print(f"⚡ [Self-Play] 帧数:{frames_processed} | Loss:{loss.item():.4f} | LR:{current_lr:.2e} | MaxEps:{shared_max_eps.value:.3f} | 吞吐:{fps:.1f} fps")

            # ==========================================
            # 🌟 3. 质检与保存机制
            # ==========================================
            if frames_processed - last_eval_frame >= EVAL_FREQ_FRAMES:
                last_eval_frame = frames_processed
                print("\n" + "🌟"*20 + f"\n[{frames_processed} 帧] 抽取质检员进行人类常识测试...")
                eval_score, eval_win_rate = evaluate(learner_model, device, eval_episodes=20)
                print(f"🎯 质检结果 -> 真实均分: {eval_score:+.2f} | 胜率: {eval_win_rate*100:.1f}%")
                
                torch.save(learner_model.state_dict(), "checkpoints/v4_selfplay_latest.pth")
                if eval_score >= best_eval_score:
                    best_eval_score = eval_score
                    torch.save(learner_model.state_dict(), "checkpoints/v4_selfplay_best.pth")
                    print(f"💾 保持最高战力！已更新至 v4_selfplay_best.pth")
                print("🌟"*20 + "\n")

            if frames_processed - last_history_frame >= HISTORY_SAVE_FREQ:
                last_history_frame = frames_processed
                history_path = f"history_models/model_frame_{frames_processed}.pth"
                torch.save(learner_model.state_dict(), history_path)
                print(f"🏛️ 已将当前版本收入历史博物馆: {history_path}")
                
                history_files = sorted(glob.glob("history_models/*.pth"), key=os.path.getctime)
                if len(history_files) > MAX_HISTORY_MODELS:
                    file_to_delete = history_files[0]
                    os.remove(file_to_delete)
                    print(f"🧹 博物馆容量满载，已清理远古版本: {file_to_delete}")

    except KeyboardInterrupt: print("\n⚠️ 收到停止指令，正在安全关闭...")
    finally:
        for _ in actor_processes: free_queue.put(None) 
        for p in actor_processes:
            p.join(timeout=3)
            if p.is_alive(): p.terminate()
        print("✅ 训练安全结束！")

if __name__ == '__main__':
    train()