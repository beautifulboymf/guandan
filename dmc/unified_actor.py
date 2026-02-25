import os
os.environ["OMP_NUM_THREADS"] = "1" 

import torch
import numpy as np
import random
import traceback 
import glob

from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent
from dmc.models import GuandanModel

UNROLL_LENGTH = 32 
MAX_HISTORY = 128 # 绝对对齐网络容量

def create_buffers(num_buffers):
    T = UNROLL_LENGTH
    specs = dict(
        target=dict(size=(T,), dtype=torch.float32),
        query=dict(size=(T, 216), dtype=torch.float32),
        context=dict(size=(T, 136), dtype=torch.float32), # 112 + 24(大局观特征)
        history=dict(size=(T, MAX_HISTORY, 112), dtype=torch.float32),
        history_mask=dict(size=(T, MAX_HISTORY), dtype=torch.float32),
        hidden_labels=dict(size=(T, 324), dtype=torch.float32) # 【上帝视角】3人 * 108维
    )
    buffers = {key: [] for key in specs}
    for _ in range(num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_())
    return buffers

def act_worker_unified(actor_id, free_queue, full_queue, shared_model, buffers, shared_max_eps, shared_phase, num_actors):
    print(f"🚀 [Actor {actor_id}] 启动！已接入全自动双阶段引擎...")
    torch.set_num_threads(1) 
    
    env_wrapper = GuandanEnvWrapper()
    bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}
    
    local_history_model = GuandanModel(hidden_dim=512).to('cpu')
    local_history_model.eval()
    
    worker_buf = {k: [] for k in ['target', 'query', 'context', 'history', 'history_mask', 'hidden_labels']}
    episodes_done = 0 

    episodes_since_history_update = 999
    
    try:
        while True:
            current_phase = shared_phase.value
            rand_level = random.randint(2, 14)
            obs = env_wrapper.reset(current_level=rand_level) 
            
            # ==========================================
            # 【完美细节1：无缝动态身份切换】
            # ==========================================
            identities = {0: 'latest'} 
            for i in range(1, 4):
                if current_phase == 0:
                    identities[i] = 'bot'
                else:
                    rand_p = random.random()
                    if rand_p < 0.70: identities[i] = 'latest'
                    elif rand_p < 0.90: identities[i] = 'history'
                    else: identities[i] = 'bot'
                    
            if 'history' in identities.values():
                history_files = glob.glob("history_models/*.pth")
                if history_files:
                    # 【核心修改】：不要每局都去读硬盘！每打 20 局才去硬盘里抽一个新历史模型！
                    episodes_since_history_update += 1
                    if episodes_since_history_update >= 20:
                        try:
                            # 读取硬盘，更新本地的历史老怪物
                            local_history_model.load_state_dict(torch.load(random.choice(history_files), map_location='cpu', weights_only=True), strict=False)
                            episodes_since_history_update = 0 # 重置计数器
                        except:
                            for k in identities: 
                                if identities[k] == 'history': identities[k] = 'latest'
                else:
                    for k in identities: 
                        if identities[k] == 'history': identities[k] = 'latest'
            
            ep_query, ep_context, ep_history, ep_mask, ep_hidden = [], [], [], [], []
            
            while True:
                cur_player, legal_actions = obs['player_id'], obs['legal_actions']
                if not legal_actions: legal_actions = [[]]
                if len(legal_actions) == 1 and not legal_actions[0]:
                    obs, reward, done, result_info = env_wrapper.step([])
                    if done: break
                    continue

                identity = identities[cur_player]
                
                if identity == 'latest':
                    x_batch = obs['x_batch']
                    with torch.no_grad():
                        # 注意：推理阶段 return_preds=False，只拿胜率 Q_Value
                        scores = shared_model(
                            torch.FloatTensor(x_batch['query']), torch.FloatTensor(x_batch['context']),
                            torch.FloatTensor(x_batch['history']), torch.FloatTensor(x_batch['history_mask']),
                            inference=True
                        ).squeeze(-1)
                        
                    current_max_eps = shared_max_eps.value
                    epsilon = 0.01 + (current_max_eps - 0.01) * (actor_id / max(1, num_actors - 1))
                    
                    if cur_player == 0 and random.random() < epsilon:
                        action_idx = random.randint(0, len(legal_actions)-1) 
                    else:
                        action_idx = torch.argmax(scores).item()
                    best_action = legal_actions[action_idx]
                    
                elif identity == 'history':
                    x_batch = obs['x_batch']
                    with torch.no_grad():
                        scores = local_history_model(
                            torch.FloatTensor(x_batch['query']), torch.FloatTensor(x_batch['context']),
                            torch.FloatTensor(x_batch['history']), torch.FloatTensor(x_batch['history_mask']),
                            inference=True
                        ).squeeze(-1)
                    action_idx = torch.argmax(scores).item()
                    best_action = legal_actions[action_idx]
                    
                elif identity == 'bot':
                    best_action = bots[cur_player].act(obs['infoset'])
                    action_idx = -1 

                if cur_player == 0:
                    ep_query.append(x_batch['query'][action_idx])
                    ep_context.append(x_batch['context'][action_idx])
                    ep_history.append(x_batch['history'][action_idx])
                    ep_mask.append(x_batch['history_mask'][action_idx])
                    # 【完美细节2：无声无息地偷录上帝视角标签】
                    ep_hidden.append(x_batch['hidden_labels'][action_idx])
                    
                obs, reward, done, result_info = env_wrapper.step(best_action)
                if done: break
            
            # ==========================================
            # 【完美细节3：大巧不工的纯稀疏奖励】
            # 取消一切过程加分，只用最终胜负（-1.0 ~ +1.0）驱动长远规划
            # ==========================================
            result = result_info['result']
            team_score = float(result['level_up']) if result['winner'] == 'A' else -float(result['level_up'])
            scaled_reward = team_score / 3.0 
            
            worker_buf['target'].extend([scaled_reward] * len(ep_query))
            worker_buf['query'].extend(ep_query)
            worker_buf['context'].extend(ep_context)
            worker_buf['history'].extend(ep_history)
            worker_buf['history_mask'].extend(ep_mask)
            worker_buf['hidden_labels'].extend(ep_hidden)
            
            episodes_done += 1
            if episodes_done % 10 == 0: 
                phase_str = "新手村" if current_phase == 0 else "诸神之战"
                print(f"✅ [Actor {actor_id}] 已完成 {episodes_done} 局 ({phase_str}) ...")
            
            while len(worker_buf['target']) >= UNROLL_LENGTH:
                idx = free_queue.get() 
                if idx is None: return 
                for t in range(UNROLL_LENGTH):
                    buffers['target'][idx][t] = worker_buf['target'][t]
                    buffers['query'][idx][t] = torch.from_numpy(worker_buf['query'][t])
                    buffers['context'][idx][t] = torch.from_numpy(worker_buf['context'][t])
                    buffers['history'][idx][t] = torch.from_numpy(worker_buf['history'][t])
                    buffers['history_mask'][idx][t] = torch.from_numpy(worker_buf['history_mask'][t])
                    buffers['hidden_labels'][idx][t] = torch.from_numpy(worker_buf['hidden_labels'][t])
                for k in worker_buf: worker_buf[k] = worker_buf[k][UNROLL_LENGTH:]
                full_queue.put(idx)

    except KeyboardInterrupt: pass
    except Exception as e:
        print(f"\n❌ [Actor {actor_id}] 发生崩溃:"); traceback.print_exc()