import torch
import numpy as np
import random
import copy
import traceback # 用于打印详细报错
from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

# 每块 (Chunk) 包含的样本数。Actor 跨局攒够 32 步就打包发给 Learner
UNROLL_LENGTH = 32 
MAX_HISTORY = 40

def create_buffers(num_buffers):
    T = UNROLL_LENGTH
    specs = dict(
        target=dict(size=(T,), dtype=torch.float32),
        query=dict(size=(T, 216), dtype=torch.float32),
        context=dict(size=(T, 112), dtype=torch.float32),
        history=dict(size=(T, MAX_HISTORY, 112), dtype=torch.float32),
        history_mask=dict(size=(T, MAX_HISTORY), dtype=torch.float32),
    )
    
    buffers = {key: [] for key in specs}
    for _ in range(num_buffers):
        for key in buffers:
            _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
            buffers[key].append(_buffer)
            
    return buffers

def act_worker(actor_id, free_queue, full_queue, shared_model, buffers, epsilon):
    print(f"🚀 [Actor {actor_id}] 启动！正在并行生成数据...")
    torch.set_num_threads(1) 
    
    env_wrapper = GuandanEnvWrapper()
    bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}
    
    # 【修复核心】：跨对局的持续收集桶（绝对不在局间清空！）
    worker_buf = {
        'target': [], 'query': [], 'context': [], 'history': [], 'history_mask': []
    }
    
    try:
        while True:
            rand_level = random.randint(2, 14)
            obs = env_wrapper.reset(rand_level) 
            
            # 单局临时记录表（仅用于最后算分）
            ep_query, ep_context, ep_history, ep_mask, ep_step_reward = [], [], [], [], []
            
            while True:
                cur_player = obs['player_id']
                legal_actions = obs['legal_actions']
                
                if not legal_actions: legal_actions = [[]]
                if len(legal_actions) == 1 and not legal_actions[0]:
                    obs, reward, done, result_info = env_wrapper.step([])
                    if done: break
                    continue

                # ==========================================
                # === AI 决策回合 ===
                # ==========================================
                if cur_player == 0:
                    x_batch = obs['x_batch']
                    
                    q_t = torch.FloatTensor(x_batch['query'])
                    c_t = torch.FloatTensor(x_batch['context'])
                    h_t = torch.FloatTensor(x_batch['history'])
                    m_t = torch.FloatTensor(x_batch['history_mask'])
                    
                    with torch.no_grad():
                        scores = shared_model(q_t, c_t, h_t, m_t).squeeze(-1)
                    
                    if random.random() < epsilon:
                        action_idx = random.randint(0, len(legal_actions) - 1)
                    else:
                        action_idx = torch.argmax(scores).item()
                        
                    best_action = legal_actions[action_idx]
                    step_reward = len(best_action) * 0.01 
                    
                    # 存入本局临时记录
                    ep_query.append(x_batch['query'][action_idx])
                    ep_context.append(x_batch['context'][action_idx])
                    ep_history.append(x_batch['history'][action_idx])
                    ep_mask.append(x_batch['history_mask'][action_idx])
                    ep_step_reward.append(step_reward)
                    
                    obs, reward, done, result_info = env_wrapper.step(best_action)
                    
                # ==========================================
                # === 规则 Bot 回合 ===
                # ==========================================
                else:
                    action = bots[cur_player].act(obs['infoset'])
                    obs, reward, done, result_info = env_wrapper.step(action)
                    
                if done: 
                    break
            
            # ==========================================
            # === 一局结束：算总分并灌入大桶 ===
            # ==========================================
            result = result_info['result']
            team_score = float(result['level_up']) if result['winner'] == 'A' else -float(result['level_up'])
            penalty = len(env_wrapper.env.players_hand[0]) * 0.05
            scaled_reward = (team_score - penalty) / 3.0 
            
            ep_target = [scaled_reward + sr for sr in ep_step_reward]
            
            # 汇总到跨局收集大桶里
            worker_buf['target'].extend(ep_target)
            worker_buf['query'].extend(ep_query)
            worker_buf['context'].extend(ep_context)
            worker_buf['history'].extend(ep_history)
            worker_buf['history_mask'].extend(ep_mask)
            
            # 【核心搬运逻辑】：桶里满 32 步，就切一块发走
            while len(worker_buf['target']) >= UNROLL_LENGTH:
                idx = free_queue.get() 
                if idx is None: return 
                
                for t in range(UNROLL_LENGTH):
                    buffers['target'][idx][t] = worker_buf['target'][t]
                    buffers['query'][idx][t] = torch.from_numpy(worker_buf['query'][t])
                    buffers['context'][idx][t] = torch.from_numpy(worker_buf['context'][t])
                    buffers['history'][idx][t] = torch.from_numpy(worker_buf['history'][t])
                    buffers['history_mask'][idx][t] = torch.from_numpy(worker_buf['history_mask'][t])
                
                # 裁掉已经装箱的数据
                for k in worker_buf:
                    worker_buf[k] = worker_buf[k][UNROLL_LENGTH:]
                
                # 叫 GPU 来收货
                full_queue.put(idx)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ [Actor {actor_id}] 发生崩溃:")
        traceback.print_exc()