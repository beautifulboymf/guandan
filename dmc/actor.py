import torch
import numpy as np
import random
import os
import sys

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dmc.env_wrapper import GuandanEnvWrapper
from dmc.agents import HeuristicAgent

class Actor:
    def __init__(self, model, epsilon=0.1):
        self.model = model
        self.epsilon = epsilon 
        self.wrapper = GuandanEnvWrapper()
        
        self.bots = {i: HeuristicAgent(player_id=i) for i in range(1, 4)}

    def play_episode(self):
        obs = self.wrapper.reset(current_level=2) 
        episode_data = [] 
        
        while True:
            cur_player = obs['player_id']
            legal_actions = obs['legal_actions']
            
            if not legal_actions: legal_actions = [[]]

            if len(legal_actions) == 1 and not legal_actions[0]:
                obs, reward, done, result_info = self.wrapper.step([])
                if done: break
                continue

            # === AI 决策回合 ===
            if cur_player == 0:
                x_batch = obs['x_batch']
                
                device = next(self.model.parameters()).device
                
                query_t = torch.FloatTensor(x_batch['query']).to(device)
                context_t = torch.FloatTensor(x_batch['context']).to(device)
                history_t = torch.FloatTensor(x_batch['history']).to(device)
                mask_t = torch.FloatTensor(x_batch['history_mask']).to(device)
                
                self.model.eval()
                with torch.no_grad():
                    scores = self.model(query_t, context_t, history_t, mask_t).squeeze(-1)
                
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, len(legal_actions) - 1)
                else:
                    action_idx = torch.argmax(scores).item()
                    
                best_action = legal_actions[action_idx]
                
                # 【核心优化 1】：过程微奖励 (Dense Reward)
                # 鼓励 AI 出顺子/连对等组合牌，每跑一张牌当步给 0.01 的小奖励
                step_reward = len(best_action) * 0.01 
                
                episode_data.append({
                    'query': x_batch['query'][action_idx],
                    'context': x_batch['context'][action_idx],
                    'history': x_batch['history'][action_idx],
                    'history_mask': x_batch['history_mask'][action_idx],
                    'step_reward': step_reward # 暂存步级奖励
                })
                
                obs, reward, done, result_info = self.wrapper.step(best_action)
                
            # === 规则 Bot 回合 ===
            else:
                action = self.bots[cur_player].act(obs['infoset'])
                obs, reward, done, result_info = self.wrapper.step(action)
                
            if done: break
                
        # === 终局结算 ===
        result = result_info['result']
        winner_team = result['winner']      
        level_up = result['level_up']       
        
        if winner_team == 'A':
            team_score = float(level_up)    
        else:
            team_score = -float(level_up)   
            
        remaining_cards = len(self.wrapper.env.players_hand[0])
        penalty = remaining_cards * 0.05
        
        # 【核心优化 2】：数值归一化防崩溃
        base_final_reward = team_score - penalty
        scaled_final_reward = base_final_reward / 3.0 # 将分数压缩进 [-1.5, 1.0] 附近
        
        # 将微奖励融合进最终 Target 中
        for data in episode_data:
            data['target'] = scaled_final_reward + data['step_reward']
            del data['step_reward'] # 清理内存
            
        return episode_data, scaled_final_reward