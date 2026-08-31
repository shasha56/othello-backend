# AIプレイヤーを管理する(強化学習アルゴリズムを使って意思決定する)
import random
import torch
from ai.model import QNetwork
import numpy as np

class Agent:
    def __init__(self,size):
        self.model = QNetwork(size)
        self.target_model = QNetwork(size)

        self.target_model.load_state_dict( # self.modelのweight,biasをコピー
            self.model.state_dict()
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001
        )

    def select_random_action(self,valid_actions,rng=random): # 合法手の中からランダムで1つ返す
        selected = rng.choice(valid_actions)

        return selected

    def select_best_action(self,valid_actions,q_values): # 合法手の中から最大Q値のものを返す
        selected_action = max(valid_actions,key=lambda action:q_values[action])

        return selected_action

    def select_epsilon_greedy_action(self,valid_actions,q_values,epsilon=0.0): # 確率epsilonでランダムに、1-epsilonで最大Q値の手を返す。
        probability = random.random()

        if probability < epsilon:
            action = self.select_random_action(valid_actions)
        else:
            action = self.select_best_action(valid_actions,q_values)

        return action


    def get_q_values(self,state): # stateからQ値を求める
        state_tensor = torch.tensor(state, dtype=torch.float32) # Tensor化
        q_values = self.model(state_tensor)
        return q_values

    def learn(self, experiences, target_qs): # weight,biasの更新
        states, actions, rewards, next_states, next_valid_actions, dones = zip(*experiences)

        state_tensors = torch.tensor(np.stack(states), dtype=torch.float32)

        q_values = self.model(state_tensors)
        current_qs = [q_values[i][actions[i]] for i in range(len(actions))]
        current_qs = torch.stack(current_qs)

        loss = torch.nn.functional.mse_loss(
            current_qs,
            target_qs
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def calculate_target_q(self, experiences, gamma=0.99): # target_q計算
        states, actions, rewards, next_states, next_valid_actions, dones = zip(*experiences)

        target_qs = []
        
        for i in range(len(dones)):
            if dones[i]:
                target_q = torch.tensor(
                    rewards[i],
                    dtype=torch.float32
                )
            else:
                next_state_tensor = torch.tensor(
                    next_states[i],
                    dtype=torch.float32
                )

                with torch.no_grad():
                    next_q_values = self.target_model(next_state_tensor)
                    best_next_action = self.select_best_action(next_valid_actions[i],next_q_values)
                    max_next_q = next_q_values[best_next_action]

                    target_q = rewards[i] + gamma * max_next_q

            target_qs.append(target_q)
        
        target_qs = torch.stack(target_qs)

        return target_qs

    def update_target_model(self): # 学習中モデルを target_model にコピー
        self.target_model.load_state_dict(
            self.model.state_dict()
        )