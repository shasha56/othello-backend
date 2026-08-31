# 学習したAIの評価
from ai.agent import Agent
from roles.env import Board
import torch
import math

def play_evaluation_game(size, evaluation_agent, random_agent, evaluation_agent_turn, rng): # AI vs ランダムの対戦

    board = Board(size)

    while True:
        player = board.turn

        current_state = board.get_state()
        valid_actions = board.get_valid_actions()

        if player == evaluation_agent_turn: # 学習AIのターン
            with torch.no_grad():
                q_values = evaluation_agent.get_q_values(current_state)
            selected_action = evaluation_agent.select_best_action(valid_actions,q_values) # 最大Q値から選ぶ
        else: # ランダムAIのターン
            selected_action = random_agent.select_random_action(valid_actions,rng) # ランダムな一手

        row, col = board.action_to_position(selected_action)
        step_result = board.step(row,col)

        match(step_result):
            case "finish":
                return board.get_winner()
            case "pass":
                continue
            case "next":
                continue
            case "retry":
                continue

def evaluate(size,evaluate_game_num,evaluation_agent,rng):

    random_agent = Agent(size) # 改善の余地あり

    evaluation_agent.model.eval() # 評価モード

    #学習AI = 黒
    black_win, black_lose, black_draw = 0, 0, 0

    for _ in range(evaluate_game_num // 2):
        result = play_evaluation_game(size, evaluation_agent,random_agent, -1, rng)

        match(result):
            case -1: # 黒の勝利
                black_win += 1
            case 1: # 黒の敗北
                black_lose += 1
            case 0: # 引き分け
                black_draw += 1

    #学習AI = 白
    white_win, white_lose, white_draw = 0, 0, 0

    for _ in range(evaluate_game_num // 2):
        result = play_evaluation_game(size, evaluation_agent, random_agent, 1, rng)

        match(result):
            case -1: # 白の敗北
                white_lose += 1
            case 1: # 白の勝利
                white_win += 1
            case 0: # 引き分け
                white_draw += 1

    return (black_win + white_win) / evaluate_game_num # 勝率を返す