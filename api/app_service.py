# 人間の手を反映、AIに打たせる、盤面を返す
from roles.env import Board
from ai.agent import Agent
import torch

board = Board(8)
agent = Agent(8)

model_name = "../models/gamma0.995.pth" # AIモデル

# checkpoint = torch.load(model_name, weights_only=False)
# agent.model.load_state_dict(checkpoint["model_state_dict"])

def reset_game(): # 盤面の初期化
    global board
    board = Board(8)

def play_turn(action=None): # 一手進める
    boards = [] # 連続した盤面の保存
    if action is not None:
        # 人間の手
        row, col = board.action_to_position(action)
        human_step_result = board.step(row,col)
        human_board = board.board.copy() # 人間着手後の盤面
        boards.append(human_board)

        match(human_step_result):
            case "finish":
                winner = board.get_winner()
                match(winner):
                    case 0:
                        return "draw", boards
                    case 1:
                        return "white", boards
                    case -1:
                        return "black", boards
            case "pass":
                return "pass", boards

    while True:
        with torch.no_grad():
            current_state = board.get_state()
            valid_actions = board.get_valid_actions()
            q_values = agent.get_q_values(current_state)
            action = agent.select_best_action(valid_actions,q_values)
            row, col = board.action_to_position(action)

            ai_step_result = board.step(row,col)
            ai_board = board.board.copy() # AI着手後の盤面
            boards.append(ai_board)

            match(ai_step_result):
                case "finish":
                    winner = board.get_winner()
                    match(winner):
                        case 0:
                            return "draw", boards
                        case 1:
                            return "white", boards
                        case -1:
                            return "black", boards
                case "pass":
                    continue
                case "next":
                    return "next", boards

def get_board(): # 盤面の取得
    return board

def is_move(action): # 合法手の確認
    row, col = board.action_to_position(action)
    moves = board.get_valid_moves()

    if (row,col) in moves:
        return True
    else:
        return False
