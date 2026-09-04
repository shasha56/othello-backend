# 人間の手を反映、AIに打たせる、盤面を返す
from roles.env import Board
from ai.agent import Agent
import torch
from pathlib import Path

board = Board(8)
agent = Agent(8)

model_dir = Path("models")
files = list(model_dir.glob("*.pth"))

if len(files) == 1:
    checkpoint = torch.load(files[0], weights_only=False)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
agent.model.eval()

def reset_game(): # 盤面の初期化
    global board
    board = Board(8)

def play_turn(action=None): # 一手進める
    boards = [] # 連続した盤面の保存
    actions = [] # 連続した次の合法手の保存
    turns = [] # 連続したターンの保存
    stones = []

    if action is not None:
        # 人間の手
        row, col = board.action_to_position(action)
        human_step_result = board.step(row,col)
        human_board = board.board.copy() # 人間着手後の盤面
        human_next_actions = board.get_valid_actions(board.get_valid_moves())
        boards.append(human_board)
        actions.append(human_next_actions)
        turns.append(board.turn)
        black_stones, white_stones = board.count_stones()
        stones.append((black_stones, white_stones))

        match(human_step_result):
            case "finish":
                winner = board.get_winner()
                match(winner):
                    case 0:
                        return "draw", boards, actions, turns, stones
                    case 1:
                        return "white", boards, actions, turns, stones
                    case -1:
                        return "black", boards, actions, turns, stones
            case "pass":
                return "pass", boards, actions, turns, stones

    while True:
        with torch.no_grad():
            current_state = board.get_state()
            moves = board.get_valid_moves()
            valid_actions = board.get_valid_actions(moves)
            q_values = agent.get_q_values(current_state)
            action = agent.select_best_action(valid_actions,q_values)
            row, col = board.action_to_position(action)

            ai_step_result = board.step(row,col)
            ai_board = board.board.copy() # AI着手後の盤面
            ai_next_actions = board.get_valid_actions(board.get_valid_moves())
            boards.append(ai_board)
            actions.append(ai_next_actions)
            turns.append(board.turn)
            black_stones, white_stones = board.count_stones()
            stones.append((black_stones, white_stones))

            match(ai_step_result):
                case "finish":
                    winner = board.get_winner()
                    match(winner):
                        case 0:
                            return "draw", boards, actions, turns, stones
                        case 1:
                            return "white", boards, actions, turns, stones
                        case -1:
                            return "black", boards, actions, turns, stones
                case "pass":
                    continue
                case "next":
                    return "next", boards, actions, turns, stones

def get_board(): # 盤面の取得
    return board

def get_turn(): # ターンの取得
    return board.turn

def is_move(action): # 合法手の確認
    row, col = board.action_to_position(action)
    moves = board.get_valid_moves()

    if (row,col) in moves:
        return True
    else:
        return False

def valid_actions(): # 合法手の取得
    moves = board.get_valid_moves()
    actions = board.get_valid_actions(moves)

    return actions

def count_stone(): # 石数カウント
    black_stones, white_stones = board.count_stones()

    return black_stones, white_stones