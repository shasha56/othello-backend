# 実際に遊ぶ実行スクリプト
from roles.env import Board
from ai.agent import Agent
import torch

def play_human_to_human(size): # 人間 vs 人間
    board = Board(size)

    while True:
        player = "黒" if board.turn == -1 else "白"
        board.print_board()
        coordinate = input(f"{player} のターン：").split()
        row, col = int(coordinate[0]), int(coordinate[1])
        action = board.step(row,col)

        match(action):
            case "finish":
                board.print_board()
                winner = board.get_winner()
                match(winner):
                    case -1:
                        print("黒の勝利！")
                    case 1:
                        print("白の勝利！")
                    case 0:
                        print("引き分け！")
                    case _:
                        print("???")
                return
            case "pass":
                pass_player = "黒" if board.turn == 1 else "白"
                print(f"{pass_player} はパスしました")
                continue
            case "next":
                print(f"次の番です")
                continue
            case "retry":
                print("やり直してください")
                continue
        print()

def play_human_to_ai(size,human_turn,model_name): # AI vs 人間
    board = Board(size)

    ai = Agent(size)
    checkpoint = torch.load(model_name, weights_only=False)
    ai.model.load_state_dict(checkpoint["model_state_dict"])

    while True:
        current_player = board.turn
        player_name = "プレイヤー" if board.turn == human_turn else "AI"
        board.print_board()
        print(f"{player_name} のターン：")

        current_state = board.get_state()
        moves = board.get_valid_moves()
        valid_actions = board.get_valid_actions(moves)

        if current_player == human_turn: # 人間のターン
            coordinate = input().split()
            if len(coordinate) != 2:
                continue
            row, col = int(coordinate[0]), int(coordinate[1])
        else:
            with torch.no_grad():
                q_values = ai.get_q_values(current_state)
                action = ai.select_best_action(valid_actions,q_values)
                row, col = board.action_to_position(action)

        step_result = board.step(row,col)

        match(step_result):
            case "finish":
                board.count_stones()
                winner = board.get_winner()
                match(winner):
                    case 0:
                        print("引き分け！")
                    case _ if winner == human_turn:
                        print("プレイヤーの勝利！")
                    case _ if winner == (human_turn * -1):
                        print("AIの勝利！")
                    case _:
                        print("???")
                return
            case "pass":
                pass_player = "プレイヤー" if board.turn == human_turn else "AI"
                print(f"{pass_player} はパスしました")
                continue
            case "next":
                print(f"次の番です")
                continue
            case "retry":
                print("やり直してください")
                continue
        print()


if __name__ == "__main__":
    # play_human_to_human(8)
    play_human_to_ai(8,-1,"models\checkpoints\gammas\gamma0.995.pth")