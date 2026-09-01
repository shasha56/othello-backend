# 人間の手を反映、AIに打たせる、盤面を返す
from roles.env import Board

board = Board(8)

def reset_game(): # 盤面の初期化
    global board
    board = Board(8)

def play_turn(action): # 一手進める
    row, col = board.action_to_position(action)
    step_result = board.step(row,col)

def get_board(): 盤面の取得
    return board