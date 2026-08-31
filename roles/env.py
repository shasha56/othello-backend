# オセロの盤面クラス(汎用化)
import numpy as np

DIRECTION = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class Board:
    def __init__ (self,size): # 初期化
        # -1:黒, 1:白, 0:空白
        self.reset(size)

    def reset(self,size): # 盤面、先攻をリセット
        self.size = size

        self.board = np.zeros((self.size,self.size), dtype=np.int8)

        center = self.size // 2

        self.board[center - 1][center - 1] = 1 # 左上
        self.board[center - 1][center] = -1 # 右上
        self.board[center][center-1] = -1 # 左下
        self.board[center][center] = 1 # 右下
        
        self.turn = -1 # 初手 黒

    def is_ranged (self,row,col): # インデックスが盤面内かチェック
        return (0 <= row <= self.size-1) and (0 <= col <= self.size-1)

    
    def reverse_stone(self,row,col,turn=None): # ひっくり返せる石の列挙
        if not self.is_ranged(row,col) or self.board[row][col] != 0:
            return []

        turn = self.turn if turn is None else turn
        opponent_stone = -1 * turn
        reverse_stones = []

        for d_row, d_col in DIRECTION:
            tem_stones = []

            move_row = row + d_row
            move_col = col + d_col

            while self.is_ranged(move_row,move_col):
            
                if  self.board[move_row][move_col] == opponent_stone:
                    tem_stones.append((move_row,move_col))
                elif self.board[move_row][move_col] == turn:
                    reverse_stones.extend(tem_stones)
                    break
                else:
                    break
        
                move_row += d_row
                move_col += d_col

        return reverse_stones

    def set_stone(self,row,col): # 石の設置
        reverse_stones = self.reverse_stone(row,col)

        if not reverse_stones:
            return False

        self.board[row][col] = self.turn
        for r_row,r_col in reverse_stones:
            self.board[r_row][r_col] = self.turn

        self.turn *= -1
        return True

    def get_valid_moves(self,turn=None): # 置ける場所全探索
        moves = []
        for row in range(self.size):
            for col in range(self.size):
                # print(f"({row} {col}) -> {self.reverse_stone(row,col)}")
                if self.reverse_stone(row,col,turn):
                    moves.append((row,col))
        
        return moves
    
    def print_board(self): # 盤面出力
        print("    ",end="")
        for i in range(self.size):
            print(f" {i}  ",end="")
        print("\n   +",end="")
        for _ in range(self.size):
            print("---+",end="")
        print()
        for row in range(self.size):
            print(f" {row} |",end="")
            for col in range(self.size):
                match(self.board[row][col]):
                    case 0:
                        cell = " "
                    case -1:
                        cell = "×"
                    case 1:
                        cell = "○"
                    case _:
                        cell = "?" 
                print(f" {cell} |",end="")
            print("\n   +",end="")
            for _ in range(self.size):
                print("---+",end="")
            print()

    def pass_turn(self): # パスの処理
        if not self.get_valid_moves():
            self.turn *= -1
            return True # パスを行った
        return False

    def is_done(self): # ゲーム終了判定
        if not self.get_valid_moves(-1) and not self.get_valid_moves(1):
            return True # ゲーム終了
        return False

    def step(self,row,col): # 石を置く一連の操作
        if self.set_stone(row,col):
            if self.is_done():
                return "finish" #終了
            elif self.pass_turn():
                return "pass" #パス
            else:
                return "next"
        else:
            return "retry" #置き直し

    def get_winner(self): # 勝敗判定
        count = np.sum(self.board)
        if count < 0:
            return -1
        elif count > 0:
            return 1
        else:
            return 0

    def action_to_position(self,action): # action(整数)から座標変換(row,col)
        row = action // self.size
        col = action % self.size

        return row, col

    def get_state(self): # 盤面を1次元に変換
        
        return (self.board * self.turn).reshape(-1) # -1は要素数からその次元のサイズを自動計算
    
    def get_valid_actions(self): # 合法手を action の番号 として取得
        moves_position = self.get_valid_moves()
        moves_action = []

        for row,col in moves_position:
            moves_action.append(row * self.size + col)

        return moves_action