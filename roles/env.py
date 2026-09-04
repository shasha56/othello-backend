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

            move_row = row + d_row
            move_col = col + d_col

            if not self.is_ranged(move_row,move_col) or self.board[move_row][move_col] != opponent_stone:
                continue

            tem_stones = [(move_row,move_col)]

            move_row += d_row
            move_col += d_col

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
                if self.board[row][col] != 0:
                    continue
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
                        cell = "●"
                    case 1:
                        cell = "○"
                    case _:
                        cell = "?" 
                print(f" {cell} |",end="")
            print("\n   +",end="")
            for _ in range(self.size):
                print("---+",end="")
            print()

    def pass_turn(self,moves): # パスの処理
        if not moves:
            self.turn *= -1
            return True # パスを行った
        return False

    def is_done(self,player_moves,opponent_moves): # ゲーム終了判定
        if not player_moves and not opponent_moves:
            return True # ゲーム終了
        return False

    def step(self,row,col): # 石を置く一連の操作
        if self.set_stone(row,col):
            player_moves = self.get_valid_moves(self.turn)

            if player_moves:
                return "next"
            else:
                opponent_moves = self.get_valid_moves(-self.turn)
                if not opponent_moves:
                    return "finish" #終了
                else:
                    self.turn *= -1
                    return "pass" #パス
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
    
    def get_valid_actions(self,moves): # 合法手を action の番号 として取得
        moves_action = []

        for row,col in moves:
            moves_action.append(row * self.size + col)

        return moves_action

    def get_valid_moves_rev(self,turn=None): # 置ける場所全探索（改良版）
        moves = {}
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != 0:
                    continue

                moves_position = []
                turn = self.turn if turn is None else turn

                for d_row, d_col in DIRECTION:
                    move_row = row + d_row
                    move_col = col + d_col

                    if self.is_ranged(move_row,move_col) and self.board[move_row][move_col] == -turn:
                        moves_position.extend(self.reverse_stone_rev(row,col,d_row,d_col,turn))

                if moves_position:
                    moves[(row,col)] = moves_position        
        return moves

    def get_valid_actions_rev(self,moves): # 合法手を action の番号 として取得(改良版)
        moves_action = []

        for row,col in moves.keys():
            moves_action.append(row * self.size + col)

        return moves_action

    def step_rev(self,row,col,player_moves): # 石を置く一連の操作
        if self.set_stone_rev(row,col,player_moves):
            player_moves = self.get_valid_moves_rev2(self.turn)

            if player_moves:
                return "next"
            else:
                opponent_moves = self.get_valid_moves_rev2(-self.turn)
                if not opponent_moves:
                    return "finish" #終了
                else:
                    self.turn *= -1
                    return "pass" #パス
        else:
            return "retry" #置き直し

    def set_stone_rev(self,row,col,player_moves): # 石の設置

        if not (row,col) in player_moves:
            return False
        
        reverse_stones = player_moves[(row,col)]

        self.board[row][col] = self.turn
        for r_row,r_col in reverse_stones:
            self.board[r_row][r_col] = self.turn

        self.turn *= -1
        return True

    def reverse_stone_rev(self,row,col,d_row,d_col,turn): # ひっくり返せる石の列挙(改良版),一つの方向

        if not self.is_ranged(row,col) or self.board[row][col] != 0:
            return []

        opponent_stone = -1 * turn
        reverse_stones = []

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

    def get_valid_moves_rev2(self,turn=None): # 置ける場所全探索（改良版2）
        moves = {}
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != 0:
                    continue

                moves_position = self.reverse_stone(row,col,turn)
                if moves_position:
                    moves[(row,col)] = moves_position        
        return moves

    def count_stones(self): # 石のカウント
        black_stones, white_stones = 0, 0
        for row in range(self.size):
            for col in range(self.size):
                match(self.board[row][col]):
                    case 1:
                        white_stones += 1
                    case -1:
                        black_stones += 1
        print(f"黒 : {black_stones}, 白 : {white_stones}")
        return black_stones, white_stones