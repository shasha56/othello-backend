# Agent同士を1ゲーム最後まで回す
from roles.env import Board
from ai.agent import Agent
import torch
from collections import deque
import random
from evaluate import evaluate
import math


def play_game(size,agent,epsilon=0.0): # 自己対戦用
    board = Board(size)

    experiences = [] # (state, action, reward, next_state, next_valid_actions, done)
        
    previous_state = {-1:None, 1:None}
    previous_action = {-1:None, 1:None}

    while True:
        player = board.turn

        current_state = board.get_state()
        
        valid_actions = board.get_valid_actions()
        q_values = agent.get_q_values(current_state)
        selected_action = agent.select_epsilon_greedy_action(valid_actions,q_values,epsilon) # 確率で最大Q値orランダムな一手を選ぶ

        row, col = board.action_to_position(selected_action)

        if previous_state[player] is not None:
            experiences.append([previous_state[player], previous_action[player], 0, current_state, valid_actions, False])

        previous_action[player] = selected_action
        previous_state[player] = current_state

        step_result = board.step(row,col)

        match(step_result):
            case "finish":
                winner = board.get_winner()
                match(winner):
                    case -1:
                        black_reward, white_reward = 1, -1
                    case 1:
                        black_reward, white_reward = -1, 1
                    case 0:
                        black_reward, white_reward = 0, 0

                if previous_state[-1] is not None:
                    experiences.append([previous_state[-1], previous_action[-1], black_reward, None, None, True])
                if previous_state[1] is not None:
                    experiences.append([previous_state[1], previous_action[1], white_reward, None, None, True])
                break
            case "pass":
                pass_player = "黒" if board.turn == 1 else "白"
                continue
            case "next":
                continue
            case "retry":
                continue

    return experiences

def agent_continuous_learn(size,max_learn_game_num,epsilon,gamma,batch_size,target_update_interval,learn_game_step, agent=None,seed=0,model_name=None): # 連続した学習(bufferの維持)

    if model_name is None:
        model_name = f"models/checkpoints/seed_{seed}_game_{game_num}.pth"

    eval_rng = random.Random(10000)

    evaluate_game_num = 500 # 評価時ゲーム数
    evaluate_times = 5 # 評価回数

    print(f"モデル評価回数 : {evaluate_times}")

    if agent is None:
        agent = Agent(size)

    buffer = deque(maxlen=1000)

    informations = []

    # before = agent.model.fc1.weight.clone()

    learn_count = 0

    print("累積ゲーム数 : 0回")
    mean_win_rate = 0 # 同一モデルを複数回評価したときの平均勝率
    mean_square = 0

    for _ in range(evaluate_times):
        tem_evaluate_win_rate = evaluate(size, evaluate_game_num, agent,eval_rng)
        mean_win_rate += tem_evaluate_win_rate
        mean_square += (tem_evaluate_win_rate ** 2)

    mean_win_rate /= evaluate_times
    mean_square /= evaluate_times

    variance = mean_square - (mean_win_rate ** 2)
    print(f"モデル平均勝率 : {mean_win_rate * 100:.1f}%")
    print(f"モデルばらつき（標準偏差） : {math.sqrt(variance) * 100:.1f}ポイント")

    informations.append([size, seed, 0, mean_win_rate, math.sqrt(variance), epsilon, gamma, batch_size, target_update_interval])


    for i in range(max_learn_game_num):

        game_num = i + 1

        buffer.extend(play_game(size,agent,epsilon))
        
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            target_qs = agent.calculate_target_q(batch,gamma)
            loss = agent.learn(batch,target_qs)
            # print(loss)
            learn_count += 1

        if learn_count % target_update_interval == 0:
            agent.update_target_model()

        if game_num % learn_game_step == 0: # 途中経過(次に評価をする)
            print(f"累積ゲーム数 : {game_num}回")
            mean_win_rate = 0 # 同一モデルを複数回評価したときの平均勝率
            mean_square = 0

            for _ in range(evaluate_times):
                tem_evaluate_win_rate = evaluate(size, evaluate_game_num, agent, eval_rng)
                mean_win_rate += tem_evaluate_win_rate
                mean_square += (tem_evaluate_win_rate ** 2)

            mean_win_rate /= evaluate_times
            mean_square /= evaluate_times

            variance = mean_square - (mean_win_rate ** 2)
            print(f"モデル平均勝率 : {mean_win_rate * 100:.1f}%")
            print(f"評価ばらつき（標準偏差） : {math.sqrt(variance) * 100:.1f}ポイント")

            informations.append([size, seed, game_num, mean_win_rate, math.sqrt(variance), epsilon, gamma, batch_size, target_update_interval])

            checkpoint = {
                "game_num": game_num,
                "seed": seed,
                "model_state_dict": agent.model.state_dict(),
                "target_model_state_dict": agent.target_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "learn_count": learn_count,
                "buffer": buffer,
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "learn_game_step": learn_game_step,
                "epsilon": epsilon,
                "gamma": gamma,
                "batch_size": batch_size,
                "max_learn_game_num": max_learn_game_num,
                "target_update_interval": target_update_interval,
                "eval_random_state": eval_rng.getstate(),
                "board_size": size
            }

            torch.save( # モデルの保存
                checkpoint,
                model_name
            )
            # torch.save( # モデルの保存(colab用)
            #     checkpoint,
            #     f"/content/drive/MyDrive/Colab Notebooks/プライベート/projects/Miniothello/models/checkpoints/seed_{seed}_game_{game_num}.pth"
            # )
    
    return informations

def agent_resume_learn(size,checkpoint_path, max_learn_game_num=None, learn_game_step=None): #学習を途中から再開
    # 定義フェーズ
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    size = checkpoint["board_size"]

    agent = Agent(size)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.target_model.load_state_dict(checkpoint["target_model_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    seed = checkpoint["seed"]
    buffer = checkpoint["buffer"]
    game_num = checkpoint["game_num"]
    learn_count = checkpoint["learn_count"]
    epsilon = checkpoint["epsilon"]
    gamma = checkpoint["gamma"]
    batch_size = checkpoint["batch_size"]
    target_update_interval = checkpoint["target_update_interval"]
    if max_learn_game_num is None:
        max_learn_game_num = checkpoint["max_learn_game_num"]
    if learn_game_step is None:
        learn_game_step = checkpoint["learn_game_step"]

    random.setstate(checkpoint["python_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    eval_rng = random.Random()
    eval_rng.setstate(checkpoint["eval_random_state"])

    # 学習フェーズ
    evaluate_game_num = 1000 # 評価時ゲーム数
    evaluate_times = 5 # 評価回数

    print(f"モデル評価回数 : {evaluate_times}")

    informations = []

    for game_num in range(game_num + 1,max_learn_game_num + 1):

        buffer.extend(play_game(size,agent,epsilon))
        
        if len(buffer) >= batch_size:
            batch = random.sample(buffer, batch_size)
            target_qs = agent.calculate_target_q(batch,gamma)
            loss = agent.learn(batch,target_qs)
            # print(loss)
            learn_count += 1

        if learn_count % target_update_interval == 0:
            agent.update_target_model()

        if game_num % learn_game_step == 0: # 途中経過(次に評価をする)
            # 評価フェーズ
            print(f"累積ゲーム数 : {game_num}回")
            mean_win_rate = 0 # 同一モデルを複数回評価したときの平均勝率
            mean_square = 0

            for _ in range(evaluate_times):
                tem_evaluate_win_rate = evaluate(size, evaluate_game_num, agent, eval_rng)
                mean_win_rate += tem_evaluate_win_rate
                mean_square += (tem_evaluate_win_rate ** 2)

            mean_win_rate /= evaluate_times
            mean_square /= evaluate_times

            variance = mean_square - (mean_win_rate ** 2)
            print(f"モデル平均勝率 : {mean_win_rate * 100:.1f}%")
            print(f"評価ばらつき（標準偏差） : {math.sqrt(variance) * 100:.1f}ポイント")

            informations.append([size, seed, game_num, mean_win_rate, math.sqrt(variance), epsilon, gamma, batch_size, target_update_interval])

            checkpoint = {
                "game_num": game_num,
                "seed": seed,
                "model_state_dict": agent.model.state_dict(),
                "target_model_state_dict": agent.target_model.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "learn_count": learn_count,
                "buffer": buffer,
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "learn_game_step": learn_game_step,
                "epsilon": epsilon,
                "gamma": gamma,
                "batch_size": batch_size,
                "max_learn_game_num": max_learn_game_num,
                "target_update_interval": target_update_interval,
                "eval_random_state": eval_rng.getstate(),
                "board_size": size
            }

            torch.save( # モデルの保存
                checkpoint,
                f"models/checkpoints/seed_{seed}_game_{game_num}.pth"
            )
            # torch.save( # モデルの保存(colab用)
            #     checkpoint,
            #     f"/content/drive/MyDrive/Colab Notebooks/プライベート/projects/Miniothello/models/checkpoints/seed_{seed}_game_{game_num}.pth"
            # )
    
    return informations