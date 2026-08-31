# 様々なパラメーターを変化させながら実験
import csv, torch, random, time
from train import agent_continuous_learn
from ai.agent import Agent

def run_experiment(size,max_learn_game_num,learn_step_game_num,parameter_name,parameter_value,writer,model_name=None): # 共通で呼び出す関数
    seed = 0
    epsilon = 0.1
    gamma = 0.9
    batch_size = 32
    target_update_interval = 5

    match(parameter_name):
        case "seed":
            seed = parameter_value
        case "epsilon":
            epsilon = parameter_value
        case "gamma":
            gamma = parameter_value
        case "batch_size":
            batch_size = parameter_value
        case "target_update_interval":
            target_update_interval = parameter_value
        case _:
            print("error")
            return

    torch.manual_seed(seed)
    random.seed(seed)
    print(f"モデル{seed+1}")
    evaluation_agent = Agent(size)
    informations = agent_continuous_learn(
        size,
        max_learn_game_num,
        epsilon,
        gamma,
        batch_size,
        target_update_interval,
        learn_step_game_num,
        evaluation_agent,
        seed,
        model_name
    )
        
    for row in informations:
        writer.writerow(row)
    print()

def run_experiment_model_num(size,max_learn_game_num,learn_step_game_num,model_num,writer=None): # seedごとのモデル比較
    print("seed比較")

    if writer is None:
        print(f"盤面 {size}×{size}")

        with open("logs/othello_model_learn_model_num_log.csv",mode="a",newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

    for i in range(model_num):
        print(f"seed: {i}")
        run_experiment(size,max_learn_game_num,learn_step_game_num,"seed",i,writer,f"models/checkpoints/seeds/seed{i}.pth")

    print()

def run_experiment_epsilon(size,max_learn_game_num,learn_step_game_num,writer=None): # epsilonごとのモデル比較
    print("epsilon比較")

    epsilons = [0.05, 0.1, 0.2, 0.3]

    if writer is None:
        print(f"盤面 {size}×{size}")

        with open("logs/othello_model_learn_epsilon_log.csv",mode="a",newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

    for epsilon in epsilons:
        print(f"epsilon: {epsilon}")
    
        run_experiment(size,max_learn_game_num,learn_step_game_num,"epsilon",epsilon,writer,f"models/checkpoints/epsilons/eps{epsilon}.pth")

    print()

def run_experiment_gamma(size,max_learn_game_num,learn_step_game_num,writer=None): # gammaごとのモデル比較
    print("gamma比較")

    gammas = [0.90, 0.95, 0.99, 0.995]

    if writer is None:

        print(f"盤面 {size}×{size}")

        with open("logs/othello_model_learn_gamma_log.csv",mode="a",newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

    for gamma in gammas:
        print(f"gamma: {gamma}")
        
        run_experiment(size,max_learn_game_num,learn_step_game_num,"gamma",gamma,writer,f"models/checkpoints/gammas/gamma{gamma}.pth")  

    print()

def run_experiment_batch_size(size,max_learn_game_num,learn_step_game_num,writer=None): # batch_sizeごとのモデル比較
    print("batch_size比較")

    batch_sizes = [16, 32, 64, 128]

    if writer is None:

        print(f"盤面 {size}×{size}")

        with open("logs/othello_model_learn_batch_size_log.csv",mode="a",newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

    for batch_size in batch_sizes:
        print(f"batch_size: {batch_size}")

        run_experiment(size,max_learn_game_num,learn_step_game_num,"batch_size",batch_size,writer,f"models/checkpoints/batch_sizes/batch_size{batch_size}.pth")

    print()

def run_experiment_target_update_interval(size,max_learn_game_num,learn_step_game_num,writer=None): # target_update_intervalごとのモデル比較
    print("target_update_interval比較")

    target_update_intervals = [1, 3, 5, 7, 10]

    if writer is None:

        print(f"盤面 {size}×{size}")

        with open("logs/othello_model_learn_target_update_interval_log.csv",mode="a",newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

    for target_update_interval in target_update_intervals:
        print(f"target_update_interval: {target_update_interval}")

        run_experiment(size,max_learn_game_num,learn_step_game_num,"target_update_interval",target_update_interval,writer,f"models/checkpoints/target_update_intervals/target_update_interval{target_update_interval}.pth")

    print()

if __name__ == "__main__":
    start = time.perf_counter()

    size = 8
    max_learn_game_num = 1
    learn_step_game_num = 1
    model_num = 1

    print(f"盤面 {size}×{size}")

    with open("logs/othello_model_learn_log.csv",mode="a",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

        run_experiment_model_num(
            size,
            max_learn_game_num,
            learn_step_game_num,
            model_num,
            writer
        )

        run_experiment_epsilon(
            size,
            max_learn_game_num,
            learn_step_game_num,
            writer
        )

        run_experiment_gamma(
            size,
            max_learn_game_num,
            learn_step_game_num,
            writer
        )

        run_experiment_batch_size(
            size,
            max_learn_game_num,
            learn_step_game_num,
            writer
        )

        run_experiment_target_update_interval(
            size,
            max_learn_game_num,
            learn_step_game_num,
            writer
        )

    end = time.perf_counter()

    experiment_time = end - start
    hour = int(experiment_time // 3600)
    minute = int((experiment_time % 3600) // 60)
    second = int(experiment_time % 60)

    print(f"実行時間: {hour}時間{minute}分{second}秒")