# 条件を決めてtrainとevaluateを呼ぶ

from train import agent_continuous_learn, agent_resume_learn
from ai.agent import Agent
import time
import torch
import random
import csv
from pathlib import Path

def learn_accumulation_model_experiment(size,seed,writer,max_learn_game_num,learn_game_step): # 学習AI累積ゲーム数におけるモデルの評価

    epsilon = 0.1
    gamma = 0.99
    batch_size = 32
    target_update_interval = 5

    torch.manual_seed(seed)
    random.seed(seed)

    print(f"\nモデル{seed+1}\nseed : {seed}")

    evaluation_agent = Agent(size)

    informations = agent_continuous_learn(
        size,
        max_learn_game_num,
        epsilon,
        gamma,
        batch_size,
        target_update_interval,
        learn_game_step,
        evaluation_agent,
        seed
    )

    for row in informations:
        writer.writerow(row)

def multi_model_experiment(size,model_num,max_learn_game_num,learn_game_step): #  複数の独立モデルを管理

    print(f"盤面{size}×{size}")

    with open("logs/mini_othello_model_learn_log.csv",mode="w",newline="") as f:
    # with open("/content/drive/MyDrive/Colab Notebooks/プライベート/projects/Miniothello/logs/mini_othello_model_learn_log.csv",mode="w",newline="") as f: # colab用
        writer = csv.writer(f)
            
        writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

        for i in range(model_num):
            learn_accumulation_model_experiment(size,i,writer,max_learn_game_num,learn_game_step)

def resume_model_experiment(
    size,
    checkpoint_path,
    writer,
    max_learn_game_num=None,
    learn_game_step=None
): # 学習再開用(1つのモデル)
    informations = agent_resume_learn(
        size,
        checkpoint_path,
        max_learn_game_num,
        learn_game_step
    )

    for row in informations:
        writer.writerow(row)

def multi_resume_experiment(
    size,
    resume_game_num,
    max_learn_game_num=None,
    learn_game_step=None
): # 学習再開用(複数モデル)

    with open("logs/mini_othello_model_learn_log_resume.csv",mode="w",newline="") as f:
    # with open("/content/drive/MyDrive/Colab Notebooks/プライベート/projects/Miniothello/logs/mini_othello_model_learn_log_resume.csv",mode="w",newline="") as f: # colab用
        writer = csv.writer(f)
        writer.writerow(["size","seed","cumulative_game_num","mean_win_rate","eval_std","epsilon","gamma","batch_size","target_update_interval"])

        checkpoint_dir = Path("models/checkpoints")
        checkpoint_paths = list(checkpoint_dir.glob(f"seed_*_game_{resume_game_num}.pth"))
        # checkpoint_paths = checkpoint_dir.glob("seed_*_game_20000.pth") # 20000ゲーム地点から全モデル再開

        for checkpoint_path in checkpoint_paths:
            resume_model_experiment(
                size,
                checkpoint_path,
                writer,
                max_learn_game_num,
                learn_game_step
            )


if __name__ == "__main__":
    start = time.perf_counter()

    multi_model_experiment(8,5,40000,4000) # (盤面サイズ,独立モデル数,最終モデルゲーム数,チェックポイントタイミング)
    # multi_resume_experiment(8,40000,80000,2000) # (盤面サイズ,再開開始ゲーム数,最大ゲーム数（任意）,チェックポイントタイミング(任意))

    end = time.perf_counter()

    experiment_time = end - start
    hour = int(experiment_time // 3600)
    minute = int((experiment_time % 3600) // 60)
    second = int(experiment_time % 60)

    print(f"実行時間: {hour}時間{minute}分{second}秒")