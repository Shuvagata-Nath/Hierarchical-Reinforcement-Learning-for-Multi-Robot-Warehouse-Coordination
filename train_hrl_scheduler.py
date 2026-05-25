import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from config import WarehouseConfig
from warehouse_env import WarehouseHRLEnv


MODEL_PATH = "outputs/hrl_warehouse_scheduler.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    cfg = WarehouseConfig(seed=args.seed)
    env_raw = WarehouseHRLEnv(cfg)

    print("Training environment:")
    print(f"  grid={cfg.width}x{cfg.height}")
    print(f"  robots={cfg.n_robots}")
    print(f"  max_orders={cfg.max_orders}")
    print(f"  max_steps={cfg.max_steps}")
    print(f"  observation_shape={env_raw.observation_space.shape}")
    print(f"  action_space={env_raw.action_space}")

    check_env(env_raw, warn=True)

    env = Monitor(WarehouseHRLEnv(cfg))
    eval_env = Monitor(WarehouseHRLEnv(cfg))

    if args.resume and os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,
            clip_range=0.2,
            seed=args.seed,
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="outputs/best_model",
        log_path="outputs/eval_logs",
        eval_freq=50_000,
        deterministic=True,
        n_eval_episodes=5,
    )

    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="outputs/checkpoints", name_prefix="hrl_congestion_scheduler")

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume, callback=[eval_callback, checkpoint_callback])
    model.save("outputs/hrl_warehouse_scheduler")
    print("Saved outputs/hrl_warehouse_scheduler.zip")
    print("Best evaluation model also saved at outputs/best_model/best_model.zip")


if __name__ == "__main__":
    main()
