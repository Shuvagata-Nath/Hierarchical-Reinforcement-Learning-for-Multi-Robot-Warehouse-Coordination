import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from config import WarehouseConfig
from warehouse_env import WarehouseHRLEnv
from controllers import GreedyScheduler, CongestionAwareGreedyScheduler, RandomScheduler
from visualize import render_env_snapshot, save_gif, save_mp4


def run_policy(env, policy, episodes=10, make_gif=False, gif_path=None, mp4_path=None):
    rows = []
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=100 + ep)
        ep_reward = 0.0
        ep_delivered = 0
        ep_collisions = 0
        ep_congestion_risk = 0.0
        for step in range(env.cfg.max_steps):
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = policy.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_delivered += info.get("delivered", 0)
            ep_collisions += info.get("collisions", 0)
            ep_congestion_risk += info.get("congestion_risk", 0.0)
            if make_gif and ep == 0:
                frames.append(render_env_snapshot(env, step=step, info=info, cumulative_reward=ep_reward, policy_name="Congestion-Aware HRL"))
            if terminated or truncated:
                break
        rows.append({"reward": ep_reward, "delivered": ep_delivered, "collisions": ep_collisions, "congestion_risk": ep_congestion_risk})
    if make_gif and frames and gif_path:
        save_gif(frames, gif_path, fps=6)
    if make_gif and frames and mp4_path:
        try:
            save_mp4(frames, mp4_path, fps=6)
        except Exception as e:
            print(f"MP4 export skipped: {e}")
            print("Tip: install FFmpeg and make sure ffmpeg is available on PATH.")
    def mean(key): return float(np.mean([r[key] for r in rows]))
    def std(key): return float(np.std([r[key] for r in rows]))
    return {
        "reward_mean": mean("reward"),
        "reward_std": std("reward"),
        "delivered_mean": mean("delivered"),
        "delivered_std": std("delivered"),
        "collisions_mean": mean("collisions"),
        "collisions_std": std("collisions"),
        "congestion_risk_mean": mean("congestion_risk"),
    }, rows


def save_csv(summary, episode_rows):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/evaluation_summary_results.csv", "w", newline="") as f:
        fields = ["policy", "reward_mean", "reward_std", "delivered_mean", "delivered_std", "collisions_mean", "collisions_std", "congestion_risk_mean"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    with open("outputs/evaluation_episode_results.csv", "w", newline="") as f:
        fields = ["policy", "episode", "reward", "delivered", "collisions", "congestion_risk"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for policy_name, rows in episode_rows.items():
            for i, row in enumerate(rows):
                out = {"policy": policy_name, "episode": i}
                out.update(row)
                writer.writerow(out)
    print("Saved outputs/evaluation_summary_results.csv")
    print("Saved outputs/evaluation_episode_results.csv")


def bar_chart(summary, metric, title, path):
    names = [s["policy"] for s in summary]
    values = [s[metric] for s in summary]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def save_charts(summary):
    bar_chart(summary, "reward_mean", "Average Reward", "outputs/reward_comparison.png")
    bar_chart(summary, "delivered_mean", "Average Completed Deliveries", "outputs/delivered_comparison.png")
    bar_chart(summary, "collisions_mean", "Collision/Congestion Events", "outputs/collision_comparison.png")
    bar_chart(summary, "congestion_risk_mean", "Assignment Path-Overlap Risk", "outputs/congestion_risk_comparison.png")


def main():
    cfg = WarehouseConfig()
    env = WarehouseHRLEnv(cfg)
    print("Evaluation environment:")
    print(f"  grid={cfg.width}x{cfg.height}")
    print(f"  robots={cfg.n_robots}")
    print(f"  max_orders={cfg.max_orders}")
    print(f"  max_steps={cfg.max_steps}")
    print(f"  observation_shape={env.observation_space.shape}")
    print(f"  action_space={env.action_space}")
    policies = [("Greedy", GreedyScheduler()), ("Congestion Greedy", CongestionAwareGreedyScheduler()), ("Random", RandomScheduler(seed=1))]
    results = []
    episode_rows = {}
    for name, policy in policies:
        print(f"Evaluating {name} baseline...")
        stats, rows = run_policy(env, policy, episodes=10)
        stats["policy"] = name
        results.append(stats)
        episode_rows[name] = rows
        print(name + ":", stats)
    model_path = "outputs/hrl_warehouse_scheduler.zip"
    if os.path.exists(model_path):
        print("Evaluating trained HRL policy...")
        model = PPO.load(model_path)
        stats, rows = run_policy(env, model, episodes=10, make_gif=True, gif_path="outputs/hrl_policy_demo.gif", mp4_path="outputs/hrl_policy_demo.mp4")
        stats["policy"] = "Congestion-Aware HRL"
        results.append(stats)
        episode_rows["Congestion-Aware HRL"] = rows
        print("HRL:", stats)
    else:
        print("No trained model found. Train first.")
    save_csv(results, episode_rows)
    save_charts(results)

if __name__ == "__main__":
    main()
