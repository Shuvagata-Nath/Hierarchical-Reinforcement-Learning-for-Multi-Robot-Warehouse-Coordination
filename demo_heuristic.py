from config import WarehouseConfig
from warehouse_env import WarehouseHRLEnv
from controllers import GreedyScheduler
from visualize import render_env_snapshot, save_gif


def main():
    cfg = WarehouseConfig(max_steps=180)
    env = WarehouseHRLEnv(cfg)
    scheduler = GreedyScheduler()
    obs, _ = env.reset(seed=cfg.seed)
    frames = []
    total_reward = 0.0
    total_delivered = 0
    for step in range(cfg.max_steps):
        action = scheduler.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_delivered += info.get("delivered", 0)
        frames.append(render_env_snapshot(env, step=step, info=info, cumulative_reward=total_reward, policy_name="Greedy Demo"))
        if step % 25 == 0:
            print(f"step={step} reward={reward:.2f} delivered={total_delivered} active_orders={info.get('active_orders')}")
            print(env.render_grid()); print()
        if terminated or truncated:
            break
    save_gif(frames, "outputs/heuristic_demo.gif", fps=6)
    print("Saved outputs/heuristic_demo.gif")

if __name__ == "__main__":
    main()
