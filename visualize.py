import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter
from matplotlib.patches import Patch

ROBOT_COLORS = [[0.9,0.1,0.1],[0.1,0.7,0.1],[0.2,0.2,0.9],[0.8,0.2,0.8],[0.1,0.8,0.8]]

def render_env_snapshot(env, step=0, info=None, cumulative_reward=0.0, policy_name="HRL"):
    h, w = env.cfg.height, env.cfg.width
    canvas = np.ones((h, w, 3), dtype=float)
    for x, y in env.shelves: canvas[y, x] = [0.2,0.2,0.2]
    for x, y in env.delivery_cells: canvas[y, x] = [0.6,0.8,1.0]
    order_labels = []
    for order in env.orders:
        if order is not None:
            x, y = order["pickup"]
            canvas[y, x] = [1.0,0.65,0.1] if order["assigned"] else [1.0,0.9,0.3]
            order_labels.append((x, y, "P"))
    robot_labels = []
    for i, r in enumerate(env.robots):
        x, y = r.pos
        canvas[y, x] = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        robot_labels.append((i+1, x, y, r.phase))
    return {"canvas": canvas, "step": step, "policy_name": policy_name, "robot_labels": robot_labels, "order_labels": order_labels, "active_orders": sum(1 for o in env.orders if o is not None), "info": info or {}, "cumulative_reward": cumulative_reward}

def _draw(ax, snapshot):
    ax.clear()
    frame = snapshot["canvas"]
    ax.imshow(frame, interpolation="nearest")
    h, w, _ = frame.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(which="minor", bottom=False, left=False)
    for y in range(h):
        for x in range(w):
            if np.allclose(frame[y, x], [0.6,0.8,1.0]):
                ax.text(x, y, "D", ha="center", va="center", fontsize=8, weight="bold")
    for x, y, label in snapshot["order_labels"]:
        ax.text(x, y, label, ha="center", va="center", fontsize=8, weight="bold")
    for robot_id, x, y, phase in snapshot["robot_labels"]:
        ax.text(x, y, f"R{robot_id}", ha="center", va="center", fontsize=8, weight="bold", color="white")
    info = snapshot["info"]
    status = f"{snapshot['policy_name']}\nStep: {snapshot['step']}\nActive orders: {snapshot['active_orders']}\nDelivered: {info.get('delivered',0)}\nCollisions: {info.get('collisions',0)}\nPath-overlap risk: {info.get('congestion_risk',0):.1f}\nReward: {snapshot['cumulative_reward']:.1f}"
    ax.text(1.02, 0.5, status, transform=ax.transAxes, fontsize=9, va="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    legend_items = [Patch(facecolor=(0.2,0.2,0.2), label="Shelf / Obstacle"), Patch(facecolor=(0.6,0.8,1.0), label="Delivery Station"), Patch(facecolor=(1.0,0.9,0.3), label="Waiting Order"), Patch(facecolor=(1.0,0.65,0.1), label="Assigned Order")]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    ax.set_title("Congestion-Aware HRL Warehouse Fulfillment", fontsize=11, weight="bold")

def save_gif(frames, path, fps=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, path, dpi=120):
        for frame in frames:
            _draw(ax, frame); writer.grab_frame()
    plt.close(fig); print(f"Saved {path}")

def save_mp4(frames, path, fps=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, path, dpi=120):
        for frame in frames:
            _draw(ax, frame); writer.grab_frame()
    plt.close(fig); print(f"Saved {path}")
