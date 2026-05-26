# Congestion-Aware HRL Warehouse Fulfillment

This project is a grid-based multi-robot warehouse fulfillment simulator using hierarchical reinforcement learning. The high-level RL scheduler assigns robots to customer orders, while A* path planning handles low-level robot navigation around shelves and obstacles.

The project compares the learned HRL scheduler against greedy, congestion-aware greedy, and random baseline policies using delivery throughput, reward, and collision/congestion metrics.

## Project Structure

```text
config.py                  Environment size, reward values, and congestion settings
warehouse_env.py           Gymnasium warehouse environment
controllers.py             Greedy, congestion-aware greedy, and random schedulers
astar.py                   A* path planner for robot navigation
train_hrl_scheduler.py     PPO training script for the HRL scheduler
evaluate.py                Evaluation script for baselines and HRL
demo_heuristic.py          Greedy rollout demo
visualize.py               GIF/MP4 visualization utilities
requirements.txt           Required Python packages
outputs/                   Generated model, charts, GIFs, and CSV files
```

## How It Works

The system is hierarchical:

```text
HRL scheduler
    chooses robot-order assignments
        ↓
A* path planner
    computes obstacle-free paths
        ↓
Warehouse simulator
    updates robot movement, deliveries, rewards, and collisions
```


## How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the HRL scheduler:

```bash
py train_hrl_scheduler.py --timesteps 1000000
```

Evaluate all policies:

```bash
py evaluate.py
```

Run only the greedy visual demo:

```bash
py demo_heuristic.py
```

To retrain from scratch:

```bash
rmdir /s /q outputs
py train_hrl_scheduler.py --timesteps 1000000
py evaluate.py
```

## Outputs

The evaluation script generates the following files:

```text
outputs/hrl_warehouse_scheduler.zip       Trained HRL model
outputs/hrl_policy_demo.gif               HRL rollout visualization
outputs/evaluation_summary_results.csv    Summary results for each policy
outputs/evaluation_episode_results.csv    Episode-level results
outputs/reward_comparison.png             Reward comparison chart
outputs/delivered_comparison.png          Delivery comparison chart
outputs/collision_comparison.png          Collision/congestion comparison chart
outputs/congestion_risk_comparison.png    Path-overlap risk comparison chart
```



## Results

The final congestion-aware HRL run produced the following result:

| Policy | Average Reward | Average Deliveries | Average Collision/Congestion Events |
|---|---:|---:|---:|
| Greedy | 23.34 | 2.0 | 832.0 |
| Congestion Greedy | 48.11 | 2.4 | 822.7 |
| Random | -424.24 | 3.6 | 778.8 |
| Congestion-Aware HRL | 246.04 | 5.9 | 746.1 |

The earlier HRL setup could get close to greedy delivery performance, but collision reduction was limited because the scheduler only received collision penalties after conflicts happened. In the final version, A* is also used to estimate path-overlap risk before assignment. This gave the HRL policy a more useful signal for avoiding crowded routes.

As a result, the congestion-aware HRL scheduler achieved the best overall performance in this experiment. It completed more deliveries than both greedy baselines while also reducing collision/congestion events.

## Findings

The main finding is that HRL became useful when the scheduler received congestion-aware information. A basic greedy scheduler only selects short-distance assignments, while the HRL policy learned from delivery rewards, collision penalties, and path-overlap risk.

This suggests that high-level learning is more effective when the environment provides meaningful decision features instead of relying only on delayed collision penalties.

