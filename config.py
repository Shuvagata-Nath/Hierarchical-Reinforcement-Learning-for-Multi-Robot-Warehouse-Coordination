from dataclasses import dataclass


@dataclass
class WarehouseConfig:
    width: int = 14
    height: int = 10
    n_robots: int = 3
    n_shelves: int = 12
    max_orders: int = 5
    max_steps: int = 300
    order_spawn_prob: float = 0.25
    seed: int = 7

    # Reward values from the best balanced setup
    collision_penalty: float = -0.08
    step_penalty: float = -0.005
    assignment_reward: float = 1.0
    delivery_reward: float = 60.0
    invalid_action_penalty: float = -2.0
    backlog_penalty: float = -0.02

    # Congestion-aware extension
    # Penalizes assignments whose planned route overlaps with other active robot paths.
    path_overlap_penalty: float = -0.30
    congestion_radius_penalty: float = -0.05
