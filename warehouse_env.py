import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import WarehouseConfig
from controllers import Robot
from astar import astar, manhattan


class WarehouseHRLEnv(gym.Env):
    """
    Congestion-aware hierarchical warehouse environment.

    HRL decides high-level robot-order assignments.
    A* handles low-level movement.

    Extension:
    When an assignment is made, the environment estimates planned path overlap with
    other active robot paths and penalizes high-overlap assignments. This encourages
    HRL to learn safer task allocation rather than only shortest assignment.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.cfg = config or WarehouseConfig()
        self.render_mode = render_mode

        self.noop_action = self.cfg.n_robots * self.cfg.max_orders
        self.action_space = spaces.Discrete(self.noop_action + 1)

        # Robot features: x, y, idle, carrying
        # Order features: active, assigned, pickup_x, pickup_y, delivery_x, delivery_y,
        # nearest_distance, congestion_risk
        obs_dim = self.cfg.n_robots * 4 + self.cfg.max_orders * 8
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self.shelves = set()
        self.pickup_cells = []
        self.delivery_cells = []
        self.robots = []
        self.orders = []

    def encode_action(self, robot_idx, order_idx):
        return robot_idx * self.cfg.max_orders + order_idx

    def decode_action(self, action):
        action = int(action)
        if action == self.noop_action:
            return None, None
        robot_idx = action // self.cfg.max_orders
        order_idx = action % self.cfg.max_orders
        return robot_idx, order_idx

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.shelves = self._make_shelves()
        self.pickup_cells = self._make_pickup_cells()
        self.delivery_cells = self._make_delivery_cells()

        starts = [(0, 0), (0, self.cfg.height - 1), (1, self.cfg.height // 2), (2, 0), (2, self.cfg.height - 1)]
        self.robots = [Robot(i, starts[i % len(starts)]) for i in range(self.cfg.n_robots)]

        self.orders = [None for _ in range(self.cfg.max_orders)]
        for _ in range(min(3, self.cfg.max_orders)):
            self._spawn_order()

        return self._get_obs(), {}

    def _make_shelves(self):
        shelves = set()
        shelf_columns = [4, 5, 8, 9]
        aisle_rows = [3, 6]

        for x in shelf_columns:
            for y in range(1, self.cfg.height - 1):
                if y in aisle_rows:
                    continue
                shelves.add((x, y))

        return shelves

    def _make_delivery_cells(self):
        return [(self.cfg.width - 1, 2), (self.cfg.width - 1, self.cfg.height - 3)]

    def _make_pickup_cells(self):
        candidates = []
        for sx, sy in self.shelves:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                c = (sx + dx, sy + dy)
                if self._is_free_static(c):
                    candidates.append(c)
        return sorted(list(set(candidates)))

    def _is_free_static(self, cell):
        x, y = cell
        if not (0 <= x < self.cfg.width and 0 <= y < self.cfg.height):
            return False
        if cell in self.shelves:
            return False
        return True

    def _spawn_order(self):
        empty = [i for i, o in enumerate(self.orders) if o is None]
        if not empty:
            return False

        pickup = self.pickup_cells[int(self.rng.integers(0, len(self.pickup_cells)))]
        delivery = self.delivery_cells[int(self.rng.integers(0, len(self.delivery_cells)))]

        idx = empty[0]
        self.orders[idx] = {"pickup": pickup, "delivery": delivery, "assigned": False, "delivered": False}
        return True

    def _planned_assignment_path(self, robot_idx, order_idx):
        robot = self.robots[robot_idx]
        order = self.orders[order_idx]
        if order is None:
            return None

        path1 = astar(robot.pos, order["pickup"], self.cfg.width, self.cfg.height, self.shelves)
        if path1 is None:
            return None
        path2 = astar(order["pickup"], order["delivery"], self.cfg.width, self.cfg.height, self.shelves)
        if path2 is None:
            return None
        return list(path1) + list(path2)

    def estimate_assignment_congestion(self, robot_idx, order_idx):
        candidate = self._planned_assignment_path(robot_idx, order_idx)
        if not candidate:
            return 0.0

        candidate_set = set(candidate)
        overlap = 0

        for j, other in enumerate(self.robots):
            if j == robot_idx:
                continue
            if other.path:
                overlap += len(candidate_set.intersection(set(other.path)))

        order = self.orders[order_idx]
        local = 0
        if order is not None:
            px, py = order["pickup"]
            for other in self.robots:
                if manhattan(other.pos, (px, py)) <= 2:
                    local += 1

        return float(overlap + 0.5 * local)

    def _get_obs(self):
        obs = []

        for r in self.robots:
            obs.extend([
                r.pos[0] / max(1, self.cfg.width - 1),
                r.pos[1] / max(1, self.cfg.height - 1),
                1.0 if r.phase == "idle" else 0.0,
                1.0 if r.carrying else 0.0,
            ])

        max_dist = self.cfg.width + self.cfg.height
        max_risk_norm = 20.0

        for o_idx, order in enumerate(self.orders):
            if order is None:
                obs.extend([0.0] * 8)
                continue

            nearest = min(manhattan(r.pos, order["pickup"]) for r in self.robots)
            risks = []
            for r_idx, r in enumerate(self.robots):
                if r.phase == "idle":
                    risks.append(self.estimate_assignment_congestion(r_idx, o_idx))
            risk = min(risks) if risks else max_risk_norm

            obs.extend([
                1.0,
                1.0 if order["assigned"] else 0.0,
                order["pickup"][0] / max(1, self.cfg.width - 1),
                order["pickup"][1] / max(1, self.cfg.height - 1),
                order["delivery"][0] / max(1, self.cfg.width - 1),
                order["delivery"][1] / max(1, self.cfg.height - 1),
                min(nearest / max_dist, 1.0),
                min(risk / max_risk_norm, 1.0),
            ])

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.t += 1
        reward = self.cfg.step_penalty

        info = {"delivered": 0, "collisions": 0, "invalid_action": False, "active_orders": 0, "congestion_risk": 0.0}

        robot_idx, order_idx = self.decode_action(action)

        if robot_idx is not None:
            risk_before = self.estimate_assignment_congestion(robot_idx, order_idx) if self._action_indices_valid(robot_idx, order_idx) else 0.0
            valid = self._try_assignment(robot_idx, order_idx)

            if valid:
                reward += self.cfg.assignment_reward
                reward += self.cfg.path_overlap_penalty * risk_before
                reward += self.cfg.congestion_radius_penalty * risk_before
                info["congestion_risk"] = risk_before
            else:
                reward += self.cfg.invalid_action_penalty
                info["invalid_action"] = True
        else:
            active_unassigned = sum(1 for o in self.orders if o is not None and not o["assigned"])
            idle_robots = sum(1 for r in self.robots if r.phase == "idle")
            if active_unassigned > 0 and idle_robots > 0:
                reward += self.cfg.invalid_action_penalty * 0.25

        delivered_ids, collisions = self._move_robots()
        info["collisions"] = collisions
        reward += self.cfg.collision_penalty * collisions

        for oid in delivered_ids:
            order = self.orders[oid]
            if order is None:
                continue
            info["delivered"] += 1
            reward += self.cfg.delivery_reward
            order["delivered"] = True

        if self.rng.random() < self.cfg.order_spawn_prob:
            self._spawn_order()

        for i, order in enumerate(self.orders):
            if order is not None and order["delivered"]:
                self.orders[i] = None

        active_orders = sum(1 for o in self.orders if o is not None)
        info["active_orders"] = active_orders
        reward += self.cfg.backlog_penalty * active_orders

        terminated = False
        truncated = self.t >= self.cfg.max_steps
        return self._get_obs(), float(reward), terminated, truncated, info

    def _action_indices_valid(self, robot_idx, order_idx):
        return 0 <= robot_idx < len(self.robots) and 0 <= order_idx < len(self.orders) and self.orders[order_idx] is not None

    def _try_assignment(self, robot_idx, order_idx):
        if not self._action_indices_valid(robot_idx, order_idx):
            return False

        robot = self.robots[robot_idx]
        order = self.orders[order_idx]

        if robot.phase != "idle":
            return False
        if order["assigned"]:
            return False

        ok = robot.assign(order_id=order_idx, pickup=order["pickup"], delivery=order["delivery"], width=self.cfg.width, height=self.cfg.height, obstacles=self.shelves)
        if ok:
            order["assigned"] = True
        return ok

    def _move_robots(self):
        desired = [r.desired_next_cell() for r in self.robots]
        current = [r.pos for r in self.robots]

        counts = {}
        for cell in desired:
            counts[cell] = counts.get(cell, 0) + 1

        final_positions = []
        collisions = 0

        for i, r in enumerate(self.robots):
            nxt = desired[i]
            blocked = False

            if counts[nxt] > 1:
                blocked = True

            for j, other_pos in enumerate(current):
                if i == j:
                    continue
                if nxt == other_pos and desired[j] == r.pos:
                    blocked = True

            if blocked and nxt != r.pos:
                collisions += 1
                final_positions.append(r.pos)
            else:
                final_positions.append(nxt)

        delivered_ids = []
        for r, pos in zip(self.robots, final_positions):
            delivered = r.commit_move(pos, self.cfg.width, self.cfg.height, self.shelves)
            if delivered is not None:
                delivered_ids.append(delivered)
        return delivered_ids, collisions

    def render_grid(self):
        grid = [["." for _ in range(self.cfg.width)] for _ in range(self.cfg.height)]
        for x, y in self.shelves:
            grid[y][x] = "#"
        for x, y in self.delivery_cells:
            grid[y][x] = "D"
        for order in self.orders:
            if order is not None:
                x, y = order["pickup"]
                grid[y][x] = "P"
        for i, r in enumerate(self.robots):
            x, y = r.pos
            grid[y][x] = str(i)
        return "\n".join("".join(row) for row in grid)

    def close(self):
        pass
