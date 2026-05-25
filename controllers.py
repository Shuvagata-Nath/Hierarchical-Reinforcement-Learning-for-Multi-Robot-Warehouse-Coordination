import numpy as np
from astar import astar, manhattan


class Robot:
    def __init__(self, robot_id, start_pos):
        self.robot_id = robot_id
        self.start_pos = start_pos
        self.pos = start_pos
        self.phase = "idle"
        self.order_id = None
        self.pickup = None
        self.delivery = None
        self.path = []
        self.carrying = False

    def reset(self):
        self.pos = self.start_pos
        self.phase = "idle"
        self.order_id = None
        self.pickup = None
        self.delivery = None
        self.path = []
        self.carrying = False

    def assign(self, order_id, pickup, delivery, width, height, obstacles):
        if self.phase != "idle":
            return False

        to_pickup = astar(self.pos, pickup, width, height, obstacles)
        if to_pickup is None:
            return False

        to_delivery = astar(pickup, delivery, width, height, obstacles)
        if to_delivery is None:
            return False

        self.phase = "to_pickup"
        self.order_id = order_id
        self.pickup = pickup
        self.delivery = delivery
        self.path = list(to_pickup)
        self.carrying = False
        return True

    def desired_next_cell(self):
        if self.path:
            return self.path[0]
        return self.pos

    def commit_move(self, final_pos, width, height, obstacles):
        self.pos = final_pos

        if self.path and self.path[0] == final_pos:
            self.path.pop(0)

        if self.phase == "to_pickup" and self.pos == self.pickup and not self.path:
            path_to_delivery = astar(self.pos, self.delivery, width, height, obstacles)
            if path_to_delivery is None:
                self.phase = "idle"
                return None
            self.phase = "to_delivery"
            self.carrying = True
            self.path = list(path_to_delivery)
            return None

        if self.phase == "to_delivery" and self.pos == self.delivery and not self.path:
            oid = self.order_id
            self.phase = "idle"
            self.order_id = None
            self.pickup = None
            self.delivery = None
            self.carrying = False
            self.path = []
            return oid

        return None


class GreedyScheduler:
    def act(self, env):
        best_action = env.noop_action
        best_score = float("inf")

        for r_idx, r in enumerate(env.robots):
            if r.phase != "idle":
                continue
            for o_idx, order in enumerate(env.orders):
                if order is None or order["assigned"]:
                    continue
                d = manhattan(r.pos, order["pickup"])
                if d < best_score:
                    best_score = d
                    best_action = env.encode_action(r_idx, o_idx)

        return best_action


class CongestionAwareGreedyScheduler:
    def act(self, env):
        best_action = env.noop_action
        best_score = float("inf")

        for r_idx, r in enumerate(env.robots):
            if r.phase != "idle":
                continue
            for o_idx, order in enumerate(env.orders):
                if order is None or order["assigned"]:
                    continue
                dist = manhattan(r.pos, order["pickup"])
                risk = env.estimate_assignment_congestion(r_idx, o_idx)
                score = dist + 3.0 * risk
                if score < best_score:
                    best_score = score
                    best_action = env.encode_action(r_idx, o_idx)

        return best_action


class RandomScheduler:
    def __init__(self, seed=1):
        self.rng = np.random.default_rng(seed)

    def act(self, env):
        return int(self.rng.integers(0, env.action_space.n))
