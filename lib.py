import numpy as np
import pandas as pd

Configuration = np.ndarray

class Grid(object):
    def __init__(self, R: int, C: int, relevance: np.ndarray, wormholes: list[tuple[int, int]], snake_lengths: list[int]):
        self.R = R
        self.C = C
        self.wormholes = wormholes
        self.wh_mask = np.zeros((R, C))
        for i,j in self.wormholes:
            self.wh_mask[i,j] = 1
        self.relevance = relevance
        self.snake_lengths = snake_lengths
        self.wh_neighs = [(i,j) for ip, jp in wormholes for i, j in self.neighbors_simple(ip, jp)]
        self.configuration = None
        self.grid_mask = np.zeros((R, C))
        self.reward = None
        self.accepted_moves = 0
        self.attempted_moves = 0

    def pbc(self, i: int, j: int) -> tuple[int, int]:
        i_inbounds = i % self.R
        j_inbounds = j % self.C
        return i_inbounds, j_inbounds

    def neighbors_simple(self, i: int, j: int) -> list[tuple[int, int]]:
        neighs = [
            self.pbc(i, j+1), # N
            self.pbc(i, j-1), # S
            self.pbc(i-1, j), # W
            self.pbc(i+1, j)  # E
        ]
        return neighs

    def neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        return (
            self.neighbors_simple(i, j)
            if (i,j) not in self.wormholes
            else self.wh_neighs
        )

#    def generate_random_configuration(self, snake_lengths: list[int]) -> Configuration:
#        conf = np.full((R, C), "")
#        for snake_id_m1, snake_length in enumerate(snake_lengths):
#            snake_id = str(snake_id_m1 + 1)
#            # Place snake's head.
#            conf
#            ir = np.random.randint(0, self.R)
#            jr = np.random.randint(0, self.C)
#            conf[ir, jr] += f"_{snake_id}"
#            curr_ij = ir, jr
#            for _ in range(snake_length-1):
#                allowed_ij = [i,j for i,j in self.neighbors(*curr_ij) if conf[i,j] == "" or i,j in self.wormholes]
#                assert len(allowed_ij) > 0, "Snake got stuck!"
#                ir, jr = np.random.choice(allowed_ij)
#                conf[ir, jr] += f"_{snake_id}"
#                curr_ij = ir, jr
#        return conf

    def free_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        return [
            (i_n, j_n) for i_n, j_n in self.neighbors(i, j)
            if (1-self.grid_mask)[i_n, j_n] or self.wh_mask[i_n, j_n]
        ]

    def generate_random_configuration(self, snake_lengths: list[int]) -> Configuration:
        conf = []
        for snake_length in snake_lengths:
            # Place the head of the snake.
            allowed_ij = list(zip(*np.where(self.grid_mask + self.wh_mask == 0)))
            idx_r = np.random.randint(0, len(allowed_ij))
            ij_r = allowed_ij[idx_r]
            conf.append([ij_r])
            self.grid_mask[ij_r[0], ij_r[1]] = 1
            for k in range(snake_length-2):
                # We allow all neighbors provided they are free OR wormholes.
                allowed_ij = self.free_neighbors(*ij_r)
                assert len(allowed_ij) > 0, "Snake got stuck!"
                idx_r = np.random.randint(0, len(allowed_ij))
                ij_r = allowed_ij[idx_r]
                conf[-1].append(ij_r)
                self.grid_mask[ij_r[0], ij_r[1]] = 1
            # Snake cannot end at a wormhole cell: we allow all neighbors provided they are free
            # AND NOT wormholes.
            allowed_ij = [
                (i,j) for i,j in self.neighbors(*ij_r) if not (self.grid_mask[i,j] or self.wh_mask[i,j])
            ]
            assert len(allowed_ij) > 0, "Snake got stuck!"
            idx_r = np.random.randint(0, len(allowed_ij))
            ij_r = allowed_ij[idx_r]
            conf[-1].append(ij_r)
            self.grid_mask[ij_r[0], ij_r[1]] = 1
        return conf

    def reward_function(self, configuration: Configuration) -> int:
        return self.relevance[self.configuration != ""].sum()

    def initialize(self):
        self.configuration = self.generate_random_configuration(self.snake_lengths)
        self.reward = self.reward_function(self.configuration)
        return

    def propose_move(self) -> tuple[Configuration, int]:
        # Choose one snake to move.
        snake_id = np.random.randint(len(self.snake_lengths))
        ij_head = self.configuration[snake_id][0]
        allowed_ij = self.free_neighbors(*ij_head)
        if len(allowed_ij) == 0:
            # TODO move from tail
            proposed_configuration = self.configuration
            delta_reward = 0
        else:
            idx_h = np.random.randint(0, len(allowed_ij))
            ij_h = allowed_ij[idx_h]
            i_h, j_h = ij_h
            # Move snake forward to ij_r (delete last segment of tail).
            i_t, j_t = self.configuration[snake_id][-1]
            proposed_configuration = [ij_h] + self.configuration[snake_id][:-1]
            delta_reward = self.relevance[i_h, j_h] - self.relevance[i_t, j_t]
        return proposed_configuration, delta_reward

    def try_step(self, beta: float):
        proposed_conf, delta_reward = self.propose_move()
        self.attempted_moves += 1
        r = np.random.rand()
        if r < np.exp(-beta * delta_reward):
            self.configuration = proposed_conf
            self.reward += delta_reward
            self.accepted_moves += 1

    def acceptance_rate(self) -> float:
        return self.accepted_moves / self.attempted_moves

#class Annealer(object):
#    def __init__(self, T0):
#        self.T0 = T0
#        self.T = T0
#
#    def cost(self, x) -> float:
#        ...
#
#    def step(self):
#        ...
#
#    def try_step(self):
#        r = np.random.rand()
#
#    def
