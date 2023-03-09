import sys

import numpy as np
import pandas as pd

Configuration = np.ndarray

def read_data(file):
    ''' this function returns two dataframe from the .txt input file'''

    cnt = 0

    with open(file, "r") as f:
        for i, line in enumerate(f):
            line = line.rstrip(' \n')
            if i == 0:
                n_cols, n_rows, n_snakes = list(map(int,line.split(' ')))
                matrix = np.zeros([n_rows, n_cols])
                relevance_matrix = np.zeros([n_rows, n_cols])

            elif i == 1:
                snake_lengths = list(map(int,line.split(' ')))

            else:
                l = line.split(' ')
                matrix[cnt,:] = [0.01 if l[i] == '*' else l[i] for i in range(len(l))]
                relevance_matrix[cnt,:] = [0 if l[i] == '*' else l[i] for i in range(len(l))]

                cnt += 1

    wormholes = tuple(zip(*np.where(matrix==0.01)))
    relevance_matrix = relevance_matrix.astype('int64')

    assert n_snakes == len(snake_lengths)

    return n_cols, n_rows, n_snakes, snake_lengths, matrix, relevance_matrix, wormholes

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
        self.trivial_steps = 0
        self.n_sweeps = 0
        self.last_delta_reward = None
        self.best_reward = -np.inf
        self.best_configuration = None
        self.reward_history = []
        self.attempted_delta_reward_history = []

    def pbc(self, i: int, j: int) -> tuple[int, int]:
        i_inbounds = i % self.R
        j_inbounds = j % self.C
        return i_inbounds, j_inbounds

    def neighbors_simple(self, i: int, j: int) -> list[tuple[int, int]]:
        neighs = [
            self.pbc(i, j+1), # E
            self.pbc(i, j-1), # W
            self.pbc(i-1, j), # N
            self.pbc(i+1, j)  # S
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
        return self.relevance[self.grid_mask != 0].sum()

    def initialize(self):
        self.configuration = self.generate_random_configuration(self.snake_lengths)
        self.reward = self.reward_function(self.configuration)
        return

    def propose_move(self) -> tuple[Configuration, np.ndarray, int]:
        # Choose one snake to move.
        snake_id = np.random.randint(len(self.snake_lengths))
        ij_head = self.configuration[snake_id][0]
        allowed_ij = self.free_neighbors(*ij_head)
        proposed_configuration = self.configuration.copy()
        proposed_grid_mask = self.grid_mask.copy()
        if len(allowed_ij) == 0:
            # TODO move from tail
            self.trivial_steps += 1
            delta_reward = 0
        else:
            idx_h = np.random.randint(0, len(allowed_ij))
            ij_h = allowed_ij[idx_h]
            i_h, j_h = ij_h
            # Move snake forward to ij_r (delete last segment of tail).
            i_t, j_t = self.configuration[snake_id][-1]
            proposed_configuration[snake_id] = [ij_h] + self.configuration[snake_id][:-1]
            proposed_grid_mask[i_h, j_h] = 1
            proposed_grid_mask[i_t, j_t] = 0
            delta_reward = self.relevance[i_h, j_h] - self.relevance[i_t, j_t]
        self.last_delta_reward = delta_reward
        return proposed_configuration, proposed_grid_mask, delta_reward

    def try_step(self, beta: float):
        proposed_conf, proposed_grid_mask, delta_reward = self.propose_move()
        self.attempted_moves += 1
        r = np.random.rand()
        if r < np.exp(beta * delta_reward):
            self.configuration = proposed_conf
            self.grid_mask = proposed_grid_mask
            self.reward += delta_reward
            self.accepted_moves += 1
            if self.reward > self.best_reward:
                self.best_reward = self.reward
                self.best_configuration = self.configuration

    def anneal(self, beta0: float, beta1: float, beta_mult: float = 1.05):
        beta = beta0
        n_steps_per_sweep = len(self.snake_lengths)
        self.reward_history = [self.reward]
        while beta < beta1:
            for k_step in range(n_steps_per_sweep):
                self.try_step(beta)
                try:
                    assert self.validate_conf()
                except AssertionError:
                    sys.stderr.write(f"{old_conf=}\n")
                    sys.stderr.write(f"{self.configuration=}\n")
                    raise AssertionError(f"Invalid configuration at step {k_step} of sweep {k_sweep}!")
                self.reward_history.append(self.reward)
                self.attempted_delta_reward_history.append(self.last_delta_reward)
            # Decrease temperature at the end of each sweep.
            beta *= beta_mult
            self.n_sweeps += 1

    def validate_conf(self) -> bool:
        flat_conf = [el for snake in self.configuration for el in snake]
        seen = set()
        dupes = set([x for x in flat_conf if x in seen or seen.add(x)])
        return len(dupes.intersection(self.wormholes)) == len(dupes)

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
