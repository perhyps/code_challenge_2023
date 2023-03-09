from argparse import ArgumentParser
import os
import pickle
import sys

import matplotlib.pyplot as plt

from lib import *


def main():
    parser = ArgumentParser()

    parser.add_argument("-b", dest="beta", type=float, default=1., help="Inverse temperature.")
    parser.add_argument("-i", dest="input_path", type=str, required=True, help="Path to input file.")
    parser.add_argument("-o", dest="output_path", type=str, default="best_config.p", help="Path to output file.")

    args = parser.parse_args()

    C, R, S, snake_lengths, _, relevance, wormholes = read_data(args.input_path)

    grid = Grid(R=R, C=C, relevance=relevance, snake_lengths=snake_lengths, wormholes=wormholes)
    grid.initialize()
    rewards = [grid.reward]
    attempted_delta_rewards = []

    nsteps = 10_000
    for k_step in range(nsteps):
        old_conf = grid.configuration
        grid.try_step(args.beta)
        try:
            assert grid.validate_conf()
        except AssertionError:
            sys.stderr.write(f"{old_conf=}\n")
            sys.stderr.write(f"{grid.configuration=}\n")
            raise AssertionError(f"Invalid configuration at step {k_step}!")
        rewards.append(grid.reward)
        attempted_delta_rewards.append(grid.last_delta_reward)
    attempted_delta_rewards = np.array(attempted_delta_rewards)

    sys.stdout.write(f"# Reward\n")
    sys.stdout.write(f"#    initial: {rewards[0]}\n")
    sys.stdout.write(f"#    best   : {grid.best_reward}\n")
    sys.stdout.write(f"#    final  : {rewards[-1]}\n")
    sys.stdout.write(f"# Acceptance rate: {grid.acceptance_rate()*100:.2f}%\n")
    sys.stdout.write(f"# Trivial steps  : {grid.trivial_steps/grid.attempted_moves*100:.2f}%\n")
    sys.stdout.write(f"# Delta reward\n")
    sys.stdout.write(f"#   avg    : {np.average(attempted_delta_rewards):.3f}\n")
    sys.stdout.write(f"#   avg nnz: {np.average(attempted_delta_rewards[attempted_delta_rewards != 0]):.3f}\n")
    sys.stdout.write(f"#   max    : {np.max(attempted_delta_rewards)}\n")

    # Save best configuration.
    output_path = args.output_path
    with open(output_path, "wb") as output:
        pickle.dump(grid.best_configuration, output)
        sys.stderr.write(f"Output saved to <{output_path}>.\n")

    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")

    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    exit_val = main()
    sys.exit(exit_val)
