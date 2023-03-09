from argparse import ArgumentParser
import os
import pickle
import sys

import matplotlib.pyplot as plt

from lib import *


def main():
    parser = ArgumentParser()

    parser.add_argument("-b", dest="beta", nargs=2, type=float, default=(.1, 10.), help="Starting and ending inverse temperatures.")
    parser.add_argument("-m", dest="beta_mult", type=float, default=1.05, help="Multiplier for beta after each sweep.")
    parser.add_argument("-i", dest="input_path", type=str, required=True, help="Path to input file.")
    parser.add_argument("-o", dest="output_path", type=str, default=None, help="Path to output file.")
    parser.add_argument("-p", action="store_true", dest="plot_reward", default=False, help="Plot reward history.")

    args = parser.parse_args()

    C, R, S, snake_lengths, _, relevance, wormholes = read_data(args.input_path)

    grid = Grid(R=R, C=C, relevance=relevance, snake_lengths=snake_lengths, wormholes=wormholes)
    grid.initialize()

    beta0, beta1 = args.beta
    grid.anneal(beta0=beta0, beta1=beta1, beta_mult=args.beta_mult)
    reward_history = np.array(grid.reward_history)
    attempted_delta_reward_history = np.array(grid.attempted_delta_reward_history)

    sys.stdout.write(f"# No. sweeps: {grid.n_sweeps}\n")
    sys.stdout.write(f"# Reward\n")
    sys.stdout.write(f"#    initial: {reward_history[0]}\n")
    sys.stdout.write(f"#    best   : {grid.best_reward}\n")
    sys.stdout.write(f"#    final  : {reward_history[-1]}\n")
    sys.stdout.write(f"# Acceptance rate: {grid.acceptance_rate()*100:.2f}%\n")
    sys.stdout.write(f"# Trivial steps  : {grid.trivial_steps/grid.attempted_moves*100:.2f}%\n")
    sys.stdout.write(f"# Delta reward\n")
    sys.stdout.write(f"#   avg    : {np.average(attempted_delta_reward_history):.3f}\n")
    sys.stdout.write(f"#   avg nnz: {np.average(attempted_delta_reward_history[attempted_delta_reward_history != 0]):.3f}\n")
    sys.stdout.write(f"#   max    : {np.max(attempted_delta_reward_history)}\n")

    # Save best configuration.
    output_path = (
        args.output_path if args.output_path is not None
        else os.path.join("output", ".".join(os.path.basename(args.input_path).split(".")[:-1]) + ".p")
    )
    overwrite_file = True
    if os.path.isfile(output_path):
        with open(output_path, "rb") as infile:
            data = pickle.load(infile)
            if (pr := data["reward"]) <= grid.best_reward:
                pass
            else:
                sys.stderr.write(f"Not overwriting file as previous reward is better ({pr} > {grid.best_reward}).\n")
                overwrite_file = False
    if overwrite_file:
        with open(output_path, "wb") as output:
            pickle.dump({"reward": grid.best_reward, "conf": grid.best_configuration}, output)
            sys.stderr.write(f"Output saved to <{output_path}>.\n")

    if args.plot_reward:
        fig, ax = plt.subplots()
        ax.plot(reward_history)
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    exit_val = main()
    sys.exit(exit_val)
