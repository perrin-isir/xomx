import argparse
import os
import numpy as np
from IPython import embed

debug = embed


def get_args(default_folder):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step", metavar="S", type=int, nargs="?", default=None, help="execute step S"
    )
    parser.add_argument(
        "--n_total_steps", default=1, help="total number of steps in the tutorial"
    )
    parser.add_argument(
        "--savedir",
        default=os.path.join(
            os.path.expanduser("~"), "results", "xomx", default_folder
        ),
        help="directory in which data and outputs will be stored",
    )
    args_ = parser.parse_args()
    return args_


def step_init(args, n_total_steps):
    args.n_total_steps = n_total_steps
    if args.step is not None:
        assert 1 <= args.step <= n_total_steps
        stp = args.step
    elif not os.path.exists(os.path.join(args.savedir, "next_step.txt")):
        stp = 1
    else:
        stp = np.loadtxt(os.path.join(args.savedir, "next_step.txt"), dtype="int")
    print("STEP", stp)
    return stp


def step_increment(stp, args):
    # noinspection PyTypeChecker
    np.savetxt(
        os.path.join(args.savedir, "next_step.txt"),
        [min(stp + 1, args.n_total_steps)],
        fmt="%u",
    )
