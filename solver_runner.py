import os
import traceback

import torch.multiprocessing as multiprocessing
import tqdm
from dg_util.python_utils import tensorboard_logger

import arg_parser
from solvers.base_solver import BaseSolver


def main():
    args = arg_parser.parse_args()

    if args.debug:
        train_logger = None
        val_logger = None
    else:
        train_logger = tensorboard_logger.Logger(os.path.join(args.tensorboard_dir, "train"))
        val_logger = tensorboard_logger.Logger(os.path.join(args.tensorboard_dir, "val"))

    solver: BaseSolver = args.solver(args, train_logger, val_logger)

    curr_iteration = 1
    try:
        if args.test_first:
            print("Running initial Val")
            solver.reset_epoch()
            solver.run_val()

        starting_lr = solver.adjust_learning_rate()
        while solver.epoch < args.epochs:
            solver.reset_epoch()
            print("Running Train")
            for ii in tqdm.tqdm(range(solver.iterations_per_epoch)):
                if args.use_warmup:
                    if curr_iteration <= 500:
                        lr_scale = min(1.0, curr_iteration / 500.0)
                        new_lr = lr_scale * starting_lr
                        for pg in solver.optimizer.param_groups:
                            pg["lr"] = new_lr
                        print("new lr", new_lr)
                        curr_iteration += 1
                output = solver.run_train_iteration()
            print("Running Val")
            solver.run_val()
            solver.epoch += 1
        solver.end()
    except:
        traceback.print_exc()
    finally:
        if args.save:
            print("Saving models")
            solver.save()


if __name__ == "__main__":
    cxt = multiprocessing.get_context()
    print(cxt.get_start_method())
    main()
