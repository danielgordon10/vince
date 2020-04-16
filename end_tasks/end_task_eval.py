import torch.multiprocessing as multiprocessing

import arg_parser


def main():
    args = arg_parser.parse_args()
    solver = args.solver(args, None, None)
    solver.run_eval()


if __name__ == "__main__":
    cxt = multiprocessing.get_context()
    print(cxt.get_start_method())
    main()
