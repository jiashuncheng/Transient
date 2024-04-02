import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true",):
        return True
    elif isinstance(v, str) and v.lower() in ("false",):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def paramaters():
    print("--> Loading parameters...")
    p = argparse.ArgumentParser()
    #ANCHOR - All
    p.add_argument('--delay', type=str, default='6',choices=['random', '3', '4', '5', '6'])
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--mode', type=str, default='test',choices=['train','test'])
    p.add_argument('--exp', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--v_m', type=float, default=0.2)
    p.add_argument('--v_gamma', type=float, default=1.0)
    p.add_argument('--rnn_bias', type=float, default=0)
    p.add_argument('--noise_h', type=str2bool, default=False)
    p.add_argument('--batch_size', type=int, default=64)
    #ANCHOR - train
    p.add_argument('--num_iterations', type=int, default=1000)
    p.add_argument('--max_delay', type=int, default=6)
    p.add_argument('--hidden_num', type=int, default=300)
    p.add_argument('--transient', type=str2bool, default=True)
    p.add_argument('--regions', type=str2bool, default=True)
    p.add_argument('--in_', type=str2bool, default=True)
    p.add_argument('--out', type=str2bool, default=True)
    p.add_argument('--save_model', action='store_true')
    p.add_argument('--connection_prob', type=float, default=0.8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--alpha', type=float, default=0.98)
    #ANCHOR - test
    p.add_argument('--model_path', type=str, default=None)
    p.add_argument('--sample', type=int, default=0, choices=[0, 1])
    p.add_argument('--match', type=int, default=0, choices=[0, 1])
    p.add_argument('--theta', type=int, default=180)

    args, _ = p.parse_known_args()

    if args.delay == '3':
        args.delay = 3
    elif args.delay == '4':
        args.delay = 4
    elif args.delay == '5':
        args.delay = 5
    elif args.delay == '6':
        args.delay = 6

    print('Optional argument values:')
    for key, value in vars(args).items():
        print('--', key, ':', value)

    return args

args = paramaters()