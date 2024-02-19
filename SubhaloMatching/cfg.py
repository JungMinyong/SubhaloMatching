import argparse
import multiprocessing


def get_cfg():
    """ generates configuration from user input in console """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--fname", type=str, help="load model from file")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--adam_l2", type=float, default=1e-6, help="weight decay (L2 penalty)"
    )
    parser.add_argument(
        "--loss2_l2", type=float, default=1e-6, help="weight decay (smoothing loss)"
    )
    parser.add_argument("--epoch", type=int, default=200, help="total epoch number")
    parser.add_argument(
        "--eval_every", type=int, default=5, help="how often to evaluate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="dataset workers number",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=12345,
        help="trained bins",
    )
    parser.add_argument('--tng50', action='store_true')
    parser.add_argument('-w', '--wandb', action='store_true')   
    parser.add_argument('-r', '--restart', action='store_true')    
    parser.add_argument('--batch', type=int, default=32, help="batch size")
    parser.add_argument('-N', '--Nsample', type=int, default=1e10, help="sample size")    
    parser.add_argument('--bin', type=int, default=0, help="index of mass bin")

    return parser.parse_args()
#python parameter_train-valid.py --bins 123456 --batch 32 --num_worker 48 --eval_every 1

            