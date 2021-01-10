import argparse
import h5py
import numpy as np
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aff', 
        type=str)
    parser.add_argument(
        '--bdr', 
        type=str)
    parser.add_argument(
        '--bdr_thresh',
        type=float,
        default=0.5)
    parser.add_argument(
        '--stride', 
        type=int,
        default=2)
    args = parser.parse_args()

    # DeepEMbedding package
    home = os.path.expanduser('~')
    pkg_path = os.path.join(home, 'Workbench/torms3/DeepEMbedding/python')
    sys.path.append(pkg_path)

    # Data    
    with h5py.File(args.aff, 'r') as f:
        aff = f['/main'][...].astype(np.float32)
    with h5py.File(args.bdr, 'r') as f:
        bdr = f['/main'][...].astype(np.float32)

    # Mutex Watershed
    from mws import MutexWatershed
    offsets = [[0, 0, -1], [0, -1, 0], [-1, 0, 0],
            [0, 0, -5], [0, -5, 0], [0, -5, -5], [0, 5, -5],
            [-1, 0, -5], [-1, -5, 0], [1, 0, -5], [1, -5, 0],
            [-2, 0, 0],
            ]
    mws = MutexWatershed(aff, bdr, offsets)
    seg, mst = mws.run(bdr_thresh=args.bdr_thresh, stride=args.stride)