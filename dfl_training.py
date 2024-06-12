import logging
import argparse
import sys
import asyncio
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import syft as sy

def define_and_get_arguments(args=sys.argv[1:]):
    # 选定参数
    parser = argparse.ArgumentParser(
        description="Run federated learning using websocket client workers."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size of the training"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        help="batch size used for the test data"
    )
    parser.add_argument(
        "--training_rounds",
        type=int,
        default=5,
        help="number of federated learning rounds"
    )
    parser.add_argument(
        "--federate_after_n_batches",
        type=int,
        default=10,
        help="number of training steps performed on each remote worker before averaging",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="seed used for randomization"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket client workers will be started in verbose mode",
    )
    parser.add_argument("--stage", type=int, default=1, help="continual learning stage")

    args = parser.parse_args(args=args)
    return args