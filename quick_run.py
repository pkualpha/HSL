import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import wandb
import yaml
from tqdm import tqdm

from args import get_parser
from data import get_data
from models.models import HSL
from train import Logger, eval_acc, evaluate, run_exp

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="index of available GPUs")
parser.add_argument("--dname", type=str, default='cora')
args = parser.parse_args()
args.job_type = "tune"
args.runs = 10
args.method = "HSL"
args.old_hparam = False


def run_exp(args, job_type="try", project_name="HSL3", return_detail=False):
    model_name = args.method

    fname = "config_old.yml" if args.old_hparam else "config.yml"
    with open(fname, "r") as setting:
        all_config = yaml.safe_load(setting)

    config = all_config["common"]
    config.update(all_config[args.dname]["HSL"])
    config.update(vars(args))
    args = argparse.Namespace(**config)

    data, split_idx_lst = get_data(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model = HSL(args, data)

    if device == "cpu":
        print("using cpu")
    data = data.to(device)
    model = model.to(device)

    logger = Logger(args.runs, args.method)
    criterion2 = nn.CrossEntropyLoss()
    eval_func = eval_acc
    model.train()

    ### Training loop ###
    runtime_list = []
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx["train"].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )

        best_val = 0
        lower_steps = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data)
            if args.discrete_sample == "hard_concrete":
                loss_l0 = args.L0lambda * (out[1] + out[2])
                loss = loss_l0 + criterion2(out[0][train_idx], data.y[train_idx])
            elif args.contrast:
                loss_scl = args.lambda_contrast * model.supcon(
                    out[1][train_idx], data.y[train_idx]
                )
                loss = loss_scl + criterion2(out[0][train_idx], data.y[train_idx])
            else:
                loss = criterion2(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:5])

            if result[1] >= best_val:
                best_val = max(best_val, result[1])
                lower_steps = 0
            else:
                lower_steps += 1
                if lower_steps > args.early_stop:
                    break

        end_time = time.time()
        runtime_list.append(end_time - start_time)
        res = logger.get_result(run)

    best_val, best_test, edge_rate, inci_rate = logger.print_statistics()

    if return_detail:
        return (
            best_val.cpu().numpy(),
            best_test.cpu().numpy(),
            edge_rate.cpu().numpy(),
            inci_rate.cpu().numpy(),
        )
    else:
        return best_test.mean(), logger


choices = [
    "20newsW100",
    "zoo",
    "NTU2012",
    "coauthor_cora",
    "cora",
    "citeseer",
    "coauthor_dblp",
]

print(args.dname)
res, logger = run_exp(args, job_type='tune')
