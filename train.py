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


def run_exp(args, job_type="try", project_name="HSL3", return_detail=False):
    # if torch.cuda.device_count() == 1:
    #     # TODO
    #     args.raw_data_path = "data/raw/"
    #     args.data_path = "/data/0shared/caiderun/HSL-data/"
    #     args.wandb_dir = "/data/0shared/caiderun/exp_logs/wandb_logs/"

    model_name = args.method

    fname = "config_old.yml" if args.old_hparam else "config.yml"
    with open(fname, "r") as setting:
        all_config = yaml.safe_load(setting)

    config = all_config["common"]
    config.update(all_config[args.dname]["HSL"])

    config.update(vars(args))
    args = argparse.Namespace(**config)

    data, split_idx_lst = get_data(args)
    if job_type == "intro" or job_type == "noise_exp" or job_type == "tune_noise":
        # randomly add or delete
        augment_incidence(data, args.random_aug_p, aug_type=args.random_aug_type)

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
        if job_type != "tune" and job_type != "tune_noise":
            wandb_run = wandb.init(
                reinit=True,
                project=project_name,
                dir=args.wandb_dir,
                config=args,
                name=model_name,
                job_type=job_type,
            )
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
            if model_name == "HSL" and args.discrete_sample == "hard_concrete":
                loss_l0 = args.L0lambda * (out[1] + out[2])
                loss = loss_l0 + criterion2(out[0][train_idx], data.y[train_idx])
            elif model_name == "HSL" and args.contrast:
                if args.contrast_type == 'neighbor':
                    loss_scl = args.lambda_contrast * model.supcon(
                        out[1][train_idx], mask2=model.scl_mask[train_idx][:, train_idx]
                    )
                else:
                    loss_scl = args.lambda_contrast * model.supcon(out[1][train_idx])
                loss = loss_scl + criterion2(out[0][train_idx], data.y[train_idx])
            else:
                loss = criterion2(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()

            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:5])

            if job_type != "tune" and job_type != "tune_noise":
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "loss_l0": loss_l0.detach().cpu().item()
                        if args.discrete_sample == "hard_concrete"
                        else -1,
                        "loss_scl": loss_scl.detach().cpu().item()
                        if args.contrast
                        else -1,
                        "train_acc": result[0],
                        "valid_acc": result[1],
                        "test_acc": result[2],
                        "edge_rate": result[4] if model_name == "HSL" else -1,
                        "inci_rate": result[5] if model_name == "HSL" else -1,
                        "valid_loss": result[6],
                        "test_loss": result[7],
                    }
                )

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

        if job_type != "tune" and job_type != "tune_noise":
            wandb.log(
                {
                    "time": end_time - start_time,
                    "best_step": res["step"],
                    "final_valid_acc": res["valid_acc"],
                    "final_test_acc": res["test_acc"],
                    "final_edge_rate": res["edge_rate"],
                    "final_inci_rate": res["inci_rate"],
                }
            )
            wandb_run.finish(quiet=True)

    if job_type != "tune" and job_type != "tune_noise":
        print(
            "{} {}, {}m{:.2f}s±{:.2f}s, Layer:{}, MLP:{}-{}, clf-hidden:{}, lr:{}, wd:{}, heads:{}, dropout:{}".format(
                args.method,
                args.dname,
                np.mean(runtime_list) // 60,
                np.mean(runtime_list) % 60,
                np.std(runtime_list),
                args.All_num_layers,
                args.MLP_num_layers,
                args.MLP_hidden,
                args.Classifier_hidden,
                args.lr,
                args.wd,
                args.heads,
                args.dropout,
            )
        )

    best_val, best_test, edge_rate, inci_rate = logger.print_statistics()

    if return_detail:
        return (
            best_val.cpu().numpy(),
            best_test.cpu().numpy(),
            edge_rate.cpu().numpy(),
            inci_rate.cpu().numpy(),
        )
    else:
        return best_test.mean()


class Logger(object):
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == (5 if self.info == "HSL" else 3)
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def get_result(self, run):
        result = 100 * torch.tensor(self.results[run])
        argmax = result[:, 1].argmax().item()
        if self.info == "HSL":
            return {
                "step": argmax,
                "valid_acc": result[argmax, 1],
                "test_acc": result[argmax, 2],
                "train_acc": result[argmax, 0],
                "edge_rate": result[argmax, 3],
                "inci_rate": result[argmax, 4],
            }
        else:
            return {
                "step": argmax,
                "valid_acc": result[argmax, 1],
                "test_acc": result[argmax, 2],
                "train_acc": result[argmax, 0],
            }

    def print_statistics(self):
        best_results = []
        for i in range(len(self.results)):
            r = self.get_result(i)
            if self.info == "HSL":
                best_results.append(
                    (
                        r["valid_acc"],
                        r["train_acc"],
                        r["test_acc"],
                        r["edge_rate"],
                        r["inci_rate"],
                    )
                )
            else:
                best_results.append((r["valid_acc"], r["train_acc"], r["test_acc"]))

        best_result = torch.tensor(best_results)

        b = best_result[:, 0]
        c = best_result[:, 1]
        d = best_result[:, 2]
        print(
            f"Test: {d.mean():.2f} ± {d.std():.2f},\tValid: {b.mean():.2f} ± {b.std():.2f},\tTrain: {c.mean():.2f} ± {c.std():.2f}",
            time.strftime("%m-%d_%H:%M:%S", time.localtime()),
        )
        if self.info == "HSL":
            e = best_result[:, 3]
            f = best_result[:, 4]
            print(
                f"edge rate: {e.mean():.2f} ± {e.std():.2f}, inci rate: {f.mean():.2f} ± {f.std():.2f}"
            )
            return b, d, e, f
        else:
            return b, d

    def plot_result(self, run=None):
        plt.style.use("seaborn")
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f"Run {run + 1:02d}:")
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(["Train", "Valid", "Test"])


@torch.no_grad()
def evaluate(model, data, split_idx, eval_func):
    model.eval()
    edge_rate = -1
    incidence_rate = -1
    if model.__class__.__name__ != "HSL":
        out = model(data)
    else:
        out, masks = model(data, return_mask=True)

        if masks[0] is not None:
            edge_mask = masks[0].numpy()
            if masks[1] is not None:
                final_inci = model.final_edge_index
                inci_mask = masks[1].numpy().squeeze()
                msk = np.zeros_like(edge_mask)
                msk[final_inci[1, inci_mask.nonzero()[0]]] = 1
                edge_mask *= msk
            edge_rate = edge_mask.sum() / data.num_hyperedges

        if masks[1] is not None:
            inci_mask = masks[1].numpy().squeeze()
            if masks[0] is not None:
                msk = np.zeros_like(inci_mask)
                selected_edge_no = edge_mask.nonzero()[0]
                for i in selected_edge_no:
                    msk[final_inci[1] == i] = 1
                inci_mask = inci_mask * msk

            incidence_rate = inci_mask.sum() / model.num

    out = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(data.y[split_idx["test"]], out[split_idx["test"]])

    train_loss = F.nll_loss(out[split_idx["train"]], data.y[split_idx["train"]]).item()
    valid_loss = F.nll_loss(out[split_idx["valid"]], data.y[split_idx["valid"]]).item()
    test_loss = F.nll_loss(out[split_idx["test"]], data.y[split_idx["test"]]).item()

    return (
        train_acc,
        valid_acc,
        test_acc,
        edge_rate,
        incidence_rate,
        train_loss,
        valid_loss,
        test_loss,
    )


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def augment_incidence(data, p, aug_type="node"):
    """
    For example, p = -0.3 : delete 30%, p = 0.3 : add 30%
    """
    if p == 0:
        return
    edge_index = data.edge_index
    _, sorted_idx = torch.sort(edge_index[1])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)  # sorted by HE idx

    he_start = edge_index[1].min().item()
    first_selfloop_idx = he_start + data.num_hyperedges
    selfloop_start = torch.where(edge_index[1] == first_selfloop_idx)[0].min()

    row, col = edge_index[:, :selfloop_start]
    num = row.shape[0]

    if aug_type == "node":
        # delete or add node from hyperedges
        if p < 0:
            mask_idx = np.random.choice(num, int(abs(p) * num), replace=False)
            mask = np.ones(num, dtype=bool)
            mask[mask_idx] = False

            left_edge_index = torch.stack([row[mask], col[mask]], dim=0)
            new_edge_index = torch.cat(
                [left_edge_index, edge_index[:, selfloop_start:]], dim=1
            )
        else:
            row = np.random.choice(data.n_x, int(num * p))
            col = np.random.choice(data.num_hyperedges, int(num * p)) + he_start
            left_edge_index = torch.tensor(np.stack([row, col]))
            new_edge_index = torch.cat([left_edge_index, edge_index], dim=1)

    elif aug_type == "edge":
        # delete hyperedges
        assert p <= 0
        mask_idx = np.random.choice(
            data.num_hyperedges, int(abs(p) * data.num_hyperedges), replace=False
        )
        mask = np.ones(num, dtype=bool)

        mask_idx += he_start
        for h in mask_idx:
            a = torch.where(col == h)[0]
            mask[a] = False

        left_edge_index = torch.stack([row[mask], col[mask]], dim=0)
        new_edge_index = torch.cat(
            [left_edge_index, edge_index[:, selfloop_start:]], dim=1
        )

    _, sorted_idx = torch.sort(new_edge_index[1])
    new_edge_index = new_edge_index[:, sorted_idx].type(
        torch.LongTensor
    )  # sorted by HE idx

    data.edge_index = new_edge_index


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_exp(args, job_type=args.job_type, project_name=args.project_name)
