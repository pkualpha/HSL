import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def get_metrics(y_true, proba, verbose=0):
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, fbeta_score, top_k_accuracy_score

    assert (y_true == y_true).sum() == y_true.shape[0]
    y_pred = proba.argmax(axis=-1)
    acc = accuracy_score(y_true, y_pred)
    if verbose:
        print("accuracy = {}".format(acc))
    return {"acc": acc}


def write_json(data, path, sort_keys=False, verbose=1):
    with open(path, "w") as outfile:
        json.dump(data, outfile, sort_keys=sort_keys, indent=4)
    if verbose:
        print("saved to ", path)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    """Adapted from https://github.com/snap-stanford/ogb/"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")

            return best_result[:, 1], best_result[:, 3]

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
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(data.y[split_idx["train"]], out[split_idx["train"]])
    valid_acc = eval_func(data.y[split_idx["valid"]], out[split_idx["valid"]])
    test_acc = eval_func(data.y[split_idx["test"]], out[split_idx["test"]])

    #     Also keep track of losses
    train_loss = F.nll_loss(out[split_idx["train"]], data.y[split_idx["train"]])
    valid_loss = F.nll_loss(out[split_idx["valid"]], data.y[split_idx["valid"]])
    test_loss = F.nll_loss(out[split_idx["test"]], data.y[split_idx["test"]])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    #     ipdb.set_trace()
    #     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
