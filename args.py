import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", default="try", type=str, choices=["try", "noise_exp", "iterate", "intro", "tune"])
    parser.add_argument("--project_name", default="HSL4", type=str)
    parser.add_argument(
        "--dname",
        default="cora",
        choices=[
            "20newsW100",
            "zoo",
            "NTU2012",
            "coauthor_cora",
            "coauthor_dblp",
            "cora",
            "citeseer",
        ],
    )
    parser.add_argument("--method", default="AllSetTransformer")
    parser.add_argument("--gpu", type=int, default=9, help="index of available GPUs")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--old_hparam", action="store_true", default=False)
    parser.add_argument("--discrete_sample", default="gumbel", choices=["gumbel", "hard_concrete"])
    parser.add_argument("--contrast", action="store_true", default=False)
    parser.add_argument("--p2sample_type", default="random", choices=["random", "topk_add", "topk_all"])
    parser.add_argument("--p2sample_add_p", default=0, type=float)
    parser.add_argument("--p1init_rate", default=3, type=float)
    parser.add_argument("--p2init_rate", default=2, type=float)

    parser.add_argument("--unignn_hsl", action="store_true", default=False)

    # for intro study
    # parser.add_argument("--random_aug_p", default=0, type=float, help="ratio of augmentation")
    # parser.add_argument(
    #     "--random_aug_type",
    #     default="node",
    #     type=str,
    #     choices=["node", "edge"],
    #     help="type of augmentation",
    # )
    return parser
