import scipy.sparse as sp
import torch
import torch_sparse

from convert_datasets_to_pygDataset import dataset_Hypergraph
from preprocessing import *


def get_data(args):
    ### Load and preprocess data ###
    existing_dataset = [
        "20newsW100",
        "ModelNet40",
        "zoo",
        "NTU2012",
        "Mushroom",
        "coauthor_cora",
        "coauthor_dblp",
        "yelp",
        "amazon-reviews",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
        "cora",
        "citeseer",
        "pubmed",
    ]

    synthetic_list = [
        "amazon-reviews",
        "walmart-trips",
        "house-committees",
        "walmart-trips-100",
        "house-committees-100",
    ]

    if args.dname in existing_dataset:
        dname = args.dname
        p2raw = args.raw_data_path
        root = args.data_path
        if dname in synthetic_list:
            f_noise = args.feature_noise
            if f_noise is not None:
                dataset = dataset_Hypergraph(name=dname, feature_noise=f_noise, p2raw=p2raw, root=root)
        else:
            dataset = dataset_Hypergraph(name=dname, p2raw=p2raw, root=root)
        data = dataset.data
        args.num_features = dataset.num_features
        args.num_classes = dataset.num_classes
        if args.dname in [
            "yelp",
            "walmart-trips",
            "house-committees",
            "walmart-trips-100",
            "house-committees-100",
        ]:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, "n_x"):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, "num_hyperedges"):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor([data.edge_index[0].max() - data.n_x + 1])

    if args.method in ["AllSetTransformer", "AllDeepSets", "HSL"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)

        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ["CEGCN", "CEGAT"]:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE="V2V")

    elif args.method in ["HyperGCN"]:
        data = ExtractV2E(data)
    #     ipdb.set_trace()
    #   Feature normalization, default option in HyperGCN
    # X = data.x
    # X = sp.csr_matrix(utils.normalise(np.array(X)), dtype=np.float32)
    # X = torch.FloatTensor(np.array(X.todense()))
    # data.x = X

    # elif args.method in ['HGNN']:
    #     data = ExtractV2E(data)
    #     if args.add_self_loop:
    #         data = Add_Self_Loops(data)
    #     data = ConstructH(data)
    #     data = generate_G_from_H(data)

    elif args.method in ["HNHN"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ["HCHA", "HGNN"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ["UniGCNII", "DHGNN"]:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        # Compute degV and degE

        # not needed for UniGAT
        # device = torch.device(
        #     "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
        # )

        # (row, col), value = torch_sparse.from_scipy(data.edge_index)
        # V, E = row, col
        # V, E = V.to(device), E.to(device)

        # degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
        # from torch_scatter import scatter

        # degE = scatter(degV[V], E, dim=0, reduce="mean")
        # degE = degE.pow(-0.5)
        # degV = degV.pow(-0.5)
        # degV[degV.isinf()] = 1
        # args.UniGNN_degV = degV
        # args.UniGNN_degE = degE

        # V, E = V.cpu(), E.cpu()
        # del V
        # del E

    #     Get splits
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)

    return data, split_idx_lst
