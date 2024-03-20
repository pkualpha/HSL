import math

import numpy as np
import pyro
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.utils import softmax
from torch_scatter import scatter

from models.layers import MLP, HalfNLHconv

gamma = -0.1
zeta = 1.1
eps = 1e-20


class HSL(nn.Module):
    def __init__(self, args, data):
        super(HSL, self).__init__()
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.aggr = args.aggregate
        self.NormLayer = args.normalization
        self.InputNorm = True
        self.GPR = args.GPR
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()

        self.first_selfloop_idx = data.edge_index[1].min().item() + data.num_hyperedges
        self.num = (
            torch.where(data.edge_index[1] == self.first_selfloop_idx)[0].min().item()
        )
        # num of non self loop edge index
        self.p1sample, self.p2sample = args.p1sample, args.p2sample
        if self.p1sample or self.p2sample:
            self.beta = args.hc_beta
            self.const1 = self.beta * np.log(-gamma / zeta + eps)
            self.discrete_sample = args.discrete_sample

        if self.p1sample:
            self.p1temperature = args.p1temperature
            self.p1useMLP = args.p1useMLP
            self.p1init_rate = args.p1init_rate
            if self.p1useMLP:
                self.p1MLP = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.p2MLP_hidden,
                    out_channels=1,
                    num_layers=args.p2MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
                nn.init.constant_(self.p1MLP.lins[-1].bias, 2)
            else:
                self.edge_mask = Parameter(
                    self.p1init_rate * torch.ones(data.num_hyperedges)
                )
                # only mask non self loop hyperedges

        self.incident_mask = None
        if self.p2sample:
            self.p2temperature = args.p2temperature
            self.p2sample_add_p = args.p2sample_add_p
            self.p2sample_type = args.p2sample_type
            self.p2init_rate = args.p2init_rate
            self.p2MLP = MLP(
                in_channels=args.MLP_hidden * 2,
                hidden_channels=args.p2MLP_hidden,
                out_channels=1,
                num_layers=args.p2MLP_num_layers,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False,
            )
            nn.init.constant_(self.p2MLP.lins[-1].bias, self.p2init_rate)
            self.cos_weight = nn.Parameter(torch.Tensor(args.cos_head, args.MLP_hidden))

        self.contrast = args.contrast
        if self.contrast:
            self.lambda_contrast = args.lambda_contrast
            self.supcon = SupConLoss()
            self.contrast_type = args.contrast_type
            if self.contrast_type == 'neighbor':
                self.scl_mask = self.get_scl_mask(data)

        if self.All_num_layers == 0:
            self.classifier = MLP(
                in_channels=args.num_features,
                hidden_channels=args.Classifier_hidden,
                out_channels=args.num_classes,
                num_layers=args.Classifier_num_layers,
                dropout=self.dropout,
                Normalization=self.NormLayer,
                InputNorm=False,
            )
        else:
            self.V2EConvs.append(
                HSLLayer(
                    in_channels=args.num_features,
                    hid_dim=args.MLP_hidden,
                    out_channels=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    heads=args.heads,
                )
            )
            self.E2VConvs.append(
                HSLLayer(
                    in_channels=args.MLP_hidden,
                    hid_dim=args.MLP_hidden,
                    out_channels=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    heads=args.heads,
                )
            )
            for _ in range(self.All_num_layers - 1):
                self.V2EConvs.append(
                    HSLLayer(
                        in_channels=args.MLP_hidden,
                        hid_dim=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=args.MLP_num_layers,
                        heads=args.heads,
                    )
                )
                self.E2VConvs.append(
                    HSLLayer(
                        in_channels=args.MLP_hidden,
                        hid_dim=args.MLP_hidden,
                        out_channels=args.MLP_hidden,
                        num_layers=args.MLP_num_layers,
                        heads=args.heads,
                    )
                )
            if self.GPR:
                self.MLP = MLP(
                    in_channels=args.num_features,
                    hidden_channels=args.MLP_hidden,
                    out_channels=args.MLP_hidden,
                    num_layers=args.MLP_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
                self.GPRweights = Linear(self.All_num_layers + 1, 1, bias=False)
                self.classifier = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.Classifier_hidden,
                    out_channels=args.num_classes,
                    num_layers=args.Classifier_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )
            else:
                self.classifier = MLP(
                    in_channels=args.MLP_hidden,
                    hidden_channels=args.Classifier_hidden,
                    out_channels=args.num_classes,
                    num_layers=args.Classifier_num_layers,
                    dropout=self.dropout,
                    Normalization=self.NormLayer,
                    InputNorm=False,
                )

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.p1sample:
            if self.p1useMLP:
                self.p1MLP.reset_parameters()
                nn.init.constant_(self.p1MLP.lins[-1].bias, self.p1init_rate)
            else:
                nn.init.constant_(self.edge_mask, self.p1init_rate)
        if self.p2sample:
            self.p2MLP.reset_parameters()
            nn.init.constant_(self.p2MLP.lins[-1].bias, self.p2init_rate)
            nn.init.xavier_uniform_(self.cos_weight)

    def forward(self, data, return_mask=False):
        data.edge_index[1] -= data.edge_index[1].min()
        x, edge_index = data.x, data.edge_index
        reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        if self.contrast:
            c_edge_index = edge_index
            c_reversed_edge_index = reversed_edge_index

        if self.p2sample:
            p = self.p2sample_add_p
            all_node = self.V2EConvs[0].lin_V(x)

            if p > 0:
                if self.p2sample_type == "random":
                    # randomly add incident nodes per epoch
                    row = np.random.choice(data.n_x, int(self.num * p))
                    col = np.random.choice(data.num_hyperedges, int(self.num * p))
                    add_incidence = torch.tensor(
                        np.stack([row, col]),
                        device=edge_index.device,
                        dtype=edge_index.dtype,
                    )
                else:
                    with torch.no_grad():
                        all_he_m = self.V2EConvs[0](x, edge_index, None)[
                            : data.num_hyperedges
                        ]
                        all_node_m = all_node
                        node_len = all_node_m.shape[0]
                        edge_len = all_he_m.shape[0]

                        if node_len > 10000:
                            downsample = 26 if node_len > 1000 else 5
                            selected_node = torch.randperm(
                                node_len, device=all_node.device
                            )[: node_len // downsample]
                            selected_edge = torch.randperm(
                                edge_len, device=all_node.device
                            )[: edge_len // downsample]
                            all_node_m = all_node_m[selected_node]
                            all_he_m = all_he_m[selected_edge]

                        node_fc = all_node_m.unsqueeze(1) * self.cos_weight
                        all_node_m = F.normalize(node_fc, p=2, dim=-1).permute(
                            (1, 0, 2)
                        )

                        edge_fc = all_he_m.unsqueeze(1) * self.cos_weight
                        all_he_m = F.normalize(edge_fc, p=2, dim=-1).permute((1, 2, 0))

                        H = torch.matmul(all_node_m, all_he_m).mean(0)

                        if node_len <= 10000 and self.p2sample_type == "topk_add":
                            r1, c1 = edge_index[:, : self.num]
                            H[r1, c1] = -1e6

                        v, i = torch.topk(H.flatten(), int(self.num * p))
                        row = torch.div(i, H.shape[1], rounding_mode="floor")
                        col = i % H.shape[1]

                        if node_len > 10000:
                            row = selected_node[row]
                            col = selected_edge[col]

                        add_incidence = torch.stack([row, col])

                no_loop_edge_index = torch.cat(
                    [add_incidence, edge_index[:, : self.num]], dim=1
                )
            else:
                no_loop_edge_index = edge_index[:, : self.num]

            if not self.training:
                self.final_edge_index = no_loop_edge_index.cpu().numpy()

            node_idx, he_idx = no_loop_edge_index
            edge_index = torch.cat(
                [no_loop_edge_index, edge_index[:, self.num :]], dim=1
            )
            reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)

            x_node = all_node[node_idx]
            all_he = self.V2EConvs[0](x, edge_index, None)
            x_he = all_he[he_idx]

            incident_mask = self.p2MLP(torch.cat([x_node, x_he], dim=1))
            incident_mask = torch.sigmoid(incident_mask)
            if self.discrete_sample == "gumbel":
                self.incident_mask = pyro.distributions.RelaxedBernoulliStraightThrough(
                    temperature=self.p2temperature, probs=incident_mask
                ).rsample()
            else:
                self.incident_mask = self.hard_concrete_sample(incident_mask)
                p2loss = self.L0Loss(incident_mask)

        if self.p1sample:
            if self.p1useMLP:
                edge_mask = self.p1MLP(all_he).squeeze()
            else:
                edge_mask = self.edge_mask
            if self.discrete_sample == "gumbel":
                edge_mask = pyro.distributions.RelaxedBernoulliStraightThrough(
                    temperature=self.p1temperature, probs=torch.sigmoid(edge_mask)
                ).rsample()
            else:
                edge_mask = self.hard_concrete_sample(edge_mask)
                p1loss = self.L0Loss(edge_mask)

        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, self.incident_mask))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, self.incident_mask)
                x = F.relu(x)
                xs.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
        else:
            x = F.dropout(x, p=0.2, training=self.training)  # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, self.incident_mask))
                if self.p1sample:
                    x[: edge_mask.shape[0], :] * edge_mask.unsqueeze(-1)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](x, reversed_edge_index, self.incident_mask))
                x = F.dropout(x, p=self.dropout, training=self.training)

        if return_mask:
            return self.classifier(x), (
                edge_mask.detach().cpu() if self.p1sample else None,
                self.incident_mask.detach().cpu() if self.p2sample else None,
            )
        elif self.p1sample or self.p2sample:
            if self.discrete_sample == "hard_concrete":
                return (
                    self.classifier(x),
                    p1loss if self.p1sample else None,
                    p2loss if self.p2sample else None,
                )
            elif self.contrast:
                if self.contrast_type == 'neighbor':
                    self.scl_mask = self.scl_mask.to(data.x.device)
                cx = F.dropout(data.x, p=0.2, training=self.training)  # Input dropout
                for i, _ in enumerate(self.V2EConvs):
                    cx = F.relu(self.V2EConvs[i](cx, c_edge_index, None))
                    cx = F.dropout(cx, p=self.dropout, training=self.training)
                    cx = F.relu(self.E2VConvs[i](cx, c_reversed_edge_index, None))
                    cx = F.dropout(cx, p=self.dropout, training=self.training)
                feat = torch.stack([x, cx], dim=1)
                return self.classifier(x), feat
            else:
                return self.classifier(x)
        else:
            return self.classifier(x)

    def L0Loss(self, logAlpha):
        return torch.sigmoid(logAlpha - self.const1).mean()

    def hard_concrete_sample(self, logAlpha, min=0, max=1):
        if self.training:
            U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps
            s = torch.sigmoid((torch.log(U / (1 - U)) + logAlpha) / self.beta)
        else:
            s = torch.sigmoid(logAlpha / self.beta)
        s_bar = s * (zeta - gamma) + gamma
        mask = F.hardtanh(s_bar, min, max)
        return mask

    def get_scl_mask(self, data):
        """
        Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
        """
        edge_index = np.array(data.edge_index)
        # Don't use edge_index[0].max()+1, as some nodes maybe isolated
        num_nodes = data.x.shape[0]
        num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
        H = np.zeros((num_nodes, num_hyperedges))
        cur_idx = 0
        for he in np.unique(edge_index[1]):
            nodes_in_he = edge_index[0][edge_index[1] == he]
            H[nodes_in_he, cur_idx] = 1.0
            cur_idx += 1

        H = sp.csr_matrix(H)
        mask = torch.tensor(((H * H.T).todense() != 0) * 1)
        return mask


class HSLLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_dim,
        out_channels,
        num_layers,
        heads=1,
        negative_slope=0.2,
    ):
        super(HSLLayer, self).__init__()
        self.aggr = "add"
        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope

        self.lin_K = Linear(in_channels, self.heads * self.hidden)
        self.lin_V = Linear(in_channels, self.heads * self.hidden)
        self.att_r = Parameter(torch.Tensor(1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(
            in_channels=self.heads * self.hidden,
            hidden_channels=self.heads * self.hidden,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=0.0,
            Normalization="None",
        )
        self.ln0 = nn.LayerNorm(self.heads * self.hidden)
        self.ln1 = nn.LayerNorm(self.heads * self.hidden)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        nn.init.xavier_uniform_(self.att_r)

    def forward(
        self, x, edge_index, incident_mask=None, return_attention_weights=False
    ):
        H, C = self.heads, self.hidden

        x_K = self.lin_K(x).view(-1, H, C)
        x_V = self.lin_V(x).view(-1, H, C)
        alpha_r = (x_K * self.att_r).sum(dim=-1)  # (B, H)

        # out = self.propagate(edge_index, x=x_V, alpha=alpha_r, aggr=self.aggr)
        x = x_V[edge_index[0]]
        alpha = F.leaky_relu(alpha_r[edge_index[0]], self.negative_slope)
        alpha = softmax(alpha, edge_index[1], num_nodes=edge_index[1].max() + 1)
        if incident_mask is not None:
            alpha[: incident_mask.shape[0], :] *= incident_mask
        x = x * alpha.unsqueeze(-1)

        out = scatter(x, edge_index[1], dim=0, reduce=self.aggr)

        out += self.att_r
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        out = self.ln1(out + F.relu(self.rFF(out)))

        return (out, (edge_index, alpha)) if return_attention_weights else out


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """

    def __init__(self, temperature=0.07, contrast_mode="all"):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None, mask2=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device
        features = F.normalize(features, p=2, dim=-1)
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # z = torch.cat((z1, z2), dim=0)
        # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        if mask2 is not None:
            mask2 = mask2.repeat(anchor_count, contrast_count).to(device)
            row_select = (mask2).sum(axis=1) > 2
            # print(logits.shape, mask2.shape, logits_mask.shape)
            logits_mask = logits_mask * mask2
            logits = logits[row_select]
            logits_mask = logits_mask[row_select]
            mask = mask[row_select]

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()
        # loss
        # loss = - (temperature / base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        # return loss
