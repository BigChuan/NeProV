import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def normt_spm(mx, method="in"):
    if method == "in":
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    if method == "sym":
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx
    raise ValueError(f"Unknown norm method: {method}")


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """
    Dense fp32 graph convolution.
    Keeps the public constructor compatible with the current CAMS code:
        GraphConv(in_channels, out_channels, dropout=..., relu=...)
    """
    def __init__(self, in_channels, out_channels, dropout=0.0, relu=True):
        super().__init__()
        self.dropout = float(dropout)
        self.layer = nn.Linear(in_channels, out_channels)
        self.relu = bool(relu)

    def forward(self, inputs, adj):
        """
        inputs: [N, D]
        adj: [N, N], dense or sparse

        We always run the graph propagation in fp32 and convert sparse adj to dense
        to avoid the CUDA sparse half/amp path that caused your crashes.
        """
        orig_dtype = inputs.dtype
        device = inputs.device

        x = inputs
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_fp32 = x.float()
        weight_fp32 = self.layer.weight.float()
        bias_fp32 = self.layer.bias.float() if self.layer.bias is not None else None

        support = F.linear(x_fp32, weight_fp32, bias=None)  # [N, D_out]

        if adj.is_sparse:
            adj = adj.to_dense()
        adj_fp32 = adj.float().to(device)

        outputs = adj_fp32 @ support
        if bias_fp32 is not None:
            outputs = outputs + bias_fp32

        if self.relu:
            outputs = F.relu(outputs, inplace=False)

        return outputs.to(orig_dtype)


class PromptGraphRefiner(nn.Module):
    """
    Lightweight GCN refiner over [attr, obj, pair] prompt features.

    API kept compatible with the current CAMS code:
        PromptGraphRefiner(dim, hidden_dim=None, dropout=0.1, residual_alpha=0.5)
        PromptGraphRefiner.build_adj(num_attrs, num_objs, pair_list, attr2idx, obj2idx)
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.1, residual_alpha=0.5):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(dim)
        self.residual_alpha = float(residual_alpha)
        self.conv1 = GraphConv(dim, hidden_dim, dropout=dropout, relu=True)
        self.conv2 = GraphConv(hidden_dim, dim, dropout=0.0, relu=False)

    @staticmethod
    def build_adj(num_attrs, num_objs, pair_list, attr2idx, obj2idx):
        """
        Build a dense normalized adjacency over nodes ordered as:
            [all attrs | all objs | all pairs]

        pair_list can be either:
            - list of (attr_name, obj_name)
            - list of (attr_idx, obj_idx)
            - LongTensor [num_pairs, 2]
        """
        # Normalize pair_list to a CPU LongTensor [num_pairs, 2]
        if torch.is_tensor(pair_list):
            pair_idx = pair_list.long().cpu()
        else:
            if len(pair_list) == 0:
                pair_idx = torch.empty((0, 2), dtype=torch.long)
            else:
                first = pair_list[0]
                # numeric pair tuples/lists
                if isinstance(first[0], (int, np.integer)) and isinstance(first[1], (int, np.integer)):
                    pair_idx = torch.tensor(pair_list, dtype=torch.long)
                else:
                    # string pair tuples
                    pair_idx = torch.tensor(
                        [(attr2idx[attr], obj2idx[obj]) for attr, obj in pair_list],
                        dtype=torch.long,
                    )

        num_pairs = int(pair_idx.size(0))
        num_nodes = int(num_attrs + num_objs + num_pairs)

        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

        # self loops
        adj.fill_diagonal_(1.0)

        # pair <-> attr / obj and attr <-> obj if observed in a pair
        for i in range(num_pairs):
            a = int(pair_idx[i, 0].item())
            o = int(pair_idx[i, 1].item())
            attr_node = a
            obj_node = num_attrs + o
            pair_node = num_attrs + num_objs + i

            # observed attr-object co-occurrence
            adj[attr_node, obj_node] = 1.0
            adj[obj_node, attr_node] = 1.0

            # pair to primitives
            adj[pair_node, attr_node] = 1.0
            adj[attr_node, pair_node] = 1.0
            adj[pair_node, obj_node] = 1.0
            adj[obj_node, pair_node] = 1.0

        # symmetric degree normalization: D^{-1/2} A D^{-1/2}
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

        return adj.contiguous()

    def forward(self, x, adj):
        refined = self.conv1(x, adj)
        refined = self.conv2(refined, adj)
        out = (1.0 - self.residual_alpha) * x + self.residual_alpha * refined
        return F.normalize(out, dim=-1)
