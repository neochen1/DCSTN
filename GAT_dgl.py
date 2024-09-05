import os

import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
import dgl
# import dgl.nn.pytorch as dglnn
from dgl import DGLError

# from dgl.nn.pytorch.conv import GATConv
# # from gatv2 import GATv2
# from dgl.utils.internal import expand_as_pair
# from dgl.nn.functional import edge_softmax


# class GraphAttention(nn.Module):
#     def __init__(self,
#                  node_num,
#                  in_dim,
#                  edge_f_dim,
#                  out_dim,
#                  num_heads,
#                  feat_drop,
#                  attn_drop,
#                  alpha,
#                  edge_feature_attn,
#                  residual):
#         super(GraphAttention, self).__init__()
#
#         self.node_num = node_num
#         self.num_heads = num_heads
#
#
#         # self.edge_feature_attn = edge_feature_attn
#         self.node_feature_attn = edge_feature_attn
#
#
#
#         self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
#         if feat_drop:
#             self.feat_drop = nn.Dropout(feat_drop)
#         else:
#             self.feat_drop = lambda x: x
#         if attn_drop:
#             self.attn_drop = nn.Dropout(attn_drop)
#         else:
#             self.attn_drop = lambda x: x
#
#         self.leaky_relu = nn.LeakyReLU(alpha)
#         self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
#         self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
#         self.attn_e = nn.Sequential(
#             nn.Linear(edge_f_dim, out_dim),
#             self.leaky_relu,
#             nn.Linear(out_dim, num_heads)
#         )
#         # nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
#         # nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
#         # nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
#         self.residual = residual
#         self.batch_norm = nn.BatchNorm1d(num_heads * out_dim)
#
#         if residual:
#             if in_dim != out_dim:
#                 self.res_fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
#                 nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)
#             else:
#                 self.res_fc = None
#
#     def forward(self, g, inputs, last=False):
#         # prepare
#         print("inputs in", inputs)
#         h = self.feat_drop(inputs)  # NxD
#         print("inputs drop", h)
#
#         ft = self.fc(h)
#         print("ft step 00", ft)
#
#         ft = ft.reshape((h.shape[0], self.num_heads, -1))  # NxHxD'
#         print("ft step 01", ft)
#
#
#         head_ft = ft.transpose(0, 1)  # HxNxD'
#         a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)  # NxHx1
#         a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)  # NxHx1
#         g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
#
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX1", ret_ft_XX)
#
#         # # 1. compute edge attention
#         # g.apply_edges(self.edge_attention)
#         g.apply_nodes(self.node_attention)
#
#
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX2", ret_ft_XX)
#
#         # # 2. compute softmax in two parts: exp(x - max(x)) and sum(exp(x - max(x)))
#         # self.edge_softmax(g)
#         self.node_softmax(g)
#
#
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX3", ret_ft_XX)
#
#
#         # # 2. compute the aggregated node features scaled by the dropped,
#         # # unnormalized attention values.
#
#         g_mul = fn.src_mul_edge('ft', 'a_drop', 'ft')
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX4", ret_ft_XX)
#
#
#         g_sum = fn.sum('ft', 'ft')
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX5", ret_ft_XX)
#
#         g.update_all(g_mul, g_sum)
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX6", ret_ft_XX)
#
#
#         # g.update_all(fn.src_mul_edge('ft', 'a_drop', 'ft'), fn.sum('ft', 'ft'))
#
#         ret_ft = g.ndata['ft']
#         ret_z = g.ndata['z']
#         print("ret_ft",ret_ft)
#         print("ret_z",ret_z)
#
#
#         # 3. apply normalizer
#         ret = g.ndata['ft'] / g.ndata['z']
#         print("ret step 3", ret)
#
#         # 4. residual:
#         if self.residual:
#             if self.res_fc is not None:
#                 resval = self.res_fc(h).reshape((h.shape[0], self.num_heads, -1))
#             else:
#                 resval = torch.unsqueeze(h, 1)
#             ret = ret + resval
#             print("ret step 4", ret)
#
#         # 5. batch norm:
#         if last == False:
#             ret = ret.flatten(1)
#             # ret2 = self.batch_norm(ret.flatten(1))     #cause nan
#             print("ret step5", ret)
#             # print("ret2 step5", ret2)
#
#             # print("ret length", len(ret))
#             # print("ret", ret.size())         #(sample_num*len) * hideen_length
#
#         else:
#             ret = ret.mean(1)
#             # print("last layer ret length", len(ret))
#             # print("last layer ret", ret.size())
#
#             """graph embedding"""
#             node_num = self.node_num
#             samples = len(ret)//node_num    # force to be int
#
#             # for i in range(samples):
#             #     sample = ret[i * node_num:(i + 1) * node_num, :].mean(0)
#             #     print("sample", len(sample))
#             #     print("sample", sample.size())
#             ret = torch.stack([ret[i*node_num:(i+1)*node_num,:].mean(0) for i in range(samples)], 0)
#             # print("graph ret", ret.size())
#             # print("graph ret", ret)
#
#         return ret
#
#         # if self.last:
#         #     return dgl.mean_nodes(g, 'ft')
#         # else:
#         #     return g.ndata.pop('ft')
#
#     def node_attention(self, nodes):
#         a = nodes.src['a1'] + nodes.dst['a2']
#         if self.node_feature_attn:
#             e_attn = self.attn_e(nodes.data['x']).unsqueeze(-1)
#             a = a + e_attn
#         a = self.leaky_relu(a)
#
#         return {'a': a}
#
#     def edge_attention(self, edges):
#         # an edge UDF to compute unnormalized attention values from src and dst
#         a = edges.src['a1'] + edges.dst['a2']
#         if self.edge_feature_attn:
#             e_attn = self.attn_e(edges.data['x']).unsqueeze(-1)
#             a = a + e_attn
#         a = self.leaky_relu(a)
#
#         return {'a': a}
#
#     def node_softmax(self, g):
#         # compute the max
#         g.update_all(fn.copy_src('a', 'a'), fn.max('a', 'a_max'))
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX21", ret_ft_XX)
#
#         # minus the max and exp
#         g.apply_nodes(lambda nodes: {'a': torch.exp(nodes.data['a'] - nodes.dst['a_max'])})
#         # compute dropout
#         g.apply_nodes(lambda nodes: {'a_drop': self.attn_drop(nodes.data['a'])})
#         # compute normalizer
#         g.update_all(fn.copy_src('a', 'a'), fn.sum('a', 'z'))
#
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX22", ret_ft_XX)
#
#
#     def edge_softmax(self, g):
#         # compute the max
#         g.update_all(fn.copy_edge('a', 'a'), fn.max('a', 'a_max'))
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX21", ret_ft_XX)
#
#         # minus the max and exp
#         g.apply_edges(lambda edges: {'a': torch.exp(edges.data['a'] - edges.dst['a_max'])})
#         # compute dropout
#         g.apply_edges(lambda edges: {'a_drop': self.attn_drop(edges.data['a'])})
#         # compute normalizer
#         g.update_all(fn.copy_edge('a', 'a'), fn.sum('a', 'z'))
#
#         ret_ft_XX = g.ndata['ft']
#         print("ret_ft_XX22", ret_ft_XX)







# class GATConv(nn.Module):
#     def __init__(self,
#                  node_num,
#                  in_feats,
#                  out_feats,
#                  num_heads,
#                  feat_drop=0.,
#                  attn_drop=0.,
#                  negative_slope=0.2,
#                  residual=False,
#                  activation=None):
#         super(GATConv, self).__init__()
#
#         self.node_num = node_num
#
#         self._num_heads = num_heads
#         # expand_as_pair 函数可以返回一个二维元组。
#         self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
#         self._out_feats = out_feats
#
#         if isinstance(in_feats, tuple):
#             self.fc_src = nn.Linear(
#                 self._in_src_feats, out_feats * num_heads, bias=False)
#             self.fc_dst = nn.Linear(
#                 self._in_dst_feats, out_feats * num_heads, bias=False)
#         else:
#             #全连接层
#             self.fc = nn.Linear(
#                 self._in_src_feats, out_feats * num_heads, bias=False)
#
#         """
#         论文里的h_i和h_j是先concat再通过全连接层做点积，代码里是先全连接层做点积再相加
#         代码将公式里的a分解为[attn_l || attn_r]
#         也即a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
#         结果是一样的，但代码实现方式效率更高
#         """
#         self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
#         self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
#         #对所有元素中每个元素按概率更改为0
#         self.feat_drop = nn.Dropout(feat_drop)
#         #对所有元素中每个元素按概率更改为0
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         if residual:
#             if self._in_dst_feats != out_feats:
#                 self.res_fc = nn.Linear(
#                     self._in_dst_feats, num_heads * out_feats, bias=False)
#             else:
#                 self.res_fc = Identity()
#         else:
#             self.register_buffer('res_fc', None)
#         self.reset_parameters()
#         self.activation = activation
#
#     #初始化参数
#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         gain = nn.init.calculate_gain('relu')
#         if hasattr(self, 'fc'):
#             nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         else: # bipartite graph neural networks
#             nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
#             nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_l, gain=gain)
#         nn.init.xavier_normal_(self.attn_r, gain=gain)
#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
#
#     #前向传播
#     def forward(self, graph, feat, last=False):
#         #graph.local_scope()是为了避免意外覆盖现有的特征数据
#         with graph.local_scope():
#             if isinstance(feat, tuple):
#                 h_src = self.feat_drop(feat[0])
#                 h_dst = self.feat_drop(feat[1])
#                 feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
#                 feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
#             else:
#                 h_src = h_dst = self.feat_drop(feat)
#                 #Wh_i(src)、Wh_j(dst)在各head的特征组成的矩阵: (1, num_heads, out_feats)
#                 feat_src = feat_dst = self.fc(h_src).view(
#                     -1, self._num_heads, self._out_feats)
#
#             #Wh_i * a_l， 并将各head得到的注意力系数aWh_i相加
#             el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
#             #Wh_j * a_r， 并将各head得到的注意力系数aWh_j相加
#             er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
#             graph.srcdata.update({'ft': feat_src, 'el': el})
#             graph.dstdata.update({'er': er})
#             #(a^T [Wh_i || Wh_j] = )a_l Wh_i + a_r Wh_j
#             graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
#             #e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
#             e = self.leaky_relu(graph.edata.pop('e'))
#             #\alpha_i,j = softmax e_ij
#             graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
#             #'m' = \alpha * Wh_j
#             #feature = \sum(\alpha_i,j * Wh_j)
#             graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
#                              fn.sum('m', 'ft'))
#             rst = graph.dstdata['ft']
#
#             # 残差
#             if self.res_fc is not None:
#                 resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
#                 rst = rst + resval
#
#             # 激活函数
#             if self.activation:
#                 rst = self.activation(rst)
#
#
#             # 5. batch norm:
#             if last == False:
#                 rst = rst.flatten(1)
#                 # ret2 = self.batch_norm(ret.flatten(1))     #cause nan
#                 print("ret step5", rst)
#                 # print("ret2 step5", ret2)
#
#                 # print("ret length", len(ret))
#                 # print("ret", ret.size())         #(sample_num*len) * hideen_length
#
#             else:
#                 rst = rst.mean(1)
#                 # print("last layer ret length", len(ret))
#                 # print("last layer ret", ret.size())
#
#                 """graph embedding"""
#                 node_num = self.node_num
#                 samples = len(rst) // node_num  # force to be int
#
#                 # for i in range(samples):
#                 #     sample = ret[i * node_num:(i + 1) * node_num, :].mean(0)
#                 #     print("sample", len(sample))
#                 #     print("sample", sample.size())
#                 rst = torch.stack([rst[i * node_num:(i + 1) * node_num, :].mean(0) for i in range(samples)], 0)
#                 # print("graph ret", ret.size())
#                 # print("graph ret", ret)
#
#             return rst





from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch.nn import Identity
import dgl.nn.pytorch as dglnn

class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], self._num_heads, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(1, self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst


class Replace_GAT(nn.Module):
    def __init__(self,
                 node_num,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation
                 ):
        super(Replace_GAT, self).__init__()

        self.node_num = node_num

        self.ff = GATConv(in_feats=in_dim, out_feats=num_classes, num_heads=heads)

    def forward(self, g, h):
        h = torch.mean(self.ff(g, h), dim=1)

        node_num = self.node_num
        samples = len(h) // node_num  # force to be int

        # for i in range(samples):
        #     sample = ret[i * node_num:(i + 1) * node_num, :].mean(0)
        #     print("sample", len(sample))
        #     print("sample", sample.size())
        h = torch.stack([h[i * node_num:(i + 1) * node_num, :].mean(0) for i in range(samples)], 0)
        # print("graph logits", h.size())

        return h


# class GATModel(nn.Module):
#     def __init__(self,
#                  node_num,
#                  num_layers,
#                  in_dim,
#                  num_hidden,
#                  num_classes,
#                  heads,
#                  activation):
#         super(GATModel, self).__init__()
#         # self.g = g
#
#         self.node_num = node_num
#
#         self.num_layers = num_layers
#         self.gat_layers = nn.ModuleList()
#         self.activation = activation
#
#         self.activation = nn.LeakyReLU()
#
#         # input projection (no residual)
#         # self.gat_layers.append(GATConv(
#         #     in_dim, num_hidden, heads[0],
#         #     feat_drop, attn_drop, negative_slope, False, self.activation))
#         self.gat_layers.append(GATConv(in_dim, num_hidden, heads, activation=self.activation))
#
#
#         # hidden layers
#         for l in range(1, num_layers):
#             # due to multi-head, the in_dim = num_hidden * num_heads
#             self.gat_layers.append(GATConv(
#                 num_hidden * heads, num_hidden, heads, activation=self.activation))
#
#         # output projection
#         self.gat_layers.append(GATConv(
#             num_hidden * heads, num_classes, heads, activation=None))
#
#     def forward(self, g, inputs):
#         # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#         # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#         h = inputs
#
#         for l in range(self.num_layers):
#             print("h device", h.device)
#             print("g device", g.device)
#             # print("net", [layer.device for layer in self.gat_layers])
#
#             h = self.gat_layers[l](g, h)
#             h = nn.reshape(h, (h.shape[0], -1))
#
#         # output projection
#         logits = nn.reduce_mean(self.gat_layers[-1](g, h), axis=1)
#
#         return logits

class GATModel(nn.Module):
    def __init__(self,
                 node_num,
                 num_layers,
                 in_feats,
                 num_hidden,
                 out_feats,
                 num_heads,
                 activation):
        super(GATModel, self).__init__()

        self.node_num = node_num
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()


        # self.activation = activation

        self.activation = nn.LeakyReLU()

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_feats, out_feats, num_heads))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
            out_feats * num_heads, out_feats, num_heads))
        # output projection
        self.gat_layers.append(GATConv(
            out_feats * num_heads, out_feats, num_heads))

    def forward(self, g, h):
        # h = g.ndata['x']

        for l in range(self.num_layers):

            h = self.gat_layers[l](g, h).flatten(1)
            # print("h step 1", h)

            h = self.activation(h)


        # output projection
        # logits = self.gat_layers[-1](g, h)
        logits = torch.mean(self.gat_layers[-1](g, h), dim=1)

        # print("logits", logits.size())

        node_num = self.node_num
        samples = len(logits) // node_num  # force to be int

        # for i in range(samples):
        #     sample = ret[i * node_num:(i + 1) * node_num, :].mean(0)
        #     print("sample", len(sample))
        #     print("sample", sample.size())
        logits = torch.stack([logits[i * node_num:(i + 1) * node_num, :].mean(0) for i in range(samples)], 0)
        # print("graph logits", logits.size())
        # print("graph logits", logits)

        return logits