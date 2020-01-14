import torch
from torch import nn
from torch.nn import functional as F

args = {
    'learning_rate': 0.01,
    'epochs': 200,
    'hidden': 128,
    'dropout': 0.5,
    'weight_decay': 5e-4,
    'early_stopping': 10,
    'max_degree': 3
}

device = torch.device('cuda')


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

#
# class GraphConvolution(nn.Module):
#
#     def __init__(self,
#                  input_dim,
#                  output_dim,
#                  num_features_nonzero,
#                  dropout=0.,
#                  bias=False,
#                  activation=F.relu):
#         super(GraphConvolution, self).__init__()
#
#         self.dropout = dropout
#         self.bias = bias
#         self.activation = activation
#         self.num_features_nonzero = num_features_nonzero
#         self.weight = nn.Parameter(torch.randn(input_dim, output_dim)).to(device)
#         self.bias = None
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(output_dim)).to(device)
#
#     def forward(self, inputs):
#         # print('inputs:', inputs)
#         x, support = inputs
#
#         x = F.dropout(x, self.dropout)
#
#         xw = torch.einsum('ijk, kl -> ijl', x, self.weight)
#
#         out = torch.einsum('ijk, ijj -> ijk', xw, support)
#
#         if self.bias is not None:
#             out += self.bias[None, None, :]
#
#         return self.activation(out), support

class GraphConvolution(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 num_features_nonzero,
                 dropout=0.,
                 bias=False,
                 activation=F.relu):
        super(GraphConvolution, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.num_features_nonzero = num_features_nonzero
        self.affine1 = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        x = F.dropout(x, self.dropout)

        xw = self.affine1(x)

        out = torch.einsum('ijk, ijj -> ijk', xw, support)

        return self.activation(out), support


class GCN(nn.Module):

    def __init__(self, input_dim, output_dim, num_features_nonzero):
        super(GCN, self).__init__()

        self.input_dim = input_dim  # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        print('num_features_nonzero:', num_features_nonzero)

        self.layer1 = GraphConvolution(self.input_dim,
                                       args['hidden'],
                                       num_features_nonzero,
                                       activation=F.relu,
                                       dropout=args['dropout'])

        self.layer2 = GraphConvolution(args['hidden'],
                                       output_dim,
                                       num_features_nonzero,
                                       activation=F.relu,
                                       dropout=args['dropout'])

        self.layer3 = GraphConvolution(args['hidden'],
                                       output_dim,
                                       num_features_nonzero,
                                       activation=F.relu,
                                       dropout=args['dropout'])



    def forward(self, inputs):

        x, support = inputs

        x, support = self.layer1((x, support))

        x, support = self.layer2((x, support))

        x, support = self.layer3((x, support))

        return x, support
