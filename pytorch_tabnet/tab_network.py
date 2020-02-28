import torch
from torch.nn import Linear, BatchNorm1d, ReLU
import numpy as np
from pytorch_tabnet import sparsemax


def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim+output_dim)/np.sqrt(4*input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


def initialize_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim+output_dim)/np.sqrt(input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return


class GBN(torch.nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(x.shape[0] // self.virtual_batch_size +
                         ((x.shape[0] % self.virtual_batch_size) > 0))
        res = torch.Tensor([]).to(x.device)
        for x_ in chunks:
            y = self.bn(x_)
            res = torch.cat([res, y], dim=0)

        return res


class TabNetNoEmbeddings(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 n_d=8, n_a=8,
                 n_steps=3, gamma=1.3,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02):
        """
        Defines main part of the TabNet network without the embedding layers.

        Parameters
        ----------
        - input_dim : int
            Number of features
        - output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        - n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        - n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        - n_steps: int
            Number of sucessive steps in the newtork (usually betwenn 3 and 10)
        - gamma : float
            Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        - n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        - n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        - epsilon: float
            Avoid log(0), this should be kept very low
        """
        super(TabNetNoEmbeddings, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        if self.n_shared > 0:
            shared_feat_transform = torch.nn.ModuleList()
            for i in range(self.n_shared):
                if i == 0:
                    shared_feat_transform.append(Linear(self.input_dim,
                                                        2*(n_d + n_a),
                                                        bias=False))
                else:
                    shared_feat_transform.append(Linear(n_d + n_a, 2*(n_d + n_a), bias=False))

        else:
            shared_feat_transform = None

        self.initial_splitter = FeatTransformer(self.input_dim, n_d+n_a, shared_feat_transform,
                                                n_glu_independent=self.n_independent,
                                                virtual_batch_size=self.virtual_batch_size,
                                                momentum=momentum)

        self.feat_transformers = torch.nn.ModuleList()
        self.att_transformers = torch.nn.ModuleList()

        for step in range(n_steps):
            transformer = FeatTransformer(self.input_dim, n_d+n_a, shared_feat_transform,
                                          n_glu_independent=self.n_independent,
                                          virtual_batch_size=self.virtual_batch_size,
                                          momentum=momentum)
            attention = AttentiveTransformer(n_a, self.input_dim,
                                             virtual_batch_size=self.virtual_batch_size,
                                             momentum=momentum)
            self.feat_transformers.append(transformer)
            self.att_transformers.append(attention)

        self.final_mapping = Linear(n_d, output_dim, bias=False)
        initialize_non_glu(self.final_mapping, n_d, output_dim)

    def forward(self, x):
        res = 0

        prior = torch.ones(x.shape).to(x.device)
        M_explain = torch.zeros(x.shape).to(x.device)
        M_loss = 0
        att = self.initial_splitter(x)[:, self.n_d:]
        masks = {}

        for step in range(self.n_steps):
            M = self.att_transformers[step](prior, att)
            masks[step] = M
            M_loss += torch.mean(torch.sum(torch.mul(M, torch.log(M+self.epsilon)),
                                           dim=1)) / (self.n_steps)
            # update prior
            prior = torch.mul(self.gamma - M, prior)
            # output
            masked_x = torch.mul(M, x)
            out = self.feat_transformers[step](masked_x)
            d = ReLU()(out[:, :self.n_d])
            res = torch.add(res, d)
            # explain
            step_importance = torch.sum(d, dim=1)
            M_explain += torch.mul(M, step_importance.unsqueeze(dim=1))
            # update attention
            att = out[:, self.n_d:]

        res = self.final_mapping(res)
        return res, M_loss, M_explain, masks


class TabNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8,
                 n_steps=3, gamma=1.3, cat_idxs=[], cat_dims=[], cat_emb_dim=1,
                 n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02, device_name='auto'):
        """
        Defines TabNet network

        Parameters
        ----------
        - input_dim : int
            Initial number of features
        - output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        - n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        - n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        - n_steps: int
            Number of sucessive steps in the newtork (usually betwenn 3 and 10)
        - gamma : float
            Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
        - cat_idxs : list of int
            Index of each categorical column in the dataset
        - cat_dims : list of int
            Number of categories in each categorical column
        - cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        - n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        - n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        - epsilon: float
            Avoid log(0), this should be kept very low
        """
        super(TabNet, self).__init__()
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size

        if type(cat_emb_dim) == int:
            self.cat_emb_dims = [cat_emb_dim]*len(self.cat_idxs)
        else:
            # check that all embeddings are provided
            assert(len(cat_emb_dim) == len(cat_dims))
            self.cat_emb_dims = cat_emb_dim
        self.embeddings = torch.nn.ModuleList()
        for cat_dim, emb_dim in zip(self.cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))

        # record continuous indices
        self.continuous_idx = torch.ones(self.input_dim, dtype=torch.bool)
        self.continuous_idx[self.cat_idxs] = 0

        if isinstance(cat_emb_dim, int):
            self.post_embed_dim = self.input_dim + (cat_emb_dim - 1)*len(self.cat_idxs)
        else:
            self.post_embed_dim = self.input_dim + np.sum(cat_emb_dim) - len(cat_emb_dim)
        self.post_embed_dim = np.int(self.post_embed_dim)
        self.tabnet = TabNetNoEmbeddings(self.post_embed_dim, output_dim, n_d, n_a, n_steps,
                                         gamma, n_independent, n_shared, epsilon,
                                         virtual_batch_size, momentum)
        self.initial_bn = BatchNorm1d(self.post_embed_dim, momentum=0.01)

        # Defining device
        if device_name == 'auto':
            if torch.cuda.is_available():
                device_name = 'cuda'
            else:
                device_name = 'cpu'
        self.device = torch.device(device_name)
        self.to(self.device)

    def apply_embeddings(self, x):
        """Apply embdeddings to raw inputs"""
        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()))
                cat_feat_counter += 1
        post_embeddings = torch.cat(cols, dim=1).float()
        return post_embeddings

    def forward(self, x):
        x = self.apply_embeddings(x)
        x = self.initial_bn(x)
        return self.tabnet(x)


class AttentiveTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        """
        Initialize an attention transformer.

        Parameters
        ----------
        - input_dim : int
            Input size
        - output_dim : int
            Outpu_size
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """
        super(AttentiveTransformer, self).__init__()
        self.fc = Linear(input_dim, output_dim, bias=False)
        initialize_non_glu(self.fc, input_dim, output_dim)
        self.bn = GBN(output_dim, virtual_batch_size=virtual_batch_size,
                      momentum=momentum)

        # Sparsemax
        self.sp_max = sparsemax.Sparsemax(dim=-1)
        # Entmax
        # self.sp_max = sparsemax.Entmax15(dim=-1)

    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        x = torch.mul(x, priors)
        x = self.sp_max(x)
        return x


class FeatTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, shared_layers, n_glu_independent,
                 virtual_batch_size=128, momentum=0.02):
        super(FeatTransformer, self).__init__()
        """
        Initialize a feature transformer.

        Parameters
        ----------
        - input_dim : int
            Input size
        - output_dim : int
            Outpu_size
        - n_glu_independant
        - shared_blocks : torch.nn.ModuleList
            The shared block that should be common to every step
        - momentum : float
            Float value between 0 and 1 which will be used for momentum in batch norm
        """

        params = {
            'n_glu': n_glu_independent,
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum
        }

        if shared_layers is None:
            self.shared = None
            self.specifics = GLU_Block(input_dim, output_dim,
                                       first=True,
                                       **params)
        else:
            self.shared = GLU_Block(input_dim, output_dim,
                                    first=True,
                                    shared_layers=shared_layers,
                                    n_glu=len(shared_layers),
                                    virtual_batch_size=virtual_batch_size,
                                    momentum=momentum)
            self.specifics = GLU_Block(output_dim, output_dim,
                                       **params)

    def forward(self, x):
        if self.shared is not None:
            x = self.shared(x)
        x = self.specifics(x)
        return x


class GLU_Block(torch.nn.Module):
    """
        Independant GLU block, specific to each step
    """

    def __init__(self, input_dim, output_dim, n_glu=2, first=False, shared_layers=None,
                 virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.first = first
        self.shared_layers = shared_layers
        self.n_glu = n_glu
        self.glu_layers = torch.nn.ModuleList()

        params = {
            'virtual_batch_size': virtual_batch_size,
            'momentum': momentum
        }

        fc = shared_layers[0] if shared_layers else None
        self.glu_layers.append(GLU_Layer(input_dim, output_dim,
                                         fc=fc,
                                         **params))
        for glu_id in range(1, self.n_glu):
            fc = shared_layers[glu_id] if shared_layers else None
            self.glu_layers.append(GLU_Layer(output_dim, output_dim,
                                             fc=fc,
                                             **params))

    def forward(self, x):
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
        if self.first:  # the first layer of the block has no scale multiplication
            x = self.glu_layers[0](x)
            layers_left = range(1, self.n_glu)
        else:
            layers_left = range(self.n_glu)

        for glu_id in layers_left:
            x = torch.add(x, self.glu_layers[glu_id](x))
            x = x*scale
        return x


class GLU_Layer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, fc=None,
                 virtual_batch_size=128, momentum=0.02):
        super(GLU_Layer, self).__init__()

        self.output_dim = output_dim
        if fc:
            self.fc = fc
        else:
            self.fc = Linear(input_dim, 2*output_dim, bias=False)
        initialize_glu(self.fc, input_dim, 2*output_dim)

        self.bn = GBN(2*output_dim, virtual_batch_size=virtual_batch_size,
                      momentum=momentum)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        return out
