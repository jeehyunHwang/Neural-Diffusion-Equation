import torch
import torch.nn as nn
from torch_geometric.data import Data
from torchdiffeq import odeint, odeint_adjoint

from utils import get_ffnn

from blocks import ODEfunc, ODEBlock, FFNN

class ODENet(nn.Module):
    """An ODEBlock followed by a Linear layer.
    Parameters
    ----------
    device : torch.device
    data_dim : int
        Dimension of data.
    hidden_dim : int
        Dimension of hidden layers.
    output_dim : int
        Dimension of output after hidden layer. Should be 1 for regression or
        num_classes for classification.
    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.
    time_dependent : bool
        If True adds time as input, making ODE time dependent.
    non_linearity : string
        One of 'relu' and 'softplus'
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, 
                 node_features,
                 enc_node_features,
                 laplacian,
                 one_hot_encoder,
                 edge_attr_type,
                 num_nodes,
                 use_mini=True, 
                 use_physics=True,
                 augment_dim=0,
                 enc_desc=None,
                 dec_desc=None,
                 mini_nn_desc=None,
                 time_dependent=False, 
                 num_processing_steps=10, 
                 use_initial=False,
                 multistep=1,
                 method='rk4',
                 adjoint=True,
                 tol=1e-3, 
                 dropout_rate=0,
                 device='cpu',
                 ):
        super(ODENet, self).__init__()
        self.num_processing_steps = num_processing_steps
        self.multistep = multistep
        self.device=device
        self.odefunc = ODEfunc(enc_node_features, laplacian, one_hot_encoder, edge_attr_type,
                          num_nodes, use_mini, use_physics, augment_dim, mini_nn_desc, dropout_rate, 
                          time_dependent, device)
        self.use_initial = use_initial
        self.augment_dim = augment_dim
        self.node_enc = FFNN(node_features, enc_node_features, enc_desc, dropout_rate, True, False, input_tanh=False)
        self.node_dec = FFNN(enc_node_features, 1, dec_desc, dropout_rate, True, False, input_tanh=False)

        self.odeblock = ODEBlock(device, self.odefunc, method, adjoint, tol)

    def forward(self, x, eval_times, num_processing_steps):
        encoded_inputs = []
        outs = []
        self.num_processing_steps = num_processing_steps

        for step_t in range(self.num_processing_steps):
            outs_tmp = []
            encoded_input = self.node_enc(x[step_t].x)
            time_integral = eval_times[step_t].t

            if self.use_initial:
                outs_tmp += [self.node_dec(encoded_input)]
                for i in range(self.multistep):
                    ode_out, phy_params = self.odeblock(encoded_input, time_integral)
                    encoded_input = ode_out[-1]
                    tmp_out = self.node_dec(ode_out)[-1]
                    outs_tmp += [tmp_out]
                    time_integral = torch.ones(2, dtype=torch.float32, device=self.device) + time_integral
                outs += [outs_tmp]

            else:
                for i in range(self.multistep):
                    ode_out, phy_params = self.odeblock(encoded_input, time_integral)
                    encoded_input = ode_out[-1]
                    tmp_out = self.node_dec(ode_out)[-1]
                    outs_tmp += [tmp_out]
                    time_integral = torch.ones(2, dtype=torch.float32, device=self.device) + time_integral
                outs += [outs_tmp]

        return outs, phy_params