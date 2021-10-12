import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint

from utils import get_ffnn


MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

class FFNN(torch.nn.Module):

    def __init__(self, input_size, output_size, nn_desc, dropout_rate=0.0,
                 bias=True, residual=False, bn=False, input_tanh=True):
        super().__init__()

        # create feed-forward NN
        in_size = input_size
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=output_size,
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias, bn=bn
        )
        self.input_tanh = input_tanh
        if residual:
            print('use residual network: input_size={}, output_size={}'.format(
                input_size, output_size))
            if input_size <= output_size:
                if output_size % input_size == 0:
                    self.case = 1
                    self.mult = int(output_size / input_size)
                else:
                    raise ValueError('for residual: output_size needs to be '
                                     'multiple of input_size')

            if input_size > output_size:
                if input_size % output_size == 0:
                    self.case = 2
                    self.mult = int(input_size / output_size)
                else:
                    raise ValueError('for residual: input_size needs to be '
                                     'multiple of output_size')
        else:
            self.case = 0
            
    def forward(self, nn_input, mask=None):
        if self.input_tanh:
            out = self.ffnn(torch.tanh(nn_input))
        else:
            out = self.ffnn(nn_input)

        if self.case == 0:
            return out
        elif self.case == 1:
            identity = nn_input.repeat(1, self.mult)
            return identity + out
        elif self.case == 2:
            identity = torch.mean(torch.stack(nn_input.chunk(self.mult, dim=1)),
                                  dim=0)
            return identity + out

class ODEfunc(nn.Module):

    def __init__(self, 
                enc_node_feat, 
                laplacian, 
                one_hot_encoder,
                edge_attr_type,
                num_nodes,
                use_mini,
                use_physics,
                augment_dim,
                mini_nn_desc,
                dropout, 
                time_dependent,
                device):
        super(ODEfunc, self).__init__()
        self.nfe = 0
        self.input_dim=enc_node_feat
        self.laplacian = laplacian
        self.one_hot_encoder = one_hot_encoder
        self.time_dependent = time_dependent
        self.num_nodes = num_nodes
        self.use_mini = use_mini
        self.use_physics = use_physics
        self.device = device
        self.augment_dim = augment_dim
        
        if self.use_mini and self.time_dependent:
            self.mini_net = FFNN(self.input_dim + self.augment_dim + 1, self.input_dim, mini_nn_desc, input_tanh=False)

        elif self.use_mini and not self.time_dependent:
            self.mini_net = FFNN(self.input_dim + self.augment_dim, self.input_dim, mini_nn_desc, input_tanh=False)

        else:
            self.mini_net = None

        if self.use_physics==1:
            # [TODO] for edge attr - wise k <LA>
            _ = torch.rand(size=(edge_attr_type,1), dtype=torch.float32, device=self.device, requires_grad=True)

            # [TODO] for edge wise k - random initialize <SD, NOAA>
            # _ = torch.rand(size=(num_nodes,num_nodes), dtype=torch.float32, device=self.device, requires_grad=True)

            # [TODO] for single scalar k
            # _ = (torch.rand(1, dtype=torch.float32, device=self.device, requires_grad=True))

            self.phy_params = nn.Parameter(_).to(self.device)

        else:
            self.phy_params = None

    def forward(self, t, x):
        if self.augment_dim != 0 and self.use_mini:
            augment = x[:, self.input_dim:]
            x_phy = x[:, :self.input_dim] #228,64
            forward_x = x_phy
        else:
            augment = None
            x_phy = x
            forward_x = x_phy

        if self.use_physics==1:
            # [TODO] for edge attributes <LA>
             _ = self.one_hot_encoder.mm(self.phy_params).reshape(self.num_nodes, self.num_nodes)
             phy_forward = torch.mul(_, self.laplacian).mm(x_phy)

            # [TODO] for edgewise laplacian <SD, NOAA>
            # phy_forward = (self.phy_params*self.laplacian).mm(x_phy)

        else:
            phy_forward = torch.zeros(x.shape, dtype=torch.float32, device=self.device)

        #228,64
        if self.time_dependent:
            t_vec = torch.ones(forward_x.shape[0], 1).to(self.device) * t
            t_and_x = torch.cat([t_vec, forward_x], 1)
        else:
            t_and_x = forward_x

        # return
        if self.use_mini:
            new_x = phy_forward + self.mini_net(t_and_x)
            if augment!= None:
                return torch.cat([new_x, torch.zeros_like(augment)], 1)
            else:
                return new_x
        else:
            return phy_forward


class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.
    Parameters
    ----------
    device : torch.device
    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.
    tol : float
        Error tolerance.
    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, 
                 device='cuda', 
                 odefunc=None,
                 method='rk4',
                 adjoint=True,
                 tol=1e-3):
        super(ODEBlock, self).__init__()

        self.device = device
        self.odefunc = odefunc
        self.tol = tol
        self.method = method
        self.adjoint = adjoint

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)
        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0
        aug_x = x

        # self.odefunc.phy_features = phy_coeff_mat
        if eval_times is None:  
            integration_time = torch.tensor([0, 1]).float().type_as(aug_x)
        else:
            integration_time = eval_times.type_as(aug_x)

        if self.adjoint:
            if self.method == 'dopri5':
                out = odeint_adjoint(self.odefunc, aug_x, integration_time,
                                    rtol=self.tol, atol=self.tol, method=self.method,
                                    options={'max_num_steps': MAX_NUM_STEPS})
            else:
                out = odeint_adjoint(self.odefunc, aug_x, integration_time,
                                    rtol=self.tol, atol=self.tol, method=self.method)

        else:
            if self.method == 'dopri5':
                out = odeint(self.odefunc, aug_x, integration_time,
                            rtol=self.tol, atol=self.tol, method=self.method,
                            options={'max_num_steps': MAX_NUM_STEPS})
            else:
                out = odeint(self.odefunc, aug_x, integration_time,
                            rtol=self.tol, atol=self.tol, method=self.method)

        phy_params = self.odefunc.phy_params
        
        return out, phy_params
