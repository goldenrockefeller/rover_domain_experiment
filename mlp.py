import torch
import copy


from rockefeg.policyopt.map import BaseDifferentiableMap
from rockefeg.cyutil.array import DoubleArray
import numpy as np

# scripted_module = torch.jit.script(MyModule(2, 3))

class TorchMlp(BaseDifferentiableMap):
    def __init__(self, n_in_dims, n_hidden_neurons, n_out_dims):
        self.n_in_dims = n_in_dims
        self.n_hidden_neurons = n_hidden_neurons
        self.n_out_dims = n_out_dims
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in_dims, n_hidden_neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_neurons, n_out_dims))
        self.model.to(dtype = torch.double)

    def copy(self, copy_obj = None):
        new_mlp = (
            TorchMlp(self.n_in_dims, self.n_hidden_neurons, self.n_out_dims) )
        new_mlp.model = copy.deepcopy(self.model)
        new_mlp.model.zero_grad()
        return new_mlp

    def n_parameters(self):
        return sum(p.numel() for p in model.parameters())

    def parameters(self):
        parameters = torch.cat([param.view(-1) for param in self.model.parameters()])
        return DoubleArray(parameters.detach().numpy())

    def set_parameters(self, parameters):
        np_parameters = np.asarray(parameters.view)
        offset = 0

        for param in self.model.parameters():
            param_view = param.view(-1)

            param.view(-1).copy_(
                torch.from_numpy(
                    np_parameters[offset : offset + len(param_view)] ))

            offset += len(param_view)

    def eval(self, input):
        np_input = np.asarray(input.view)
        eval = self.model.forward(torch.from_numpy(np_input))
        return DoubleArray(eval.detach().numpy())

