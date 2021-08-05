import numpy as np
import blank
from rockefeg.cyutil.array import DoubleArray
from rockefeg.policyopt.rbf_network import RbfNetwork


r = RbfNetwork(10,1,1)
r.set_parameters(DoubleArray(1. * np.random.random(r.n_parameters())))
a = DoubleArray(1. * np.random.random(10))

def arr(x):
    return np.asarray(x.view)

# a0 = a.copy(); a0.view[0] += 0.001
# a1 = a.copy(); a1.view[1] += 0.001
# a2 = a.copy(); a2.view[2] += 0.001
# a3 = a.copy(); a3.view[3] += 0.001

parameters = r.parameters()
n_parameters = len(parameters)
delta_r = r.copy()
delta_parameters = parameters.copy()
grad_delta_parameters = np.zeros(n_parameters)
out_grad = np.array([2.])
grad_wrt_parameters = arr(r.grad_wrt_center_locations(a, DoubleArray(out_grad)).copy())

for parameter_id in range(r.n_centers() * r.n_in_dims()):
    delta_parameters.view[parameter_id] += 0.001
    delta_r.set_parameters(delta_parameters)

    grad_delta_parameters[parameter_id] = (
        ((
        np.asarray(delta_r.eval(a).view)
        - np.asarray(r.eval(a).view)
        )*out_grad).sum()
        /0.001)

    delta_parameters.view[parameter_id] -= 0.001
    print(grad_delta_parameters[parameter_id], grad_wrt_parameters[parameter_id])
    # print(grad_delta_parameters[parameter_id], grad_wrt_parameters[parameter_id])

print("----")


def arr(x):
    return np.asarray(x.view)

def activation(x, loc, shape):
    return np.exp(-((loc - x) ** 2 * shape).sum())

def d_activation(x, loc, shape):
    return activation(x, loc, shape) * -2 * (loc-x) * shape
