import pyximport; pyximport.install()

from flat_critic import *
from goldenrockefeller.policyopt.neural_network import *
from goldenrockefeller.cyutil.array import DoubleArray


r = FlatNetwork(4, 5)

a = DoubleArray(1. * np.random.random(4))

a0 = a.copy(); a0.view[0] += 0.001
a1 = a.copy(); a1.view[1] += 0.001
a2 = a.copy(); a2.view[2] += 0.001
a3 = a.copy(); a3.view[3] += 0.001

parameters = r.parameters()
n_parameters = len(parameters)
delta_r = r.copy()
delta_parameters = parameters.copy()
grad_delta_parameters = np.zeros(n_parameters)
grad_wrt_parameters = np.asarray(r.grad_wrt_parameters(a, 3.4).view).copy()
for parameter_id in range(n_parameters):
    delta_parameters.view[parameter_id] += 0.001
    delta_r.set_parameters(delta_parameters)
    grad_delta_parameters[parameter_id] = (
        (
        np.asarray(delta_r.eval(a))
        - np.asarray(r.eval(a))
        )
        /0.001)
    delta_parameters.view[parameter_id] -= 0.001
    print(grad_delta_parameters[parameter_id], grad_wrt_parameters[parameter_id])
    # print(grad_delta_parameters[parameter_id], grad_wrt_parameters[parameter_id])

print("----")