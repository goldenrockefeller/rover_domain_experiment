import pyximport; pyximport.install()


from goldenrockefeller.policyopt.trial import Trial
from goldenrockefeller.policyopt.multiagent_system import MultiagentSystem
from goldenrockefeller.policyopt.evolution import DefaultEvolvingSystem, DefaultPhenotype
from goldenrockefeller.policyopt.neural_network import TanhLayer, ReluLinear

from goldenrockefeller.policyopt.function_approximation import DifferentiableFunctionApproximator

from goldenrockefeller.roverdomain.evaluator import DefaultEvaluator
from goldenrockefeller.roverdomain.evaluator import DifferenceEvaluator

from goldenrockefeller.policyopt.map import DifferentiableCriticMap
from goldenrockefeller.policyopt.fitness_critic import FitnessCriticSystem
# from goldenrockefeller.policyopt.value_target import TdLambdaTargetSetter

from goldenrockefeller.cyutil.array import DoubleArray

from cmaes import CmaesSystem


from flat_critic import NFlatNetwork
from flat_critic import FlatFitnessCriticSystem, MonteFlatFitnessCriticSystem
from flat_critic import QFlatFitnessCriticSystem, UqFlatFitnessCriticSystem
# from mlp import TorchMlp

# from rbfn_approximator import RbfnApproximator

# from novelty import EvolvingSystemWithNoveltySearch
# from novelty import PhenotypeWithNoveltyScore

from trial_rover_domain import TrialRoverDomain

from cauchy_phenotype import CauchyPhenotype

from mean_fitness_critic import MeanFitnessCriticSystem, MonteFitnessCriticSystem

import numpy as np

def none(arg_dict):
    pass

def global_reward(arg_dict):
    old_evaluator = arg_dict["trial"].domain.super_domain.evaluator()
    evaluator = DefaultEvaluator()

    evaluator.set_n_req(old_evaluator.n_req())
    evaluator.set_capture_dist(old_evaluator.capture_dist())

def difference_reward(arg_dict):
    old_evaluator = arg_dict["trial"].domain.super_domain.evaluator()
    evaluator = DifferenceEvaluator()

    evaluator.set_n_req(old_evaluator.n_req())
    evaluator.set_capture_dist(old_evaluator.capture_dist())


# def flat_critic_raw(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#
#     agent_systems = multiagent_system.agent_systems()
#
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#
#         fitness_critic_system = (
#             FlatFitnessCriticSystem(
#                 evolving_system, 10, 80 ))
#
#         intermediate_critic = fitness_critic_system.intermediate_critic()
#
#         intermediate_critic.learning_rate = 1e-4
#         intermediate_critic.using_conditioner = False
#         intermediate_critic.using_accelerator = False
#         intermediate_critic.using_grad_disturber = False
#         intermediate_critic.flat_network.leaky_scale = 0.01
#         intermediate_critic.conditioner.time_horizon = 5000.
#         intermediate_critic.accelerator.time_horizon = 5000.
#         intermediate_critic.grad_disturber.disturbance_factor = 0.
#
#         fitness_critic_system.trajectory_buffer().set_capacity(100)
#         fitness_critic_system.experience_target_buffer.set_capacity(5000)
#         fitness_critic_system.n_critic_updates_per_epoch = 5000
#
#         agent_systems[rover_id] = fitness_critic_system


def flat_critic_zero(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system, 10, 80 ))

        approximator = fitness_critic_system.approximator


        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 0.0
        approximator.using_conditioner = False
        approximator.grad_disturbance_factor = 0.0
        approximator.momentum_sustain = 0.
        approximator.conditioner_time_horizon = 1.

\
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system

def flat_critic_reg(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system, 10, 80 ))

        approximator = fitness_critic_system.approximator


        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 1e-5
        approximator.using_conditioner = False
        approximator.grad_disturbance_factor = 0.0
        approximator.momentum_sustain = 0.
        approximator.conditioner_time_horizon = 1.

\
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system

def flat_critic_fierce(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system, 10, 80 ))

        approximator = fitness_critic_system.approximator


        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 0.1
        approximator.using_conditioner = True
        approximator.grad_disturbance_factor = 0.05
        approximator.momentum_sustain = 0.99
        approximator.conditioner_time_horizon = 5000.


        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system

def monte_flat_critic_reg(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            MonteFlatFitnessCriticSystem(
                evolving_system, 10, 80 ))

        approximator = fitness_critic_system.approximator


        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 1e-5
        approximator.using_conditioner = False
        approximator.grad_disturbance_factor = 0.0
        approximator.momentum_sustain = 0.
        approximator.conditioner_time_horizon = 1.


        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system

def q_flat_critic_reg(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            QFlatFitnessCriticSystem(
                evolving_system, 10, 80 ))

        approximator = fitness_critic_system.approximator


        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 1e-5
        approximator.using_conditioner = False
        approximator.grad_disturbance_factor = 0.0
        approximator.momentum_sustain = 0.
        approximator.conditioner_time_horizon = 1.


        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system

def uq_flat_critic_reg(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            UqFlatFitnessCriticSystem(
                evolving_system, 8, 2, 80 ))



        approximator = fitness_critic_system.q_approximator
        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 1e-5
        approximator.using_conditioner = False
        approximator.grad_disturbance_factor = 0.0
        approximator.momentum_sustain = 0.
        approximator.conditioner_time_horizon = 1.

        approximator = fitness_critic_system.u_approximator
        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 1e-5
        approximator.using_conditioner = False
        approximator.grad_disturbance_factor = 0.0
        approximator.momentum_sustain = 0.
        approximator.conditioner_time_horizon = 1.


        fitness_critic_system.q_experience_target_buffer.set_capacity(5000)
        fitness_critic_system.u_experience_target_buffer.set_capacity(5000)


        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system


def monte_flat_critic_fierce(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            MonteFlatFitnessCriticSystem(
                evolving_system, 10, 80 ))

        approximator = fitness_critic_system.approximator


        approximator.flat_network.leaky_scale = 0.5

        approximator.learning_rate = 0.1
        approximator.using_conditioner = Trues
        approximator.grad_disturbance_factor = 0.05
        approximator.momentum_sustain = 0.99
        approximator.conditioner_time_horizon = 5000.


        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        fitness_critic_system.n_critic_updates_per_epoch = 5000

        agent_systems[rover_id] = fitness_critic_system

# def flat_critic_fierce(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#
#     agent_systems = multiagent_system.agent_systems()
#
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#
#         fitness_critic_system = (
#             FlatFitnessCriticSystem(
#                 evolving_system, 10, 80 ))
#
#         approximator = fitness_critic_system.approximator
#
#         approximator.learning_rate = 0.001
#         approximator.using_conditioner = True
#         approximator.using_accelerator = False
#         approximator.using_grad_disturber = False
#         approximator.flat_network.leaky_scale = 0.5
#         approximator.conditioner.time_horizon = 50.
#         approximator.accelerator.time_horizon = 50.
#         approximator.grad_disturber.disturbance_factor = 0.
#
#         fitness_critic_system.experience_target_buffer.set_capacity(5000)
#         fitness_critic_system.n_critic_updates_per_epoch = 5000
#
#         agent_systems[rover_id] = fitness_critic_system

def mean_fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        map = ReluLinear(10, 80, 1)
        map.leaky_scale = 0.01
        critic = DifferentiableCriticMap(map)

        intermediate_critic = DifferentiableFunctionApproximator(critic)

        fitness_critic_system = (
            MeanFitnessCriticSystem(
                evolving_system,
                intermediate_critic))

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000


        intermediate_critic.set_learning_rate(1e-5)

        agent_systems[rover_id] = fitness_critic_system


def mean_fitness_critic_fixed(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        map = ReluLinear(10, 80, 1, True)
        map.leaky_scale = 0.01
        critic = DifferentiableCriticMap(map)

        intermediate_critic = DifferentiableFunctionApproximator(critic)

        fitness_critic_system = (
            MeanFitnessCriticSystem(
                evolving_system,
                intermediate_critic))

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000


        intermediate_critic.set_learning_rate(1e-5)

        agent_systems[rover_id] = fitness_critic_system

def mean_fitness_critic_flat(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        map = NFlatNetwork(10, 80)
        critic = DifferentiableCriticMap(map)

        intermediate_critic = DifferentiableFunctionApproximator(critic)

        fitness_critic_system = (
            MeanFitnessCriticSystem(
                evolving_system,
                intermediate_critic))

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000


        intermediate_critic.set_learning_rate(1e-5)

        # intermediate_critic.set_learning_rate(1e-3)

        agent_systems[rover_id] = fitness_critic_system

def monte_fitness_critic_flat(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        map = NFlatNetwork(10, 80)
        critic = DifferentiableCriticMap(map)

        intermediate_critic = DifferentiableFunctionApproximator(critic)

        fitness_critic_system = (
            MonteFitnessCriticSystem(
                evolving_system,
                intermediate_critic))

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000


        intermediate_critic.set_learning_rate(1e-6)

        agent_systems[rover_id] = fitness_critic_system




