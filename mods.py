import pyximport; pyximport.install()


from goldenrockefeller.policyopt.trial import Trial
from goldenrockefeller.policyopt.multiagent_system import MultiagentSystem
from goldenrockefeller.policyopt.evolution import DefaultEvolvingSystem, DefaultPhenotype
from goldenrockefeller.policyopt.neural_network import TanhLayer, ReluLinear, Rbfn

from goldenrockefeller.policyopt.function_approximation import DifferentiableFunctionApproximator

from goldenrockefeller.roverdomain.evaluator import DefaultEvaluator
from goldenrockefeller.roverdomain.evaluator import DifferenceEvaluator

from goldenrockefeller.policyopt.map import DifferentiableCriticMap
from goldenrockefeller.policyopt.fitness_critic import FitnessCriticSystem
# from goldenrockefeller.policyopt.value_target import TdLambdaTargetSetter

from goldenrockefeller.cyutil.array import DoubleArray

from cmaes import CmaesSystem

from gru_fitness_critic import SumGruCriticSystem, FinalGruCriticSystem, GruApproximator
from gru_fitness_critic import RecordingSumGruCriticSystem, RecordingFinalGruCriticSystem, GruApproximator

from relu_critic import ReluNetworkApproximator, ReluFitnessCriticSystem

from flat_critic import NFlatNetwork
from flat_critic import FlatFitnessCriticSystem
from flat_critic import MonteFlatFitnessCriticSystem
from flat_critic import DiscountFlatFitnessCriticSystem
# from flat_critic import UqFlatNetworkApproximator
# from flat_critic import UFlatNetworkApproximator
# from flat_critic import QFlatNetworkApproximator
from flat_critic import MonteFlatNetworkApproximator
from flat_critic import FlatNetworkApproximator
from flat_critic import DiscountFlatNetworkApproximator

# from mlp import TorchMlp

# from rbfn_approximator import RbfnApproximator

# from novelty import EvolvingSystemWithNoveltySearch
# from novelty import PhenotypeWithNoveltyScore

from trial_rover_domain import TrialRoverDomain

from cauchy_phenotype import CauchyPhenotype

from mean_fitness_critic import MeanFitnessCriticSystem
from mean_fitness_critic import TrajFitnessCriticSystem, RecordingTrajFitnessCriticSystem
from mean_fitness_critic import AlternatingTrajFitnessCriticSystem
from mean_fitness_critic import RecordingMeanFitnessCriticSystem

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

def flat_critic_6_etb(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system, 10, 160 ))

        intermediate_critic = fitness_critic_system.intermediate_critic()

        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-6

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000



        agent_systems[rover_id] = fitness_critic_system

def flat_critic_5_etb(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system, 10, 160 ))

        intermediate_critic = fitness_critic_system.intermediate_critic()

        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-5

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000



        agent_systems[rover_id] = fitness_critic_system


def mean_fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        map = ReluLinear(10, 160, 1)
        map.leaky_scale = 0.1
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

        map = ReluLinear(10, 160, 1, True)
        map.leaky_scale = 0.1
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

        map = NFlatNetwork(10, 160)
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





def monte_flat_critic_4_etb(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            MonteFlatFitnessCriticSystem(
                evolving_system, 10, 160 ))

        intermediate_critic = fitness_critic_system.intermediate_critic()

        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-4

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000



        agent_systems[rover_id] = fitness_critic_system


def monte_flat_critic_5_etb(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            MonteFlatFitnessCriticSystem(
                evolving_system, 10, 160 ))

        intermediate_critic = fitness_critic_system.intermediate_critic()

        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-5

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000


        agent_systems[rover_id] = fitness_critic_system

def discount_flat_critic_5_etb(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            DiscountFlatFitnessCriticSystem(
                evolving_system, 10, 160 ))

        intermediate_critic = fitness_critic_system.intermediate_critic()

        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-5

        fitness_critic_system.trajectory_buffer().set_capacity(1)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000


        agent_systems[rover_id] = fitness_critic_system


def discount_flat_critic_5_etb(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        fitness_critic_system = (
            DiscountFlatFitnessCriticSystem(
                evolving_system, 10, 160 ))

        intermediate_critic = fitness_critic_system.intermediate_critic()

        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-5

        fitness_critic_system.trajectory_buffer().set_capacity(100)
        fitness_critic_system.experience_target_buffer.set_capacity(5000)
        #
        # fitness_critic_system.n_critic_updates_per_epoch = 99

        fitness_critic_system.uses_experience_targets_for_updates = True
        fitness_critic_system.n_critic_updates_per_epoch = 5000



        agent_systems[rover_id] = fitness_critic_system
