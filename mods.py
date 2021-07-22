import pyximport; pyximport.install()


from goldenrockefeller.policyopt.trial import Trial
from goldenrockefeller.policyopt.multiagent_system import MultiagentSystem
from goldenrockefeller.policyopt.evolution import DefaultEvolvingSystem, DefaultPhenotype
from goldenrockefeller.policyopt.neural_network import TanhLayer, ReluLinear, Rbfn

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

from flat_critic import FlatFitnessCriticSystem
from flat_critic import UqFlatNetworkApproximator
from flat_critic import UFlatNetworkApproximator
from flat_critic import QFlatNetworkApproximator
from flat_critic import MonteFlatNetworkApproximator
from flat_critic import FlatNetworkApproximator

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

def flat_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        intermediate_critic = FlatNetworkApproximator(10, 80)
        intermediate_critic.time_horizon = 10.
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-5

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system,
                intermediate_critic))

        fitness_critic_system.trajectory_buffer().set_capacity(500)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(1)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(50)
        # fitness_critic_system.set_critic_update_batch_size(25)

        agent_systems[rover_id] = fitness_critic_system


def monte_flat_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system

    agent_systems = multiagent_system.agent_systems()

    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]

        intermediate_critic = MonteFlatNetworkApproximator(10, 80)
        intermediate_critic.time_horizon = 10
        intermediate_critic.learning_mode = 0
        intermediate_critic.learning_rate = 1e-5

        fitness_critic_system = (
            FlatFitnessCriticSystem(
                evolving_system,
                intermediate_critic))

        fitness_critic_system.trajectory_buffer().set_capacity(500)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(1)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(50)
        # fitness_critic_system.set_critic_update_batch_size(25)

        agent_systems[rover_id] = fitness_critic_system