import pyximport; pyximport.install()

import sys
import os
import errno
import datetime
import itertools
import glob
import datetime as dt
from shutil import copy




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
# from flat_critic import UqFlatNetworkApproximator
# from flat_critic import UFlatNetworkApproximator
# from flat_critic import QFlatNetworkApproximator
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

class Runner:
    def __init__(self, experiment_name, setup_funcs):
        self.setup_funcs = setup_funcs
        self.stat_runs_completed = 0
        self.experiment_name = experiment_name
        setup_names = []
        for setup_func in setup_funcs:
            setup_names.append(setup_func.__name__)
        self.trial_name = "_".join(setup_names)

        # Create experiment folder if not already created.
        try:
            os.makedirs(os.path.join("log", experiment_name))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

        # Save experiment details
        filenames_in_folder = (
            glob.glob("./**.py", recursive = True)
            + glob.glob("./**.pyx", recursive = True)
            + glob.glob("./**.pxd", recursive = True))
        for filename in filenames_in_folder:
            copy(filename, os.path.join("log", experiment_name, filename))

    def trial_args(self):
        arg_dict = {}
        n_req = 4 # HERE
        n_rovers = 15 # HERE
        base_poi_value = 1.
        n_pois = 4
        prints_score = True # HERE

        max_n_epochs = 3000  # HERE
        n_steps = 50

        # Domain Args
        domain = TrialRoverDomain()
        domain.set_n_rovers(n_rovers)
        domain.set_n_pois(n_pois)
        super_domain = domain.super_domain
        super_domain.set_setting_max_n_steps(n_steps)
        super_domain.evaluator().set_n_req(n_req)
        super_domain.evaluator().set_capture_dist(4.)
        super_domain.rover_observations_calculator().set_min_dist(1.)
        super_domain.rover_observations_calculator().set_n_observation_sections(4)

        n_state_dims = (
            2 * super_domain.rover_observations_calculator().n_observation_sections())
        n_action_dims = 2


        # Initializer Args
        domain.poi_init_thickness = 0.
        domain.setup_size = 30. # HERE
        domain.poi_value_init_type = "sequential"
        domain.base_poi_value = base_poi_value

        # Muliagent System
        n_policies_per_agent = 50 #HERE
        n_policy_hidden_neurons = 32

        multiagent_system = MultiagentSystem()
        for rover_id in range(n_rovers):
            # evolving_system = CmaesSystem()
            # evolving_system.slow_down = 10
            # evolving_system.set_max_n_epochs(max_n_epochs)
            # evolving_system.step_size = 1.
            evolving_system = DefaultEvolvingSystem()
            evolving_system.set_max_n_epochs(max_n_epochs)
            multiagent_system.agent_systems().append(evolving_system)
            for policy_id in range(n_policies_per_agent):
                map = ReluLinear(
                    n_state_dims,
                    n_policy_hidden_neurons,
                    n_action_dims)
                map.leaky_scale = 0.0
                map = TanhLayer(map)
                phenotype = (DefaultPhenotype(map))

                phenotype.set_mutation_rate(0.01)
                phenotype.set_mutation_factor(1)

                evolving_system.phenotypes().append(phenotype)

        trial_args = {}
        trial_args["system"] = multiagent_system
        trial_args["domain"] = domain
        trial_args["experiment_name"] = self.experiment_name
        trial_args["mod_name"] = self.trial_name
        trial_args["prints_score"] = prints_score
        trial_args["saves"] = False

        arg_dict["trial"] = Trial(trial_args)
        arg_dict["n_state_action_dims"] = n_state_dims + n_action_dims

        return arg_dict


    def new_run(self):
        datetime_str = (
            dt.datetime.now().isoformat()
            .replace("-", "").replace(':', '').replace(".", "_")
        )

        print(
            "Starting trial.\n"
            f"experiment: {self.experiment_name}\n"
            f"trial: {self.trial_name}\n"
            f"stat run #: {self.stat_runs_completed}\n"
            "datetime: {datetime_str}\n\n"
            .format(**locals()) )
        sys.stdout.flush()

        # initial args here

        args = self.trial_args()


        mod_names = []
        for setup_func in self.setup_funcs:
            setup_func(args)

        trial = args["trial"]
        trial.mod_name = self.trial_name
        trial.run()