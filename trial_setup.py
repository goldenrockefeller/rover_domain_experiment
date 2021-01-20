import pyximport; pyximport.install()

from rockefeg.policyopt.trial import Trial
from rockefeg.policyopt.multiagent_system import MultiagentSystem
from rockefeg.policyopt.evolution import DefaultEvolvingSystem, DefaultPhenotype
from rockefeg.policyopt.neural_network import TanhLayer, ReluLinear, Rbfn

from rockefeg.roverdomain.evaluator import DefaultEvaluator
from rockefeg.roverdomain.evaluator import DifferenceEvaluator

# from rockefeg.policyopt.function_approximation import DifferentiableFunctionApproximator
from rockefeg.policyopt.map import DifferentiableCriticMap
# from rockefeg.policyopt.fitness_critic import FitnessCriticSystem
# from rockefeg.policyopt.value_target import TdLambdaTargetSetter

from rockefeg.cyutil.array import DoubleArray

# from mlp import TorchMlp

# from rbfn_approximator import RbfnApproximator

# from novelty import EvolvingSystemWithNoveltySearch
# from novelty import PhenotypeWithNoveltyScore

from trial_rover_domain import TrialRoverDomain

from cauchy_phenotype import CauchyPhenotype

from rockefeg.policyopt.rbf_network_approximator import RbfNetworkApproximator
from rockefeg.policyopt.rbf_network import RbfNetwork
from mean_fitness_critic import TrajFitnessCriticSystem
from mean_fitness_critic import AlternatingTrajFitnessCriticSystem

import numpy as np

def trial_setup():
    arg_dict = {}
    experiment_name = "AAMAS_2021_nreq_4" 
    n_req = 4
    n_rovers = 15
    base_poi_value = 1.
    n_pois = 4
    prints_score = False 
    
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
    domain.setup_size = 30. 
    domain.poi_value_init_type = "sequential"
    domain.base_poi_value = base_poi_value

    # Muliagent System
    n_policies_per_agent = 50
    n_policy_hidden_neurons = 32
    
    multiagent_system = MultiagentSystem()
    for rover_id in range(n_rovers):
        evolving_system = DefaultEvolvingSystem()
        evolving_system.set_max_n_epochs(max_n_epochs)
        multiagent_system.agent_systems().append(evolving_system)
        for policy_id in range(n_policies_per_agent):
            map = ReluLinear(
                n_state_dims, 
                n_policy_hidden_neurons, 
                n_action_dims)
            map.leaky_scale = 0.1
            map = TanhLayer(map)
            phenotype = (CauchyPhenotype(map)) 
                            
            phenotype.set_mutation_rate(0.01)
            phenotype.set_mutation_factor(1)
            
            evolving_system.phenotypes().append(phenotype)
        
    trial_args = {}
    trial_args["system"] = multiagent_system
    trial_args["domain"] = domain
    trial_args["experiment_name"] = experiment_name
    trial_args["mod_name"] = "experiment"
    trial_args["prints_score"] = prints_score
    trial_args["saves"] = False
    
    arg_dict["trial"] = Trial(trial_args)
    arg_dict["n_state_action_dims"] = n_state_dims + n_action_dims
    
    return arg_dict

# def torch_mlp(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     n_state_dims = 8
#     n_action_dims = 2
#     n_policy_hidden_neurons = 32
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         n_policies_per_agent = len(evolving_system.phenotypes())
#         evolving_system.set_phenotypes([])
#         
#         for policy_id in range(n_policies_per_agent):
#             map = TorchMlp(
#                 n_state_dims, 
#                 n_policy_hidden_neurons, 
#                 n_action_dims)
#             map = TanhLayer(map)
#             phenotype = (DefaultPhenotype(map)) 
#                             
#             phenotype.set_mutation_rate(0.01)
#             phenotype.set_mutation_factor(1)
#             
#             evolving_system.phenotypes().append(phenotype)
            

    

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
    
def rbf_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]
        
        
        n_centers = 16
        rbfn = RbfNetwork(10, n_centers, 1)
        for center_id in range(n_centers):
            rbfn.set_center_shape(center_id, DoubleArray(1 * np.ones(10)))
        
        
        intermediate_critic = RbfNetworkApproximator(rbfn)
        intermediate_critic.set_eval_offset(1.)
        intermediate_critic.set_info_retention_factor(0.99)
        
        fitness_critic_system = (
            TrajFitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(50)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
        
        agent_systems[rover_id] = fitness_critic_system
        
def alt_rbf_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]
        
        
        n_centers = 16
        rbfn = RbfNetwork(10, n_centers, 1)
        for center_id in range(n_centers):
            rbfn.set_center_shape(center_id, DoubleArray(1 * np.ones(10)))
        
        
        intermediate_critic = RbfNetworkApproximator(rbfn)
        intermediate_critic.set_eval_offset(1.)
        intermediate_critic.set_info_retention_factor(0.99)
        
        fitness_critic_system = (
            AlternatingTrajFitnessCriticSystemTrajFitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(50)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
        
        agent_systems[rover_id] = fitness_critic_system
    
# def min_at_inf_transfer(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         
#         n_centers = 160
#         rbfn = Rbfn(10, n_centers, 1)
#         
#         intermediate_critic = RbfnApproximator(rbfn)
#         
#         fitness_critic_system = (
#             TransferFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#         fitness_critic_system.n_epochs_before_switch = 2500
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(50)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
#         
#         locations = np.random.uniform(
#             [0., 0., 0., 0., 0., 0., 0., 0., -0.8, -0.8], 
#             [5., 5., 5., 5., 5., 5., 5., 5., 0.8, 0.8],
#             size = (n_centers, 10))
#         
#         intermediate_critic.set_center_locations(locations)
#         intermediate_critic.set_uncertainties(DoubleArray(1e8 * np.ones(n_centers))) # HERE
#         intermediate_critic.set_values(DoubleArray(1e3 * np.ones(n_centers)))
#         intermediate_critic.set_counters(DoubleArray(1 * np.ones(n_centers)))
#         
#         intermediate_critic.scale_multiplier = 1.
#         intermediate_critic.discount_factor = 0.999
#         intermediate_critic.exploration_incentive_factor = 0. # HERE
#         intermediate_critic.exploration_sampling_factor = 0.
#         intermediate_critic.process_uncertainty_rate = 0.0001
#         intermediate_critic.center_relocalization_rate = 0.
#         intermediate_critic.epsilon = 1e-9
#         intermediate_critic.min_at_inf_factor = 0.1
#                 
#         agent_systems[rover_id] = fitness_critic_system
#             
#     
# def fitness_critic(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         map = ReluLinear(10, 160, 1, True)
#         map.leaky_scale = 0.1
#         critic = DifferentiableCriticMap(map)
#         
#         intermediate_critic = DifferentiableFunctionApproximator(critic)
#         
#         fitness_critic_system = (
#             FitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(500)
#         fitness_critic_system.critic_target_buffer().set_capacity(2500)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(20)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
#         fitness_critic_system.set_critic_update_batch_size(25)
#         
#         intermediate_critic.set_learning_rate(1e-4)
#                 
#         agent_systems[rover_id] = fitness_critic_system
#         
# def mean_fitness_critic(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         map = Rbfn(10, 160, 1)
#         #map.leaky_scale = 0.1
#         critic = DifferentiableCriticMap(map)
#         
#         intermediate_critic = DifferentiableFunctionApproximator(critic)
#         
#         fitness_critic_system = (
#             MeanFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(500)
#         fitness_critic_system.critic_target_buffer().set_capacity(2500)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(20)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
#         fitness_critic_system.set_critic_update_batch_size(25)
#         
#         intermediate_critic.set_learning_rate(1e-2)
#                 
#         agent_systems[rover_id] = fitness_critic_system
#         
#         
# def transfer(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         
#         n_centers = 160
#         rbfn = Rbfn(10, n_centers, 1)
#         
#         intermediate_critic = RbfnApproximator(rbfn)
#         
#         fitness_critic_system = (
#             TransferFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#         fitness_critic_system.n_epochs_before_switch = 2500
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(50)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
#         
#         locations = np.random.uniform(
#             [0., 0., 0., 0., 0., 0., 0., 0., -0.8, -0.8], 
#             [5., 5., 5., 5., 5., 5., 5., 5., 0.8, 0.8],
#             size = (n_centers, 10))
#         
#         intermediate_critic.set_center_locations(locations)
#         intermediate_critic.set_uncertainties(DoubleArray(1e8 * np.ones(n_centers))) # HERE
#         intermediate_critic.set_values(DoubleArray(1e3 * np.ones(n_centers)))
#         intermediate_critic.set_counters(DoubleArray(1 * np.ones(n_centers)))
#         
#         intermediate_critic.scale_multiplier = 1.
#         intermediate_critic.discount_factor = 0.999
#         intermediate_critic.exploration_incentive_factor = 0. # HERE
#         intermediate_critic.exploration_sampling_factor = 0.
#         intermediate_critic.process_uncertainty_rate = 0.0001
#         intermediate_critic.center_relocalization_rate = 0.
#         intermediate_critic.epsilon = 1e-9
#                 
#         agent_systems[rover_id] = fitness_critic_system
#         
# def alternate(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         
#         n_centers = 160
#         rbfn = Rbfn(10, n_centers, 1)
#         
#         intermediate_critic = RbfnApproximator(rbfn)
#         
#         fitness_critic_system = (
#             AlternatingFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(50)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
#         
#         locations = np.random.uniform(
#             [0., 0., 0., 0., 0., 0., 0., 0., -0.8, -0.8], 
#             [5., 5., 5., 5., 5., 5., 5., 5., 0.8, 0.8],
#             size = (n_centers, 10))
#         
#         intermediate_critic.set_center_locations(locations)
#         intermediate_critic.set_uncertainties(DoubleArray(1e8 * np.ones(n_centers))) # HERE
#         intermediate_critic.set_values(DoubleArray(1e3 * np.ones(n_centers)))
#         intermediate_critic.set_counters(DoubleArray(1 * np.ones(n_centers)))
#         
#         intermediate_critic.scale_multiplier = 1.
#         intermediate_critic.discount_factor = 0.999
#         intermediate_critic.exploration_incentive_factor = 0. # HERE
#         intermediate_critic.exploration_sampling_factor = 0.
#         intermediate_critic.process_uncertainty_rate = 0.0001
#         intermediate_critic.center_relocalization_rate = 0.
#         intermediate_critic.epsilon = 1e-9
#                 
#         agent_systems[rover_id] = fitness_critic_system
#         
# def rbfn_fitness_critic(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         
#         n_centers = 160
#         rbfn = Rbfn(10, n_centers, 1)
#         
#         intermediate_critic = RbfnApproximator(rbfn)
#         
#         fitness_critic_system = (
#             MeanFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(50)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
#         
#         locations = np.random.uniform(
#             [0., 0., 0., 0., 0., 0., 0., 0., -0.8, -0.8], 
#             [5., 5., 5., 5., 5., 5., 5., 5., 0.8, 0.8],
#             size = (n_centers, 10))
#         
#         intermediate_critic.set_center_locations(locations)
#         intermediate_critic.set_uncertainties(DoubleArray(1e8 * np.ones(n_centers))) # HERE
#         intermediate_critic.set_values(DoubleArray(1e3 * np.ones(n_centers)))
#         intermediate_critic.set_counters(DoubleArray(1 * np.ones(n_centers)))
#         
#         intermediate_critic.scale_multiplier = 1.
#         intermediate_critic.discount_factor = 0.999
#         intermediate_critic.exploration_incentive_factor = 1. # HERE
#         intermediate_critic.exploration_sampling_factor = 1.
#         intermediate_critic.process_uncertainty_rate = 0.0001
#         intermediate_critic.center_relocalization_rate = 0.
#         intermediate_critic.epsilon = 1e-9
#                 
#         agent_systems[rover_id] = fitness_critic_system
#         
# def rbfn_fitness_critic_noisier(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         
#         n_centers = 160
#         rbfn = Rbfn(10, n_centers, 1)
#         
#         intermediate_critic = RbfnApproximator(rbfn)
#         
#         fitness_critic_system = (
#             MeanFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(50)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
#         
#         locations = np.random.uniform(
#             [0., 0., 0., 0., 0., 0., 0., 0., -0.8, -0.8], 
#             [5., 5., 5., 5., 5., 5., 5., 5., 0.8, 0.8],
#             size = (n_centers, 10))
#         
#         intermediate_critic.set_center_locations(locations)
#         intermediate_critic.set_uncertainties(DoubleArray(1e8 * np.ones(n_centers))) # HERE
#         intermediate_critic.set_values(DoubleArray(1e3 * np.ones(n_centers)))
#         intermediate_critic.set_counters(DoubleArray(1 * np.ones(n_centers)))
#         
#         intermediate_critic.scale_multiplier = 1.
#         intermediate_critic.discount_factor = 0.999
#         intermediate_critic.exploration_incentive_factor = 1. # HERE
#         intermediate_critic.exploration_sampling_factor = 1.
#         intermediate_critic.process_uncertainty_rate = 0.001
#         intermediate_critic.center_relocalization_rate = 0.
#         intermediate_critic.epsilon = 1e-9
#                 
#         agent_systems[rover_id] = fitness_critic_system
#         
#         
# def rbfn_fitness_critic_0(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         
#         n_centers = 160
#         rbfn = Rbfn(10, n_centers, 1)
#         
#         intermediate_critic = RbfnApproximator(rbfn)
#         
#         fitness_critic_system = (
#             MeanFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(50)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(50)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
#         
#         locations = np.random.uniform(
#             [0., 0., 0., 0., 0., 0., 0., 0., -0.8, -0.8], 
#             [5., 5., 5., 5., 5., 5., 5., 5., 0.8, 0.8],
#             size = (n_centers, 10))
#         
#         intermediate_critic.set_center_locations(locations)
#         intermediate_critic.set_uncertainties(DoubleArray(1e8 * np.ones(n_centers))) # HERE
#         intermediate_critic.set_values(DoubleArray(1e3 * np.ones(n_centers)))
#         intermediate_critic.set_counters(DoubleArray(1 * np.ones(n_centers)))
#         
#         intermediate_critic.scale_multiplier = 1.
#         intermediate_critic.discount_factor = 0.999
#         intermediate_critic.exploration_incentive_factor = 0. # HERE
#         intermediate_critic.exploration_sampling_factor = 0.
#         intermediate_critic.process_uncertainty_rate = 0.0001
#         intermediate_critic.center_relocalization_rate = 0.
#         intermediate_critic.epsilon = 1e-9
#                 
#         agent_systems[rover_id] = fitness_critic_system


# def mean_robust_fitness_critic(arg_dict):
#     multiagent_system = arg_dict["trial"].system
#     
#     agent_systems = multiagent_system.agent_systems()
#     
#     for rover_id in range(len(agent_systems)):
#         evolving_system = agent_systems[rover_id]
#         
#         map = Rbfn(10, 160, 1)
#         #map.leaky_scale = 0.1
#         critic = DifferentiableCriticMap(map)
#         
#         intermediate_critic = RobustFunctionApproximator(critic)
#         
#         fitness_critic_system = (
#             MeanFitnessCriticSystem(
#                 evolving_system,
#                 intermediate_critic))
#                 
#         fitness_critic_system.trajectory_buffer().set_capacity(500)
#         fitness_critic_system.critic_target_buffer().set_capacity(2500)
#         fitness_critic_system.set_n_critic_update_batches_per_epoch(20)
#         fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
#         fitness_critic_system.set_critic_update_batch_size(25)
#         
#         intermediate_critic.set_learning_rate(1e-2)
#                 
#         agent_systems[rover_id] = fitness_critic_system
        
def q_fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]
        
        map = ReluLinear(10, 60, 1)
        
        critic = DifferentiableCriticMap(map)
        
        intermediate_critic = DifferentiableFunctionApproximator(critic)
        
        fitness_critic_system = (
            FitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(500)
        fitness_critic_system.critic_target_buffer().set_capacity(2500)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(1)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(1)
        fitness_critic_system.set_critic_update_batch_size(1)
        
        intermediate_critic.set_learning_rate(1e-4)
        
        q_value_target_setter = TdLambdaTargetSetter(intermediate_critic)
        
        fitness_critic_system.set_value_target_setter(q_value_target_setter)
        q_value_target_setter.set_trace_decay(0.)
                
        agent_systems[rover_id] = fitness_critic_system
        
def monte_fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems[rover_id]
        
        map = ReluLinear(10, 60, 1)
        
        critic = DifferentiableCriticMap(map)
        
        intermediate_critic = DifferentiableFunctionApproximator(critic)
        
        fitness_critic_system = (
            FitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(50)
        fitness_critic_system.critic_target_buffer().set_capacity(250)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(10)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
        fitness_critic_system.set_critic_update_batch_size(25)
        
        intermediate_critic.set_learning_rate(1e-4)
        
        q_value_target_setter = TdLambdaTargetSetter(intermediate_critic)
        q_value_target_setter.set_trace_decay(1.)
        
        fitness_critic_system.set_value_target_setter(q_value_target_setter)
                
        agent_systems[rover_id] = fitness_critic_system

#             
# def new_mutation(factor):            
#     def new_mutation_x(arg_dict):
#         domain = arg_dict["trial"].domain 
#         super_domain = domain.super_domain
#         n_rovers = super_domain.setting_state().n_rovers()
#         
#         max_n_epochs = 5000  # HERE
#         
#         n_state_dims = (
#             2 * super_domain.rover_observations_calculator().n_observation_sections())
#         n_action_dims = 2
#         
#         n_policies_per_agent = 50
#         n_policy_hidden_neurons = 32
#         
#         multiagent_system = MultiagentSystem()
#         for rover_id in range(n_rovers):
#             agent_system = DefaultEvolvingSystem()
#             agent_system.set_max_n_epochs(max_n_epochs)
#             multiagent_system.append_agent_system(agent_system)
#             for policy_id in range(n_policies_per_agent):
#                 phenotype = (
#                     CauchyPhenotype(
#                         NeuroPolicy(
#                             ReluTanh(
#                                 n_state_dims,
#                                 n_policy_hidden_neurons,
#                                 n_action_dims ))))
#                                 
#                 phenotype.set_mutation_factor(factor)
#                 
#                 agent_system.append_phenotype (phenotype)
#         
#         arg_dict["trial"].system = multiagent_system
#                 
#     new_mutation_x.__name__ = "new_mutation_{0}".format(factor)
#     return new_mutation_x
# 
# def new_setup(n_rovers, n_pois, n_req, bpv):            
#     def new_setup_x(arg_dict):
#         domain = arg_dict["trial"].domain 
#         domain.set_n_rovers(n_rovers)
#         domain.set_n_pois(n_pois)
#         domain.super_domain.evaluator().set_n_req(n_req)
#         domain.base_poi_value = bpv
#                 
#     new_setup_x.__name__ = (
#         "new_setup_{0}_{1}_{2}_{3}".format(n_rovers, n_pois, n_req, bpv))
#     return new_setup_x
