import pyximport; pyximport.install()

from rockefeg.policyopt.trial import Trial
from rockefeg.policyopt.multiagent_system import MultiagentSystem
from rockefeg.policyopt.evolution import DefaultEvolvingSystem, DefaultPhenotype
from rockefeg.policyopt.neural_network import ReluTanh, ReluLinear

from rockefeg.roverdomain.evaluator import DefaultEvaluator
from rockefeg.roverdomain.evaluator import DifferenceEvaluator

from rockefeg.policyopt.function_approximation import DifferentiableFunctionApproximator
from rockefeg.policyopt.map import DifferentiableCriticMap
from rockefeg.policyopt.fitness_critic import FitnessCriticSystem
from rockefeg.policyopt.value_target import TdLambdaTargetSetter

# from novelty import EvolvingSystemWithNoveltySearch
# from novelty import PhenotypeWithNoveltyScore

from trial_rover_domain import TrialRoverDomain

# from cauchy_phenotype import CauchyPhenotype

import numpy as np

def trial_setup():
    arg_dict = {}
    experiment_name = "FitnessCritic1" 
    n_req = 5
    n_rovers = 15
    base_poi_value = 1.
    n_pois = 4
    prints_score = False
    
    max_n_epochs = 5000  # HERE
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
    n_policies_per_agent = 50
    n_policy_hidden_neurons = 32
    
    multiagent_system = MultiagentSystem()
    for rover_id in range(n_rovers):
        evolving_system = DefaultEvolvingSystem()
        evolving_system.set_max_n_epochs(max_n_epochs)
        multiagent_system.agent_systems().append(evolving_system)
        for policy_id in range(n_policies_per_agent):
            phenotype = (
                DefaultPhenotype(
                        ReluTanh(
                            n_state_dims,
                            n_policy_hidden_neurons,
                            n_action_dims )))
                            
            phenotype.set_mutation_rate(0.01)
            phenotype.set_mutation_factor(1.)
            
            evolving_system.phenotypes().append(phenotype)
        
    trial_args = {}
    trial_args["system"] = multiagent_system
    trial_args["domain"] = domain
    trial_args["experiment_name"] = experiment_name
    trial_args["mod_name"] = "experiment"
    trial_args["prints_score"] = prints_score
    
    arg_dict["trial"] = Trial(trial_args)
    arg_dict["n_state_action_dims"] = n_state_dims + n_action_dims
    
    return arg_dict

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
    
def fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems.item(rover_id)
        
        map = ReluLinear(10, 60, 1)
        
        critic = DifferentiableCriticMap(map)
        
        intermediate_critic = DifferentiableFunctionApproximator(critic)
        
        fitness_critic_system = (
            FitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(500)
        fitness_critic_system.critic_target_buffer().set_capacity(2500)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(10)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
        fitness_critic_system.set_critic_update_batch_size(25)
        
        intermediate_critic.set_learning_rate(1e-6)
                
        agent_systems.set_item(rover_id, fitness_critic_system)
        
def q_fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems.item(rover_id)
        
        map = ReluLinear(10, 60, 1)
        
        critic = DifferentiableCriticMap(map)
        
        intermediate_critic = DifferentiableFunctionApproximator(critic)
        
        fitness_critic_system = (
            FitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(500)
        fitness_critic_system.critic_target_buffer().set_capacity(2500)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(10)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
        fitness_critic_system.set_critic_update_batch_size(25)
        
        intermediate_critic.set_learning_rate(1e-6)
        
        q_value_target_setter = TdLambdaTargetSetter(intermediate_critic)
        
        fitness_critic_system.set_value_target_setter(q_value_target_setter)
        q_value_target_setter.set_trace_decay(0.)
                
        agent_systems.set_item(rover_id, fitness_critic_system)
        
def monte_fitness_critic(arg_dict):
    multiagent_system = arg_dict["trial"].system
    
    agent_systems = multiagent_system.agent_systems()
    
    for rover_id in range(len(agent_systems)):
        evolving_system = agent_systems.item(rover_id)
        
        map = ReluLinear(10, 60, 1)
        
        critic = DifferentiableCriticMap(map)
        
        intermediate_critic = DifferentiableFunctionApproximator(critic)
        
        fitness_critic_system = (
            FitnessCriticSystem(
                evolving_system,
                intermediate_critic))
                
        fitness_critic_system.trajectory_buffer().set_capacity(500)
        fitness_critic_system.critic_target_buffer().set_capacity(2500)
        fitness_critic_system.set_n_critic_update_batches_per_epoch(10)
        fitness_critic_system.set_n_trajectories_per_critic_update_batch(5)
        fitness_critic_system.set_critic_update_batch_size(25)
        
        intermediate_critic.set_learning_rate(1e-6)
        
        q_value_target_setter = TdLambdaTargetSetter(intermediate_critic)
        q_value_target_setter.set_trace_decay(1.)
        
        fitness_critic_system.set_value_target_setter(q_value_target_setter)
                
        agent_systems.set_item(rover_id, fitness_critic_system)


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