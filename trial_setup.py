import pyximport; pyximport.install()

from rockefeg.policyopt.trial import Trial
from rockefeg.policyopt.multiagent_system import MultiagentSystem
from rockefeg.policyopt.evolution import DefaultEvolvingSystem, DefaultPhenotype
from rockefeg.policyopt.neural_network import NeuroPolicy, ReluTanh

from rockefeg.roverdomain.evaluator import DefaultEvaluator
from rockefeg.roverdomain.evaluator import DifferenceEvaluator

from novelty import EvolvingSystemWithNoveltySearch
from novelty import PhenotypeWithNoveltyScore

from trial_rover_domain import TrialRoverDomain

import numpy as np

def trial_setup():
    arg_dict = {}
    experiment_name = "Novelty_n_req_3" 
    n_req = 3
    n_rovers = 15
    n_pois = 4
    prints_score = True
    
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

    # Muliagent System
    n_policies_per_agent = 50
    n_policy_hidden_neurons = 32
    
    multiagent_system = MultiagentSystem()
    for rover_id in range(n_rovers):
        agent_system = DefaultEvolvingSystem()
        agent_system.set_max_n_epochs(max_n_epochs)
        multiagent_system.append_agent_system(agent_system)
        for policy_id in range(n_policies_per_agent):
            phenotype = (
                DefaultPhenotype(
                    NeuroPolicy(
                        ReluTanh(
                            n_state_dims,
                            n_policy_hidden_neurons,
                            n_action_dims ))))
                            
            phenotype.set_mutation_rate(0.01)
            phenotype.set_mutation_factor(1.)
            
            agent_system.append_phenotype (phenotype)
        
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
    
def novelty(arg_dict):
    multiagent_system = arg_dict["trial"].system 
    old_systems = multiagent_system.agent_systems_shallow_copy()
    multiagent_system.set_agent_systems([])
    
    for old_system in old_systems:
        if not isinstance(old_system, DefaultEvolvingSystem):
            raise RuntimeError("Something went wrong")
        
        new_system = (
            EvolvingSystemWithNoveltySearch(
                n_state_action_dims = arg_dict["n_state_action_dims"], 
                n_novelty_nodes = 60, 
                using_action = False))
        
        new_system.set_max_n_epochs(old_system.max_n_epochs())
        
        multiagent_system.append_agent_system(new_system)
        for policy_id in range(old_system.n_phenotypes()):
            old_phenotype = old_system.phenotype(policy_id)
            policy = old_phenotype.policy()
            phenotype = PhenotypeWithNoveltyScore(policy)
                    
                            
            phenotype.set_mutation_rate(old_phenotype.mutation_rate())
            phenotype.set_mutation_factor(old_phenotype.mutation_factor())
            
            new_system.append_phenotype (phenotype)

#     