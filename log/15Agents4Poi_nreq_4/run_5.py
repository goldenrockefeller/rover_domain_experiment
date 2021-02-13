import pyximport; pyximport.install()
from trial_setup_5 import *
import random
from shutil import copy
import os
import errno
import datetime
import itertools
import glob

experiment_arg_dict = trial_setup()
experiment_name = experiment_arg_dict["trial"].experiment_name 

n_stat_runs = 1

mods_to_mix = [
    (difference_reward,),
    (rec_rbf_critic, rec_final_gru_critic, rec_sum_gru_critic, rec_mean_fitness_critic)
]

# rec_final_gru_critic, rec_sum_gru_critic, rec_mean_fitness_critic

active_mod_combos = list(itertools.product(*mods_to_mix))

# mods_to_mix = [
#     (difference_reward,), 
#     (mean_critic, max_inexact_critic,),
#     (fixed_weights2,)
# ]
# 
# active_mod_combos += list(itertools.product(*mods_to_mix))
# 

for mod_combo in active_mod_combos:
    if not isinstance(mod_combo, tuple):
        mod_combo_str = str(mod_combo)
        raise (
            ValueError(
                "The mod combinations (str(mod_combo) = {mod_combo_str}) "
                "must be converted to a tuple."
                .format(**locals()) )) 

# Create experiment folder if not already created.
try:
    os.makedirs(os.path.dirname("log/{experiment_name}/".format(**locals())))
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

for stat_run_id in range(n_stat_runs):
    
    # Shuffle mods for evenly distributed stat runs when parallelizing.
    random.shuffle(active_mod_combos)
    
    for mod_combo in active_mod_combos:
        arg_dict = trial_setup()
        
        mod_names = []
        for mod_func in mod_combo:
            mod_func(arg_dict)
            mod_names.append(mod_func.__name__)
        trial = arg_dict["trial"]
        trial.mod_name = "_".join(mod_names)

        print(
            "Starting trial.\n"
            "experiment: {trial.experiment_name}\n"
            "mods: {mod_names}\n"
            "stat run #: {stat_run_id}\n"
            "datetime: {trial.datetime_str}\n\n" 
            .format(**locals()) )
            
        trial.run()