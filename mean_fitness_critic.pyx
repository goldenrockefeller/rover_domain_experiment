
import datetime as dt
import os
import csv
import errno
cimport cython

from goldenrockefeller.policyopt.value_target cimport new_TotalRewardTargetSetter
from goldenrockefeller.policyopt.value_target cimport BaseValueTargetSetter
from goldenrockefeller.policyopt.buffer cimport ShuffleBuffer, new_ShuffleBuffer
from goldenrockefeller.policyopt.experience cimport ExperienceDatum, new_ExperienceDatum
from goldenrockefeller.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry, new_TargetEntry
from goldenrockefeller.policyopt.system cimport BaseSystem
from goldenrockefeller.cyutil.array cimport DoubleArray, new_DoubleArray


from goldenrockefeller.policyopt.fitness_critic cimport FitnessCriticSystem, init_FitnessCriticSystem

import numpy as np
from typing import List


@cython.warn.undeclared(True)
cdef class MeanFitnessCriticSystem(FitnessCriticSystem):

    @cython.locals(trajectory = list, target_entries = list)
    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t batch_id
        cdef Py_ssize_t trajectory_id
        cdef Py_ssize_t target_id
        cdef Py_ssize_t n_trajectories_per_batch
        cdef Py_ssize_t n_batches
        cdef Py_ssize_t batch_size
        cdef ShuffleBuffer trajectory_buffer
        cdef ShuffleBuffer critic_target_buffer
        cdef BaseValueTargetSetter value_target_setter
        trajectory: List[ExperienceDatum]
        target_entries: List[TargetEntry]
        cdef TargetEntry target_entry
        cdef BaseFunctionApproximator intermediate_critic
        cdef ExperienceDatum experience
        cdef BaseSystem system
        cdef double fitness
        cdef double error
        cdef double mean
        cdef DoubleArray eval
        cdef DoubleArray target

        n_batches = (
            self.n_critic_update_batches_per_epoch())

        n_trajectories_per_batch = (
            self.n_trajectories_per_critic_update_batch())

        trajectory_buffer = self.trajectory_buffer()
        critic_target_buffer = self.critic_target_buffer()
        intermediate_critic = self.intermediate_critic()

        value_target_setter = self.value_target_setter()

        # if not trajectory_buffer.is_empty():
        #     for batch_id in range(n_batches):
        #         for trajectory_id in range(n_trajectories_per_batch):
        #             trajectory = trajectory_buffer.next_shuffled_datum()
        #
        #             target_entries = [None] * len(trajectory)
        #
        #             fitness = 0.
        #             for experience in trajectory:
        #                 fitness += experience.reward
        #             fitness /= len(trajectory)
        #
        #             for target_id in range(len(trajectory)):
        #                 experience = trajectory[target_id]
        #                 target = new_DoubleArray(1)
        #                 target.view[0] = fitness
        #                 # target.view[0] = experience.reward
        #                 target_entry = new_TargetEntry()
        #                 target_entry.input = experience
        #                 target_entry.target = target
        #                 target_entries[target_id] = target_entry
        #
        #
        #             intermediate_critic.batch_update(target_entries)
        #
        #     eval = intermediate_critic.eval(experience)
        #     #print("Estimate: ", eval.view[0])

        if not trajectory_buffer.is_empty():
            for batch_id in range(n_batches):
                for trajectory_id in range(n_trajectories_per_batch):
                    trajectory = trajectory_buffer.next_shuffled_datum()

                    target_entries = [None] * len(trajectory)

                    fitness = 0.
                    for experience in trajectory:
                        fitness += experience.reward

                    mean = 0.
                    for experience in trajectory:
                        mean += intermediate_critic.eval(experience).view[0]
                    mean /= len(trajectory)

                    error = fitness - mean

                    for target_id in range(len(trajectory)):
                        experience = trajectory[target_id]
                        target = new_DoubleArray(1)
                        target.view[0] = (
                            intermediate_critic.eval(experience).view[0]
                            + error
                        )
                        # target.view[0] = experience.reward
                        target_entry = new_TargetEntry()
                        target_entry.input = experience
                        target_entry.target = target
                        target_entries[target_id] = target_entry


                    intermediate_critic.batch_update(target_entries)


            # print("Estimate: ",mean)


        system = self.super_system()
        system.prep_for_epoch()

cdef class TrajFitnessCriticSystem(FitnessCriticSystem):

    @cython.locals(trajectory = list, target_entries = list)
    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t batch_id
        cdef Py_ssize_t trajectory_id
        cdef Py_ssize_t target_id
        cdef Py_ssize_t n_trajectories_per_batch
        cdef Py_ssize_t n_batches
        cdef Py_ssize_t batch_size
        cdef ShuffleBuffer trajectory_buffer
        cdef ShuffleBuffer critic_target_buffer
        cdef BaseValueTargetSetter value_target_setter
        trajectory: List[ExperienceDatum]
        target_entries: List[TargetEntry]
        cdef TargetEntry target_entry
        cdef BaseFunctionApproximator intermediate_critic
        cdef ExperienceDatum experience
        cdef BaseSystem system
        cdef double fitness
        cdef double error
        cdef double mean
        cdef double eval
        cdef DoubleArray target

        n_batches = (
            self.n_critic_update_batches_per_epoch())

        n_trajectories_per_batch = (
            self.n_trajectories_per_critic_update_batch())

        trajectory_buffer = self.trajectory_buffer()
        critic_target_buffer = self.critic_target_buffer()
        intermediate_critic = self.intermediate_critic()

        value_target_setter = self.value_target_setter()

        if not trajectory_buffer.is_empty():
            for batch_id in range(n_batches):
                for trajectory_id in range(n_trajectories_per_batch):
                    trajectory = trajectory_buffer.next_shuffled_datum()


                    target_entry = new_TargetEntry()
                    target_entry.input = trajectory
                    target_entry.target = 0.
                    intermediate_critic.batch_update([target_entry])

            # eval = 0.
            # for experience in trajectory:
            #     eval += intermediate_critic.eval(experience).view[0]
            #
            # #print(np.asarray(intermediate_critic.network.center_shape(0).view))
            # print("Estimate: ", eval)


        system = self.super_system()
        system.prep_for_epoch()


cdef class TransferFitnessCriticSystem(MeanFitnessCriticSystem):
    cdef public Py_ssize_t n_epochs_elapsed
    cdef public Py_ssize_t n_epochs_before_switch

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_epochs_elapsed = 0
        self.n_epochs_before_switch = 2500

    cpdef void prep_for_epoch(self) except *:
        MeanFitnessCriticSystem.prep_for_epoch(self)
        self.n_epochs_elapsed += 1

    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        current_trajectory: List[Experience]
        cdef BaseSystem system
        cdef DoubleArray intermediate_eval

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]

        if self.n_epochs_elapsed < self.n_epochs_before_switch :
            system.receive_feedback(new_feedback)
        else:
            system.receive_feedback(experience.reward)

cdef class AlternatingTrajFitnessCriticSystem(TrajFitnessCriticSystem):
    cdef public Py_ssize_t n_epochs_elapsed

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_epochs_elapsed = 0

    cpdef void prep_for_epoch(self) except *:
        TrajFitnessCriticSystem.prep_for_epoch(self)
        self.n_epochs_elapsed += 1

    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        current_trajectory: List[ExperienceDatum]
        cdef BaseSystem system
        cdef DoubleArray intermediate_eval

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]

        if self.n_epochs_elapsed % 3 == 0 :
            system.receive_feedback(new_feedback)
        else:
            system.receive_feedback(experience.reward)

cdef class MeanSumFitnessCriticSystem(MeanFitnessCriticSystem):
    #step_wise feedback
    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        current_trajectory: List[ExperienceDatum]
        cdef BaseSystem system
        cdef DoubleArray intermediate_eval

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]

        system.receive_feedback(new_feedback + experience.reward)

cdef class MeanSumFitnessCriticSystem_0(MeanFitnessCriticSystem):
    #step_wise feedback
    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        current_trajectory: List[ExperienceDatum]
        cdef BaseSystem system
        cdef DoubleArray intermediate_eval

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]


        system.receive_feedback(new_feedback + 0. * experience.reward)

cdef class RecordingMeanFitnessCriticSystem(MeanFitnessCriticSystem):
    cdef list estimates

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.estimates = []

    @cython.locals(trajectory = list)
    cpdef void receive_score(self, double score) except *:

        cdef double mean
        cdef ExperienceDatum experience
        cdef ShuffleBuffer trajectory_buffer
        cdef BaseFunctionApproximator intermediate_critic
        trajectory: List[ExperienceDatum]

        intermediate_critic = self.intermediate_critic()
        trajectory_buffer = self.trajectory_buffer()


        if not trajectory_buffer.is_empty():
            trajectory = trajectory_buffer.next_shuffled_datum()

            mean = 0.
            for experience in trajectory:
                mean += intermediate_critic.eval(experience).view[0]
            mean /= len(trajectory)

        self.estimates.append(mean)

        self.super_system().receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        cdef object save_filename
        cdef Py_ssize_t entry_id
        cdef object exc
        cdef object save_file
        cdef object writer
        cdef list data

        entry_id = 0

        save_filename = (
            os.path.join(
                log_dirname,
                "estimates",
                "estimates_{datetime_str}.csv".format(**locals())))

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['estimates'] + self.estimates)

            writer.writerow(['n_epochs_elapsed'] + list(range(len(self.estimates))))

        self.super_system().output_final_log(log_dirname, datetime_str)

cdef class RecordingTrajFitnessCriticSystem(TrajFitnessCriticSystem):
    cdef list estimates

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.estimates = []

    @cython.locals(trajectory = list)
    cpdef void receive_score(self, double score) except *:

        cdef double sum
        cdef ExperienceDatum experience
        cdef ShuffleBuffer trajectory_buffer
        cdef BaseFunctionApproximator intermediate_critic
        trajectory: List[ExperienceDatum]

        intermediate_critic = self.intermediate_critic()
        trajectory_buffer = self.trajectory_buffer()


        if not trajectory_buffer.is_empty():
            trajectory = trajectory_buffer.next_shuffled_datum()

            sum = 0.
            for experience in trajectory:
                sum += intermediate_critic.eval(experience).view[0]

        self.estimates.append(sum)

        self.super_system().receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        cdef object save_filename
        cdef Py_ssize_t entry_id
        cdef object exc
        cdef object save_file
        cdef object writer
        cdef list data

        entry_id = 0

        save_filename = (
            os.path.join(
                log_dirname,
                "estimates",
                "estimates_{datetime_str}.csv".format(**locals())))

        # Create File Directory if it doesn't exist
        if not os.path.exists(os.path.dirname(save_filename)):
            try:
                os.makedirs(os.path.dirname(save_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


        with open(save_filename, 'w', newline='') as save_file:
            writer = csv.writer(save_file)

            writer.writerow(['estimates'] + self.estimates)

            writer.writerow(['n_epochs_elapsed'] + list(range(len(self.estimates))))

        self.super_system().output_final_log(log_dirname, datetime_str)


