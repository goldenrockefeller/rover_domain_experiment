# cython: warn.undeclared = True

cimport cython

from rockefeg.policyopt.value_target cimport new_TotalRewardTargetSetter
from rockefeg.policyopt.value_target cimport BaseValueTargetSetter
from rockefeg.policyopt.buffer cimport ShuffleBuffer, new_ShuffleBuffer
from rockefeg.policyopt.experience cimport ExperienceDatum, new_ExperienceDatum
from rockefeg.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry, new_TargetEntry
from rockefeg.policyopt.system cimport BaseSystem
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray


from rockefeg.policyopt.fitness_critic cimport FitnessCriticSystem, init_FitnessCriticSystem


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

        if not trajectory_buffer.is_empty():
            for batch_id in range(n_batches):
                for trajectory_id in range(n_trajectories_per_batch):
                    trajectory = trajectory_buffer.next_shuffled_datum()

                    target_entries = [None] * len(trajectory)

                    # fitness = 0.
                    # for experience in trajectory:
                    #     fitness += experience.reward
                    # fitness /= len(trajectory)

                    for target_id in range(len(trajectory)):
                        experience = trajectory[target_id]
                        target = new_DoubleArray(1)
                        # target.view[0] = fitness
                        target.view[0] = experience.reward
                        target_entry = new_TargetEntry()
                        target_entry.input = experience
                        target_entry.target = target
                        target_entries[target_id] = target_entry


                    intermediate_critic.batch_update(target_entries)

            eval = intermediate_critic.eval(experience)
            #print("Estimate: ", eval.view[0])


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

cdef class AlternatingFitnessCriticSystem(MeanFitnessCriticSystem):
    cdef public Py_ssize_t n_epochs_elapsed

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_epochs_elapsed = 0

    cpdef void prep_for_epoch(self) except *:
        MeanFitnessCriticSystem.prep_for_epoch(self)
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

