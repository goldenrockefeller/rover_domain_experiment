cimport cython

from rockefeg.policyopt.value_target cimport new_TotalRewardTargetSetter
from rockefeg.policyopt.value_target cimport BaseValueTargetSetter
from rockefeg.policyopt.buffer cimport ShuffleBuffer, new_ShuffleBuffer
from rockefeg.policyopt.experience cimport ExperienceDatum, new_ExperienceDatum
from rockefeg.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry, new_TargetEntry
from rockefeg.policyopt.system cimport BaseSystem
from rockefeg.cyutil.typed_list cimport TypedList, new_TypedList
from rockefeg.cyutil.typed_list cimport is_sub_full_type
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray


from rockefeg.policyopt.fitness_critic cimport FitnessCriticSystem

@cython.warn.undeclared(True)
cdef class MeanFitnessCriticSystem(FitnessCriticSystem):
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
        cdef TypedList trajectory
        cdef TypedList target_entries
        cdef TargetEntry target_entry
        cdef list target_entry_list
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

                    target_entry_list = [None] * len(trajectory)

                    fitness = 0.
                    for experience in trajectory:
                        fitness += experience.reward
                    fitness /= len(trajectory)

                    for target_id in range(len(trajectory)):
                        experience = trajectory.item(target_id)
                        target = new_DoubleArray(1)
                        target.view[0] = fitness
                        target_entry = new_TargetEntry()
                        target_entry.input = experience
                        target_entry.target = target
                        target_entry_list[target_id] = target_entry


                    target_entries = new_TypedList(TargetEntry)
                    target_entries.set_items(target_entry_list)

                    intermediate_critic.batch_update(target_entries)

            eval = intermediate_critic.eval(experience)
            #print("Estimate: ", eval.view[0])


        system = self.super_system()
        system.prep_for_epoch()


cdef class MeanSumFitnessCriticSystem(MeanFitnessCriticSystem):
    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        cdef TypedList current_trajectory
        cdef BaseSystem system
        cdef DoubleArray intermediate_eval

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.state = self.current_state()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]

        system.receive_feedback(new_feedback + experience.reward)

cdef class MeanSumFitnessCriticSystem_0(MeanFitnessCriticSystem):
    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        cdef TypedList current_trajectory
        cdef BaseSystem system
        cdef DoubleArray intermediate_eval

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.state = self.current_state()
        experience.action = self.current_action()
        experience.reward = feedback


        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        intermediate_eval = intermediate_critic.eval(experience)
        new_feedback = intermediate_eval.view[0]


        system.receive_feedback(new_feedback + 0. * experience.reward)

