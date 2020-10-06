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

                    for target_id in range(len(trajectory)):
                        experience = trajectory.item(target_id)
                        target = new_DoubleArray(1)
                        target.view[0] = fitness
                        target_entry_list[target_id] = new_TargetEntry()
                        target_entry_list[target_id].input = experience
                        target_entry_list[target_id].target = target


                    target_entries = new_TypedList(TargetEntry)
                    target_entries.set_items(target_entry_list)

                    intermediate_critic.batch_update(target_entries)

            eval = intermediate_critic.eval(experience)
            print("Estimate: ", eval.view[0])


        system = self.super_system()
        system.prep_for_epoch()