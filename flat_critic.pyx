# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = cpp_core/flat_network_approximator.cpp


cimport cython
from cpp_flat_critic cimport valarray
from cpp_flat_critic cimport Experience as CppExperienceDatum
from cpp_flat_critic cimport FlatNetworkApproximator as CppFlatNetworkApproximator
from cpp_flat_critic cimport MonteFlatNetworkApproximator as CppMonteFlatNetworkApproximator
from cpp_flat_critic cimport DiscountFlatNetworkApproximator as CppDiscountFlatNetworkApproximator
from cpp_flat_critic cimport QFlatNetworkApproximator as CppQFlatNetworkApproximator
from cpp_flat_critic cimport UFlatNetworkApproximator as CppUFlatNetworkApproximator
from cpp_flat_critic cimport UqFlatNetworkApproximator as CppUqFlatNetworkApproximator
from cpp_flat_critic cimport FlatNetwork as CppFlatNetwork
from cpp_flat_critic cimport Approximator as CppApproximator

from libcpp.memory cimport shared_ptr, unique_ptr, make_shared
from libcpp.vector cimport vector
from libcpp.utility cimport move
# from libc.math cimport isfinite


from goldenrockefeller.cyutil.array cimport DoubleArray, new_DoubleArray

from goldenrockefeller.policyopt.experience cimport ExperienceDatum, new_ExperienceDatum
from goldenrockefeller.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry, new_TargetEntry
from goldenrockefeller.policyopt.buffer cimport ShuffleBuffer, new_ShuffleBuffer
from goldenrockefeller.policyopt.fitness_critic cimport FitnessCriticSystem, init_FitnessCriticSystem
from goldenrockefeller.policyopt.system cimport BaseSystem
#

cdef valarray[double] valarray_from_DoubleArray(DoubleArray arr) except *:
    cdef valarray[double] new_arr
    cdef Py_ssize_t id

    new_arr.resize(len(arr))

    for id in range(len(arr)):
        new_arr[<size_t>id] = arr.view[id]

    return new_arr

cdef DoubleArray DoubleArray_from_valarray(valarray[double] arr):
    cdef DoubleArray new_arr
    cdef Py_ssize_t id

    new_arr = new_DoubleArray(arr.size())

    for id in range(arr.size()):
        new_arr.view[id] = arr[<size_t>id]

    return new_arr

cdef CppExperienceDatum CppExperienceDatum_from_ExperienceDatum(ExperienceDatum experience) except *:
    cdef CppExperienceDatum cpp_experience

    cpp_experience.observation = valarray_from_DoubleArray(experience.observation)
    cpp_experience.action = valarray_from_DoubleArray(experience.action)
    cpp_experience.reward = experience.reward

    return cpp_experience

cdef vector[CppExperienceDatum] vector_from_trajectory(list trajectory) except *:
    # TODO trajectory is a sequence (typing)

    cdef vector[CppExperienceDatum] cpp_trajectory
    cdef ExperienceDatum experience


    cpp_trajectory.resize(0)
    cpp_trajectory.reserve(len(trajectory))

    for experience in trajectory:
        cpp_trajectory.push_back(CppExperienceDatum_from_ExperienceDatum(experience))

    return cpp_trajectory


cdef class FlatNetwork():
    cdef shared_ptr[CppFlatNetwork] core

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppFlatNetwork](n_in_dims, n_hidden_units)

    cpdef FlatNetwork copy(self, copy_obj = None):
        cdef FlatNetwork new_network
        cdef unique_ptr[CppFlatNetwork] new_core

        if copy_obj is None:
            new_network = FlatNetwork.__new__(FlatNetwork)
        else:
            new_network = copy_obj

        new_core.swap(self.core.get().copy())
        new_network.core.reset(new_core.get())
        new_core.release()

        return new_network


    cpdef DoubleArray parameters(self):
        return DoubleArray_from_valarray(self.core.get().parameters())

    cpdef void set_parameters(self, DoubleArray parameters)  except *:
        self.core.get().set_parameters(valarray_from_DoubleArray(parameters))


    cpdef double eval(self, DoubleArray input) except *:
        return self.core.get().eval(valarray_from_DoubleArray(input))

    cpdef DoubleArray grad_wrt_parameters(self, DoubleArray input, double output_grad):
        return DoubleArray_from_valarray(self.core.get().grad_wrt_parameters(valarray_from_DoubleArray(input), output_grad))


    @property
    def leaky_scale(self):
        return self.core.get().leaky_scale

    @leaky_scale.setter
    def leaky_scale(self, double value):
        self.core.get().leaky_scale = value



cdef class FlatNetworkApproximator(BaseFunctionApproximator):
    cdef shared_ptr[CppFlatNetworkApproximator] core

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppFlatNetworkApproximator](n_in_dims, n_hidden_units)

    cpdef eval(self, input):
        return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))

    cpdef void update_using_trajectory(self, list trajectory) except *:
        self.core.get().update(vector_from_trajectory(trajectory))

    cpdef void update_using_experience(self, ExperienceDatum experience, double target_value) except *:
        self.core.get().update(CppExperienceDatum_from_ExperienceDatum(experience), target_value)

    @property
    def time_horizon(self):
        return self.core.get().optimizer.time_horizon

    @time_horizon.setter
    def time_horizon(self, double value):
        self.core.get().optimizer.time_horizon = value

    @property
    def epsilon(self):
        return self.core.get().optimizer.epsilon

    @epsilon.setter
    def epsilon(self, double value):
        self.core.get().optimizer.epsilon = value

    @property
    def learning_rate(self):
        return self.core.get().optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, double value):
        self.core.get().optimizer.learning_rate = value

    @property
    def learning_mode(self):
        return self.core.get().optimizer.learning_mode

    @learning_mode.setter
    def learning_mode(self, int value):
        self.core.get().optimizer.learning_mode = value



cdef class MonteFlatNetworkApproximator(FlatNetworkApproximator):

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = shared_ptr[CppFlatNetworkApproximator](new CppMonteFlatNetworkApproximator(n_in_dims, n_hidden_units))


cdef class DiscountFlatNetworkApproximator(FlatNetworkApproximator):

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = shared_ptr[CppFlatNetworkApproximator](new CppDiscountFlatNetworkApproximator(n_in_dims, n_hidden_units))
#
#
# cdef class QFlatNetworkApproximator(Approximator):
#     cdef shared_ptr[CppQFlatNetworkApproximator] core
#
#     def __init__(self, size_t n_in_dims, size_t n_hidden_units):
#         self.core = make_shared[CppQFlatNetworkApproximator](n_in_dims, n_hidden_units)
#
#
#     cpdef eval(self, input):
#         return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))
#
#
#     cpdef void update_using_trajectory(self, list trajectory) except *:
#         # TODO trajectory is a sequence (typing)
#         self.core.get().update(vector_from_trajectory(trajectory))
#
#     @property
#     def time_horizon(self):
#         return self.core.get().optimizer.time_horizon
#
#     @time_horizon.setter
#     def time_horizon(self, double value):
#         self.core.get().optimizer.time_horizon = value
#
#     @property
#     def epsilon(self):
#         return self.core.get().optimizer.epsilon
#
#     @epsilon.setter
#     def epsilon(self, double value):
#         self.core.get().optimizer.epsilon = value
#
#     @property
#     def learning_rate(self):
#         return self.core.get().optimizer.learning_rate
#
#     @learning_rate.setter
#     def learning_rate(self, double value):
#         self.core.get().optimizer.learning_rate = value
#
#     @property
#     def learning_mode(self):
#         return self.core.get().optimizer.learning_mode
#
#     @learning_mode.setter
#     def learning_mode(self, int value):
#         self.core.get().optimizer.learning_mode = value
#
#
# cdef class UFlatNetworkApproximator(Approximator):
#     cdef shared_ptr[CppUFlatNetworkApproximator] core
#
#
#     def __init__(self, size_t n_in_dims, size_t n_hidden_units):
#         self.core = make_shared[CppUFlatNetworkApproximator](n_in_dims, n_hidden_units)
#
#     cpdef eval(self, input):
#         return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))
#
#
#     cpdef void update_using_trajectory(self, list trajectory) except *:
#         # TODO trajectory is a sequence (typing)
#         self.core.get().update(vector_from_trajectory(trajectory))
#
#     @property
#     def time_horizon(self):
#         return self.core.get().optimizer.time_horizon
#
#     @time_horizon.setter
#     def time_horizon(self, double value):
#         self.core.get().optimizer.time_horizon = value
#
#     @property
#     def epsilon(self):
#         return self.core.get().optimizer.epsilon
#
#     @epsilon.setter
#     def epsilon(self, double value):
#         self.core.get().optimizer.epsilon = value
#
#     @property
#     def learning_rate(self):
#         return self.core.get().optimizer.learning_rate
#
#     @learning_rate.setter
#     def learning_rate(self, double value):
#         self.core.get().optimizer.learning_rate = value
#
#     @property
#     def learning_mode(self):
#         return self.core.get().optimizer.learning_mode
#
#     @learning_mode.setter
#     def learning_mode(self, int value):
#         self.core.get().optimizer.learning_mode = value
#
#
#
# cdef class UqFlatNetworkApproximator(Approximator):
#     cdef shared_ptr[CppUqFlatNetworkApproximator] core
#
#     def __init__(
#         self,
#         UFlatNetworkApproximator u_approximator,
#         QFlatNetworkApproximator q_approximator
#     ):
#
#         self.core = make_shared[CppUqFlatNetworkApproximator](u_approximator.core, q_approximator.core)
#
#     cpdef eval(self, input):
#         return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))
#
#
#     cpdef void update_using_trajectory(self, list trajectory) except *:
#         # TODO trajectory is a sequence (typing)
#         self.core.get().update(vector_from_trajectory(trajectory))



cdef class FlatFitnessCriticSystem(FitnessCriticSystem):
    cdef public Py_ssize_t n_critic_updates_per_epoch
    cdef public ShuffleBuffer experience_target_buffer
    cdef public Py_ssize_t uses_experience_targets_for_updates

    def __init__(
            self,
            BaseSystem super_system,
            size_t n_in_dims, size_t n_hidden_units):
        intermediate_critic = FlatNetworkApproximator(n_in_dims, n_hidden_units)
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_critic_updates_per_epoch = 1
        self.experience_target_buffer = new_ShuffleBuffer()


    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef BaseFunctionApproximator intermediate_critic
        cdef list current_trajectory

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        self.current_trajectory().append(experience)

        new_feedback = intermediate_critic.eval(experience)
        #
        #
        # # if not isfinite(new_feedback):
        # #     raise RuntimeError("Something went wrong: feedback is not finite.")
        #
        self.super_system().receive_feedback(new_feedback)

    cpdef void update_policy(self) except *:
        cdef list current_trajectory = self.current_trajectory()
        FitnessCriticSystem.update_policy(self)

        if self.uses_experience_targets_for_updates:
            self.extract_experience_targets(current_trajectory)

    cpdef void extract_experience_targets(self, list trajectory) except *:
        cdef ExperienceDatum experience
        cdef double traj_eval
        cdef double step_eval
        cdef double sample_fitness
        cdef BaseFunctionApproximator intermediate_critic  = self.intermediate_critic()
        cdef double error
        cdef TargetEntry target_entry
        cdef Py_ssize_t traj_len = len(trajectory)

        for experience in trajectory:
            sample_fitness += experience.reward
            traj_eval += intermediate_critic.eval(experience)

        traj_eval /= traj_len

        error = sample_fitness - traj_eval

        for experience in trajectory:
            step_eval = intermediate_critic.eval(experience)
            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = error + step_eval
            self.experience_target_buffer.add_staged_datum(target_entry)


    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t update_id
        cdef BaseSystem system
        cdef FlatNetworkApproximator approximator = self.intermediate_critic()
        cdef list trajectory
        cdef Py_ssize_t n_updates = self.n_critic_updates_per_epoch
        cdef TargetEntry target_entry


        if not self._trajectory_buffer.is_empty():
            for update_id in range(n_updates):
                if self.uses_experience_targets_for_updates:
                    target_entry = self.experience_target_buffer.next_shuffled_datum()
                    approximator.update_using_experience(target_entry.input, target_entry.target)
                else:
                    trajectory = self._trajectory_buffer.next_shuffled_datum()
                    approximator.update_using_trajectory(trajectory)

            # print(approximator.eval(trajectory[0]))
            # print(len(trajectory))
        system = self.super_system()
        system.prep_for_epoch()


cdef class MonteFlatFitnessCriticSystem(FlatFitnessCriticSystem):
    def __init__(
            self,
            BaseSystem super_system,
            size_t n_in_dims, size_t n_hidden_units):
        intermediate_critic = MonteFlatNetworkApproximator(n_in_dims, n_hidden_units)
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_critic_updates_per_epoch = 1
        self.experience_target_buffer = new_ShuffleBuffer()


    cpdef void extract_experience_targets(self, list trajectory) except *:
        cdef ExperienceDatum experience
        cdef double sample_fitness
        cdef TargetEntry target_entry

        for experience in trajectory:
            sample_fitness += experience.reward

        for experience in trajectory:
            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = sample_fitness
            self.experience_target_buffer.add_staged_datum(target_entry)


cdef class DiscountFlatFitnessCriticSystem(FlatFitnessCriticSystem):
    cdef double discount_factor

    def __init__(
            self,
            BaseSystem super_system,
            size_t n_in_dims, size_t n_hidden_units):
        intermediate_critic = DiscountFlatNetworkApproximator(n_in_dims, n_hidden_units)
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_critic_updates_per_epoch = 1
        self.experience_target_buffer = new_ShuffleBuffer()
        self.discount_factor = 0.97


    cpdef void extract_experience_targets(self, list trajectory) except *:
        cdef ExperienceDatum experience, future_experience
        cdef Py_ssize_t experience_id, future_experience_id
        cdef double discounted_fitness
        cdef TargetEntry target_entry

        for experience_id, experience in enumerate(trajectory):
            discounted_fitness = 0.
            for future_experience_id, future_experience in enumerate(trajectory):
                if future_experience_id > experience_id:
                    discounted_fitness += (
                        future_experience.reward
                        * self.discount_factor
                        ** (future_experience_id - experience_id)
                    )

            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = discounted_fitness
            self.experience_target_buffer.add_staged_datum(target_entry)

