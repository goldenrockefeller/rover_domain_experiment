# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = cpp_core/flat_network_approximator.cpp


cimport cython
from cpp_flat_critic cimport valarray
from cpp_flat_critic cimport Experience as CppExperienceDatum
from cpp_flat_critic cimport FlatNetworkApproximator as CppFlatNetworkApproximator
from cpp_flat_critic cimport MonteFlatNetworkApproximator as CppMonteFlatNetworkApproximator
from cpp_flat_critic cimport QFlatNetworkApproximator as CppQFlatNetworkApproximator
from cpp_flat_critic cimport UFlatNetworkApproximator as CppUFlatNetworkApproximator
from cpp_flat_critic cimport UqFlatNetworkApproximator as CppUqFlatNetworkApproximator

from libcpp.memory cimport shared_ptr, make_shared
from libcpp.vector cimport vector
# from libc.math cimport isfinite


from goldenrockefeller.cyutil.array cimport DoubleArray

from goldenrockefeller.policyopt.experience cimport ExperienceDatum, new_ExperienceDatum
from goldenrockefeller.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry, new_TargetEntry
from goldenrockefeller.policyopt.buffer cimport ShuffleBuffer
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


cdef class Approximator(BaseFunctionApproximator):
    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        raise NotImplementedError("Abstract class.")

    cpdef eval(self, input):
        raise NotImplementedError("Abstract class.")


    cpdef void batch_update(self, list trajectory) except *:
        # TODO trajectory is a sequence (typing)
        raise NotImplementedError("Abstract class.")


cdef class FlatNetworkApproximator(Approximator):
    cdef shared_ptr[CppFlatNetworkApproximator] core


    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppFlatNetworkApproximator](n_in_dims, n_hidden_units)

    cpdef eval(self, input):
        return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))


    cpdef void batch_update(self, list trajectory) except *:
        # TODO trajectory is a sequence (typing)
        self.core.get().update(vector_from_trajectory(trajectory))

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



cdef class MonteFlatNetworkApproximator(Approximator):
    cdef shared_ptr[CppMonteFlatNetworkApproximator] core

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppMonteFlatNetworkApproximator](n_in_dims, n_hidden_units)

    cpdef eval(self, input):
        return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))


    cpdef void batch_update(self, list trajectory) except *:
        # TODO trajectory is a sequence (typing)
        self.core.get().update(vector_from_trajectory(trajectory))

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



cdef class QFlatNetworkApproximator(Approximator):
    cdef shared_ptr[CppQFlatNetworkApproximator] core

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppQFlatNetworkApproximator](n_in_dims, n_hidden_units)


    cpdef eval(self, input):
        return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))


    cpdef void batch_update(self, list trajectory) except *:
        # TODO trajectory is a sequence (typing)
        self.core.get().update(vector_from_trajectory(trajectory))

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


cdef class UFlatNetworkApproximator(Approximator):
    cdef shared_ptr[CppUFlatNetworkApproximator] core


    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppUFlatNetworkApproximator](n_in_dims, n_hidden_units)

    cpdef eval(self, input):
        return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))


    cpdef void batch_update(self, list trajectory) except *:
        # TODO trajectory is a sequence (typing)
        self.core.get().update(vector_from_trajectory(trajectory))

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



cdef class UqFlatNetworkApproximator(Approximator):
    cdef shared_ptr[CppUqFlatNetworkApproximator] core

    def __init__(
        self,
        UFlatNetworkApproximator u_approximator,
        QFlatNetworkApproximator q_approximator
    ):

        self.core = make_shared[CppUqFlatNetworkApproximator](u_approximator.core, q_approximator.core)

    cpdef eval(self, input):
        return self.core.get().eval(CppExperienceDatum_from_ExperienceDatum(input))


    cpdef void batch_update(self, list trajectory) except *:
        # TODO trajectory is a sequence (typing)
        self.core.get().update(vector_from_trajectory(trajectory))



cdef class FlatFitnessCriticSystem(FitnessCriticSystem):

    def __init__(
            self,
            BaseSystem super_system,
            Approximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)

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


        # if not isfinite(new_feedback):
        #     raise RuntimeError("Something went wrong: feedback is not finite.")

        self.super_system().receive_feedback(new_feedback)


    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t update_id
        cdef BaseSystem system
        cdef Approximator approximator = self.intermediate_critic()
        cdef list trajectory
        cdef Py_ssize_t n_updates = (
            self.n_critic_update_batches_per_epoch()
            * self.n_trajectories_per_critic_update_batch()
        )

        if not self._trajectory_buffer.is_empty():
            for update_id in range(n_updates):
                trajectory = self._trajectory_buffer.next_shuffled_datum()
                approximator.batch_update(trajectory)

            # print(approximator.eval(trajectory[0]))
        system = self.super_system()
        system.prep_for_epoch()
