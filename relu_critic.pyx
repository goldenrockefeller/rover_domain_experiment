# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = cpp_core/relu_network_approximator.cpp
# cython: warn.undeclared = True

from cpp_relu_approximator cimport valarray
from cpp_relu_approximator cimport ReluNetworkApproximator as CppReluNetworkApproximator
cimport cython
from libcpp.memory cimport shared_ptr, make_shared
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from rockefeg.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry, new_TargetEntry
from rockefeg.policyopt.experience cimport ExperienceDatum
from rockefeg.policyopt.buffer cimport ShuffleBuffer, new_ShuffleBuffer
from rockefeg.policyopt.fitness_critic cimport FitnessCriticSystem, init_FitnessCriticSystem
from rockefeg.policyopt.system cimport BaseSystem

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

    new_arr = new_DoubleArray(<Py_ssize_t>arr.size())

    for id in range(len(new_arr)):
        new_arr.view[id] = arr[<size_t>id]

    return new_arr

cpdef DoubleArray concatenate_observation_action(input):
    cdef ExperienceDatum cy_input
    cdef DoubleArray observation
    cdef DoubleArray action
    cdef DoubleArray observation_action
    cdef Py_ssize_t n_observation_dims
    cdef Py_ssize_t n_action_dims
    cdef Py_ssize_t id

    if isinstance(input, DoubleArray):
        return input

    cy_input = input


    observation = cy_input.observation
    action = cy_input.action

    n_observation_dims = len(observation)
    n_action_dims = len(action)
    observation_action = new_DoubleArray(n_observation_dims + n_action_dims)

    for id in range(n_observation_dims):
        observation_action.view[id] = observation.view[id]

    for id in range(n_action_dims):
        observation_action.view[id + n_observation_dims] = action.view[id]

    return observation_action


cdef class ReluNetworkApproximator(BaseFunctionApproximator):
    cdef shared_ptr[CppReluNetworkApproximator] core

    def __init__(
        self,
        Py_ssize_t n_in_dims,
        Py_ssize_t n_hidden_units,
        Py_ssize_t n_out_dims,
    ):

        self.core = make_shared[CppReluNetworkApproximator](n_in_dims, n_hidden_units, n_out_dims)

    cpdef double learning_rate(self) except *:
        return self.core.get().learning_rate


    cpdef void set_learning_rate(self, double learning_rate) except *:
        self.core.get().learning_rate = learning_rate

    cpdef DoubleArray eval(self, input):
        return (
            DoubleArray_from_valarray(
                self.core.get().eval(
                    valarray_from_DoubleArray(
                        concatenate_observation_action(
                            input )))))

    @cython.locals(trajectory = list)
    cpdef void update(self, TargetEntry target_entry) except *:
        cdef DoubleArray input = target_entry.input
        cdef DoubleArray target = target_entry.target


        self.core.get().update(
            valarray_from_DoubleArray(input),
            DoubleArray_from_valarray(
                self.core.get().eval(
                    valarray_from_DoubleArray(
                            input ))) .view[0]
            - target.view[0]
        )


cdef class ReluFitnessCriticSystem(FitnessCriticSystem):
    cdef public Py_ssize_t n_updates_per_epoch
    cdef public ShuffleBuffer entry_buffer

    def __init__(
            self,
            BaseSystem super_system,
            ReluNetworkApproximator intermediate_critic):
        init_FitnessCriticSystem(self, super_system, intermediate_critic)
        self.n_updates_per_epoch = 0
        self.entry_buffer = ShuffleBuffer()


    @cython.locals(trajectory = list, target_entries = list)
    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t update_id
        cdef BaseSystem system
        cdef TargetEntry entry
        cdef ReluNetworkApproximator approximator = self.intermediate_critic()

        if not self.entry_buffer.is_empty():
            for update_id in range(self.n_updates_per_epoch):
                entry = self.entry_buffer.next_shuffled_datum()
                approximator.update(entry)

            # print(approximator.eval(entry.input).view[0])
        system = self.super_system()
        system.prep_for_epoch()

    cpdef void update_policy(self) except *:
        cdef ExperienceDatum experience
        cdef DoubleArray observation_action
        cdef DoubleArray feedback
        cdef TargetEntry entry

        self.super_system().update_policy()
        self.trajectory_buffer().add_staged_datum(self.current_trajectory())

        feedback = new_DoubleArray(1)
        feedback.view[0] = 0.
        for experience in self.current_trajectory():
            feedback.view[0] += experience.reward

        for experience in self.current_trajectory():
            observation_action = concatenate_observation_action(experience)

            entry = new_TargetEntry()
            entry.input = observation_action
            entry.target = feedback

            self.entry_buffer.add_staged_datum(entry)


        self._set_current_trajectory([])