# distutils: language = c++
# distutils: extra_compile_args = -std=c++11
# distutils: sources = cpp_core/flat_network_approximator.cpp


cimport cython
from cpp_flat_critic cimport valarray
from cpp_flat_critic cimport Experience as CppExperienceDatum
from cpp_flat_critic cimport FlatNetworkApproximator as CppFlatNetworkApproximator
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

from goldenrockefeller.policyopt.map cimport BaseDifferentiableMap


import sys
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

cdef class NFlatNetwork(BaseDifferentiableMap):
    cdef shared_ptr[CppFlatNetwork] core

    def __init__(self, size_t n_in_dims, size_t n_hidden_units):
        self.core = make_shared[CppFlatNetwork](n_in_dims, n_hidden_units)

    cpdef NFlatNetwork copy(self, copy_obj = None):
        cdef NFlatNetwork new_network
        cdef unique_ptr[CppFlatNetwork] new_core

        if copy_obj is None:
            new_network = NFlatNetwork.__new__(NFlatNetwork)
        else:
            new_network = copy_obj

        new_core.swap(self.core.get().copy())
        new_network.core.reset(new_core.get())
        new_core.release()

        return new_network


    cpdef DoubleArray parameters(self):
        return DoubleArray_from_valarray(self.core.get().parameters())

    cpdef void set_parameters(self, parameters)  except *:
        self.core.get().set_parameters(valarray_from_DoubleArray(parameters))

    cpdef Py_ssize_t n_parameters(self) except *:
        return self.core.get().n_parameters()

    cpdef DoubleArray eval(self, input):
        cdef double res = self.core.get().eval(valarray_from_DoubleArray(input))
        cdef DoubleArray res_arr = new_DoubleArray(1)
        res_arr.view[0] = res
        return res_arr

    cpdef DoubleArray grad_wrt_parameters(self,  input, output_grad = None):
        if output_grad is None:
            raise NotImplementedError()
        return DoubleArray_from_valarray(self.core.get().grad_wrt_parameters(valarray_from_DoubleArray(input), output_grad.view[0]))


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

    cpdef void update_using_experience(self, ExperienceDatum experience, double target_value) except *:
        self.core.get().update(CppExperienceDatum_from_ExperienceDatum(experience), target_value)

    @property
    def learning_rate(self):
        return self.core.get().learning_rate

    @learning_rate.setter
    def learning_rate(self, double value):
        self.core.get().learning_rate = value

    @property
    def using_conditioner(self):
        return self.core.get().using_conditioner

    @using_conditioner.setter
    def using_conditioner(self, bint value):
        self.core.get().using_conditioner = value

    @property
    def grad_disturbance_factor(self):
        return self.core.get().grad_disturbance_factor

    @grad_disturbance_factor.setter
    def grad_disturbance_factor(self, double value):
        self.core.get().grad_disturbance_factor = value

    @property
    def momentum_sustain(self):
        return self.core.get().momentum_sustain

    @momentum_sustain.setter
    def momentum_sustain(self, double value):
        self.core.get().momentum_sustain = value

    @property
    def eps(self):
        return self.core.get().eps

    @eps.setter
    def eps(self, double value):
        self.core.get().eps = value

    @property
    def conditioner_time_horizon(self):
        return self.core.get().conditioner_time_horizon

    @conditioner_time_horizon.setter
    def conditioner_time_horizon(self, double value):
        self.core.get().conditioner_time_horizon = value

    @property
    def flat_network(self):
        cdef FlatNetwork f = FlatNetwork.__new__(FlatNetwork)

        f.core = self.core.get().flat_network

        return f



cdef class BaseCriticSystem(BaseSystem):
    cdef BaseSystem _super_system
    cdef _current_observation
    cdef _current_action
    cdef list _current_trajectory

    def __init__(
            self,
            BaseSystem super_system):
        self._super_system = super_system
        self._current_observation = None
        self._current_action = None
        self._current_trajectory = []

    cpdef action(self, observation):
        cdef object action

        action = self.super_system().action(observation)

        self._set_current_observation(observation)
        self._set_current_action(action)

        return action

    cpdef bint is_ready_for_evaluation(self) except *:
        cdef BaseSystem system

        system = self.super_system()

        return system.is_ready_for_evaluation()

    cpdef bint is_done_training(self) except *:
        return self.super_system().is_done_training()

    cpdef BaseCriticSystem copy(self, copy_obj = None):
        raise NotImplementedError("TODO")

    cpdef void receive_score(self, double score) except *:
        self.super_system().receive_score(score)

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        self.super_system().output_final_log(log_dirname, datetime_str)

    cpdef BaseSystem super_system(self):
        return self._super_system

    cpdef current_observation(self):
        return self._current_observation

    cpdef void _set_current_observation(self, observation) except *:
        self._current_observation = observation

    cpdef current_action(self):
         return self._current_action

    cpdef void _set_current_action(self, action) except *:
            self._current_action = action

    cpdef list current_trajectory(self):
        # type: (...) -> Sequence[ExperienceDatum]
        return self._current_trajectory


    @cython.locals(trajectory = list)
    cpdef void _set_current_trajectory(
            self,
            trajectory: List[ExperienceDatum]
            ) except *:
        self._current_trajectory = trajectory


cdef class FlatFitnessCriticSystem(BaseCriticSystem):
    cdef public Py_ssize_t n_critic_updates_per_epoch
    cdef public ShuffleBuffer experience_target_buffer
    cdef public FlatNetworkApproximator approximator

    def __init__(
            self,
            BaseSystem super_system,
            size_t n_in_dims, size_t n_hidden_units):
        self.approximator = FlatNetworkApproximator(n_in_dims, n_hidden_units)
        self.n_critic_updates_per_epoch = 1
        self.experience_target_buffer = new_ShuffleBuffer()
        BaseCriticSystem.__init__(self, super_system)


    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef FlatNetworkApproximator approximator
        cdef list current_trajectory

        approximator = self.approximator

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        self.current_trajectory().append(experience)

        new_feedback = approximator.eval(experience)
        #
        #
        # # if not isfinite(new_feedback):
        # #     raise RuntimeError("Something went wrong: feedback is not finite.")
        #
        self.super_system().receive_feedback(new_feedback)

    cpdef void update_policy(self) except *:
        cdef list current_trajectory = self.current_trajectory()
        self.extract_experience_targets(current_trajectory)
        self.super_system().update_policy()
        self._set_current_trajectory([])


    cpdef void extract_experience_targets(self, list trajectory) except *:
        cdef ExperienceDatum experience
        cdef double traj_eval
        cdef double step_eval
        cdef double sample_fitness
        cdef FlatNetworkApproximator approximator = self.approximator
        cdef double error
        cdef TargetEntry target_entry
        cdef Py_ssize_t traj_len = len(trajectory)

        sample_fitness = 0.
        traj_eval = 0.

        for experience in trajectory:
            sample_fitness += experience.reward
            traj_eval += approximator.eval(experience)

        traj_eval /= traj_len

        error = sample_fitness - traj_eval

        for experience in trajectory:
            step_eval = approximator.eval(experience)
            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = error + step_eval
            self.experience_target_buffer.add_staged_datum(target_entry)


    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t update_id
        cdef BaseSystem system
        cdef FlatNetworkApproximator approximator = self.approximator
        cdef list trajectory
        cdef Py_ssize_t n_updates = self.n_critic_updates_per_epoch
        cdef TargetEntry target_entry


        if not self.experience_target_buffer.is_empty():
            for update_id in range(n_updates):
                target_entry = self.experience_target_buffer.next_shuffled_datum()
                approximator.update_using_experience(target_entry.input, target_entry.target)

            # raise ValueError()
            # trajectory = self._trajectory_buffer.next_shuffled_datum()
            # print(approximator.eval(target_entry.input))
            # sys.stdout.flush()
            # raise ValueError()
            # print(len(trajectory))
        system = self.super_system()
        system.prep_for_epoch()


cdef class SteppedFlatFitnessCriticSystem(BaseCriticSystem):
    cdef public Py_ssize_t n_critic_updates_per_epoch
    cdef public list experience_target_buffers
    cdef public list approximators
    cdef public Py_ssize_t n_steps

    def __init__(
            self,
            BaseSystem super_system,
            size_t n_in_dims, size_t n_hidden_units, Py_ssize_t n_steps):
        self.approximators = [FlatNetworkApproximator(n_in_dims, n_hidden_units) for _ in range(n_steps)]
        self.n_critic_updates_per_epoch = 1
        self.experience_target_buffers = [new_ShuffleBuffer()  for _ in range(n_steps)]
        self.n_steps = n_steps
        BaseCriticSystem.__init__(self, super_system)


    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef double new_feedback
        cdef FlatNetworkApproximator approximator
        cdef list current_trajectory
        cdef Py_ssize_t step_id


        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        step_id = len(self.current_trajectory())

        self.current_trajectory().append(experience)

        approximator = self.approximators[step_id]

        new_feedback = approximator.eval(experience)
        #
        #
        # # if not isfinite(new_feedback):
        # #     raise RuntimeError("Something went wrong: feedback is not finite.")
        #
        self.super_system().receive_feedback(new_feedback)

    cpdef void update_policy(self) except *:
        cdef list current_trajectory = self.current_trajectory()
        self.extract_experience_targets(current_trajectory)
        self.super_system().update_policy()
        self._set_current_trajectory([])


    cpdef void extract_experience_targets(self, list trajectory) except *:
        cdef ExperienceDatum experience
        cdef double traj_eval
        cdef double step_eval
        cdef double sample_fitness
        cdef FlatNetworkApproximator approximator
        cdef double error
        cdef TargetEntry target_entry
        cdef Py_ssize_t traj_len = len(trajectory)
        cdef Py_ssize_t step_id
        cdef ShuffleBuffer experience_target_buffer

        sample_fitness = 0.
        traj_eval = 0.

        step_id = 0
        for experience in trajectory:
            sample_fitness += experience.reward
            approximator = self.approximators[step_id]
            traj_eval += approximator.eval(experience)
            step_id += 1

        traj_eval /= traj_len

        error = sample_fitness - traj_eval

        step_id = 0
        for experience in trajectory:
            approximator = self.approximators[step_id]
            step_eval = approximator.eval(experience)
            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = error + step_eval
            experience_target_buffer = self.experience_target_buffers[step_id]
            experience_target_buffer.add_staged_datum(target_entry)
            step_id += 1


    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t update_id
        cdef BaseSystem system
        cdef FlatNetworkApproximator approximator
        cdef list trajectory
        cdef Py_ssize_t n_updates = self.n_critic_updates_per_epoch
        cdef TargetEntry target_entry
        cdef Py_ssize_t step_id
        cdef ShuffleBuffer experience_target_buffer

        for step_id in range(self.n_steps):
            experience_target_buffer = self.experience_target_buffers[step_id]
            approximator = self.approximators[step_id]
            if not experience_target_buffer.is_empty():
                for update_id in range(n_updates):
                    target_entry = experience_target_buffer.next_shuffled_datum()
                    approximator.update_using_experience(target_entry.input, target_entry.target)

            # raise ValueError()
            # trajectory = self._trajectory_buffer.next_shuffled_datum()
            # print(approximator.eval(target_entry.input))
            # sys.stdout.flush()
            # raise ValueError()
            # print(len(trajectory))
        system = self.super_system()
        system.prep_for_epoch()

cdef class MonteFlatFitnessCriticSystem(FlatFitnessCriticSystem):

    cpdef void extract_experience_targets(self, list trajectory) except *:
        cdef ExperienceDatum experience
        cdef double sample_fitness
        cdef TargetEntry target_entry

        sample_fitness = 0.

        for experience in trajectory:
            sample_fitness += experience.reward

        for experience in trajectory:
            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = sample_fitness
            self.experience_target_buffer.add_staged_datum(target_entry)

cdef class QFlatFitnessCriticSystem(FlatFitnessCriticSystem):
    cpdef void extract_experience_targets(self, list trajectory) except *:
        q_extract_experience_targets(trajectory, self.experience_target_buffer, self.approximator)


cpdef void q_extract_experience_targets(list trajectory, ShuffleBuffer experience_target_buffer, BaseFunctionApproximator approximator) except *:
        cdef ExperienceDatum experience
        cdef ExperienceDatum next_experience
        cdef TargetEntry target_entry
        cdef double reward
        cdef double next_q
        cdef Py_ssize_t step_id
        cdef Py_ssize_t n_steps

        n_steps = len(trajectory)


        # Set Q value target for the last step in the trajectory.
        experience = trajectory[-1]
        reward = experience.reward

        target_entry = new_TargetEntry()
        target_entry.input = experience
        target_entry.target = experience.reward
        experience_target_buffer.add_staged_datum(target_entry)


        for step_id in range(n_steps - 1):
            experience = trajectory[step_id]
            reward = experience.reward

            next_experience = trajectory[step_id + 1]
            next_q = approximator.eval(next_experience)

            target_entry = new_TargetEntry()
            target_entry.input = experience
            target_entry.target = next_q + reward
            experience_target_buffer.add_staged_datum(target_entry)

cdef class UFlatFitnessCriticSystem(FlatFitnessCriticSystem):
    cpdef void extract_experience_targets(self, list trajectory) except *:
        u_extract_experience_targets(trajectory, self.experience_target_buffer, self.approximator)

cpdef void u_extract_experience_targets(list trajectory, ShuffleBuffer experience_target_buffer, BaseFunctionApproximator approximator) except *:
    cdef ExperienceDatum experience
    cdef ExperienceDatum next_experience
    cdef TargetEntry target_entry
    cdef double reward
    cdef double next_u
    cdef Py_ssize_t step_id
    cdef Py_ssize_t n_steps

    n_steps = len(trajectory)

    # Set U value target for the last step in the trajectory.
    experience = trajectory[0]
    experience = experience.copy()
    experience.action = new_DoubleArray(0)
    reward = 0.

    target_entry = new_TargetEntry()
    target_entry.input = experience
    target_entry.target = experience.reward
    experience_target_buffer.add_staged_datum(target_entry)


    for step_id in range(1, n_steps):
        experience = trajectory[step_id]
        experience = experience.copy()
        experience.action = new_DoubleArray(0)

        prev_experience = trajectory[step_id- 1]
        prev_experience = prev_experience.copy()
        prev_experience.action = new_DoubleArray(0)
        prev_reward = prev_experience.reward

        prev_u = approximator.eval(prev_experience)

        target_entry = new_TargetEntry()
        target_entry.input = experience
        target_entry.target = prev_u + prev_reward
        experience_target_buffer.add_staged_datum(target_entry)


cdef class UqFlatFitnessCriticSystem(BaseCriticSystem):
    cdef public Py_ssize_t n_critic_updates_per_epoch
    cdef public ShuffleBuffer q_experience_target_buffer
    cdef public ShuffleBuffer u_experience_target_buffer
    cdef public FlatNetworkApproximator q_approximator
    cdef public FlatNetworkApproximator u_approximator

    def __init__(
            self,
            BaseSystem super_system,
            size_t n_state_dims, size_t n_action_dims, size_t n_hidden_units):
        self.q_approximator = FlatNetworkApproximator(n_state_dims + n_action_dims, n_hidden_units)
        self.u_approximator = FlatNetworkApproximator(n_state_dims, n_hidden_units)
        self.n_critic_updates_per_epoch = 1
        self.q_experience_target_buffer = new_ShuffleBuffer()
        self.u_experience_target_buffer = new_ShuffleBuffer()

        BaseCriticSystem.__init__(self, super_system)

    #step_wise feedback
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef ExperienceDatum u_experience
        cdef double new_feedback
        cdef list current_trajectory


        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        u_experience = experience.copy()
        u_experience.action = new_DoubleArray(0)

        self.current_trajectory().append(experience)

        new_feedback = self.u_approximator.eval(u_experience) + self.q_approximator.eval(experience)
        #
        #
        # # if not isfinite(new_feedback):
        # #     raise RuntimeError("Something went wrong: feedback is not finite.")
        #
        self.super_system().receive_feedback(new_feedback)

    cpdef void update_policy(self) except *:
        cdef list current_trajectory = self.current_trajectory()
        self.extract_experience_targets(current_trajectory)
        self.super_system().update_policy()
        self._set_current_trajectory([])


    cpdef void extract_experience_targets(self, list trajectory) except *:
        q_extract_experience_targets(trajectory, self.q_experience_target_buffer, self.q_approximator)
        u_extract_experience_targets(trajectory, self.u_experience_target_buffer, self.u_approximator)

    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t update_id
        cdef BaseSystem system
        cdef list trajectory
        cdef Py_ssize_t n_updates
        cdef TargetEntry target_entry
        cdef FlatNetworkApproximator approximator


        n_updates = self.n_critic_updates_per_epoch

        approximator  = self.q_approximator
        if not self.q_experience_target_buffer.is_empty():
            for update_id in range(n_updates):
                target_entry = self.q_experience_target_buffer.next_shuffled_datum()
                approximator.update_using_experience(target_entry.input, target_entry.target)

        approximator  = self.u_approximator
        if not self.u_experience_target_buffer.is_empty():
            for update_id in range(n_updates):
                target_entry = self.u_experience_target_buffer.next_shuffled_datum()
                approximator.update_using_experience(target_entry.input, target_entry.target)

            # raise ValueError()
            # trajectory = self._trajectory_buffer.next_shuffled_datum()
            # print(approximator.eval(target_entry.input))
            # sys.stdout.flush()
            # raise ValueError()
            # print(len(trajectory))
        system = self.super_system()
        system.prep_for_epoch()




# cdef class DiscountFlatFitnessCriticSystem(FlatFitnessCriticSystem):
#     cdef double discount_factor
#
#     def __init__(
#             self,
#             BaseSystem super_system,
#             size_t n_in_dims, size_t n_hidden_units):
#         intermediate_critic = FlatNetworkApproximator(n_in_dims, n_hidden_units)
#         init_FitnessCriticSystem(self, super_system, intermediate_critic)
#         self.n_critic_updates_per_epoch = 1
#         self.experience_target_buffer = new_ShuffleBuffer()
#         self.discount_factor = 0.97
#
#
#     cpdef void extract_experience_targets(self, list trajectory) except *:
#         cdef ExperienceDatum experience, future_experience
#         cdef Py_ssize_t experience_id, future_experience_id
#         cdef double discounted_fitness
#         cdef TargetEntry target_entry
#
#         for experience_id, experience in enumerate(trajectory):
#             discounted_fitness = 0.
#             for future_experience_id, future_experience in enumerate(trajectory):
#                 if future_experience_id > experience_id:
#                     discounted_fitness += (
#                         future_experience.reward
#                         * self.discount_factor
#                         ** (future_experience_id - experience_id)
#                     )
#
#             target_entry = new_TargetEntry()
#             target_entry.input = experience
#             target_entry.target = discounted_fitness
#             self.experience_target_buffer.add_staged_datum(target_entry)
#
