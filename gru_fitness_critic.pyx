cimport cython
import datetime as dt
import os
import csv
import errno
from rockefeg.policyopt.fitness_critic cimport FitnessCriticSystem, init_FitnessCriticSystem

from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from rockefeg.policyopt.experience cimport ExperienceDatum, new_ExperienceDatum
from rockefeg.policyopt.function_approximation cimport BaseFunctionApproximator
from rockefeg.policyopt.system cimport BaseSystem

from rockefeg.policyopt.buffer cimport ShuffleBuffer, new_ShuffleBuffer
import numpy as np
import copy
import torch




cdef class GruApproximator(BaseFunctionApproximator):
    cdef Py_ssize_t n_in_dims
    cdef Py_ssize_t n_hidden_dims
    cdef public object model
    cdef public double learning_rate


    def __init__(self, n_in_dims, n_hidden_dims):
        self.n_in_dims = n_in_dims
        self.n_hidden_dims = n_hidden_dims
        self.model = torch.nn.GRU(input_size = n_in_dims, hidden_size = n_hidden_dims).double()
        self.learning_rate = 1.e-5

    # def copy(self, copy_obj = None):
    #     new_mlp = (
    #         TorchMlp(self.n_in_dims, self.n_hidden_neurons, self.n_out_dims) )
    #     new_mlp.model = copy.deepcopy(self.model)
    #     new_mlp.model.zero_grad()
    #     return new_mlp

    # def n_parameters(self):
    #     return sum(p.numel() for p in model.parameters())
    #
    # def parameters(self):
    #     parameters = torch.cat([param.view(-1) for param in self.model.parameters()])
    #     return DoubleArray(parameters.detach().numpy())
    #
    # def set_parameters(self, parameters):
    #     np_parameters = np.asarray(parameters.view)
    #     offset = 0
    #
    #     for param in self.model.parameters():
    #         param_view = param.view(-1)
    #
    #         param.view(-1).copy_(
    #             torch.from_numpy(
    #                 np_parameters[offset : offset + len(param_view)] ))
    #
    #         offset += len(param_view)


    cpdef eval(self, trajectory):
        cdef list cy_trajectroy = trajectory
        cdef ExperienceDatum experience
        cdef DoubleArray flattened_input
        cdef Py_ssize_t trajectory_len
        cdef Py_ssize_t id
        cdef Py_ssize_t input_id
        cdef object input_np
        cdef object input_torch
        cdef object output_torch
        cdef object output_np
        cdef object hidden

        # Get first experience.
        experience = trajectory[0]
        flattened_input = (
            new_DoubleArray(
                len(trajectory) *
                (
                    len(experience.observation)
                    + len(experience.action)
                )
            )
        )

        input_id = 0
        for experience in trajectory:

            for id in range(len(experience.observation)):
                flattened_input.view[input_id] = experience.observation.view[id]
                input_id += 1

            for id in range(len(experience.action)):
                flattened_input.view[input_id] = experience.action.view[id]
                input_id += 1

        input_np = (
            np.asarray(flattened_input.view).reshape(
                (
                    len(trajectory),
                    1,
                    len(experience.observation) + len(experience.action)
                )
            )
        )

        input_torch = torch.from_numpy(input_np)

        output_torch, hidden = self.model(input_torch)
        return output_torch



cdef class SumGruCriticSystem(FitnessCriticSystem):
    cdef public Py_ssize_t n_steps

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        self.n_steps = 0
        init_FitnessCriticSystem(self, super_system, intermediate_critic)

    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef BaseFunctionApproximator intermediate_critic
        current_trajectory: List[Experience]
        cdef BaseSystem system
        cdef object output_np

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)

        if len(current_trajectory) == self.n_steps:
            output_np = intermediate_critic.eval(current_trajectory).detach().numpy()
            system.receive_feedback(output_np[:, 0, 0].sum())
        else:
            system.receive_feedback(0.)
        pass

    @cython.locals(trajectory = list)
    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t batch_id
        cdef Py_ssize_t trajectory_id
        cdef Py_ssize_t n_trajectories_per_batch
        cdef Py_ssize_t n_batches
        cdef ExperienceDatum experience
        cdef ShuffleBuffer trajectory_buffer
        cdef ShuffleBuffer critic_target_buffer
        trajectory: List[ExperienceDatum]
        cdef GruApproximator intermediate_critic
        cdef BaseSystem system
        cdef double fitness
        cdef object loss
        cdef object output

        n_batches = (
            self.n_critic_update_batches_per_epoch())

        n_trajectories_per_batch = (
            self.n_trajectories_per_critic_update_batch())

        trajectory_buffer = self.trajectory_buffer()
        critic_target_buffer = self.critic_target_buffer()
        intermediate_critic = self.intermediate_critic()

        if not trajectory_buffer.is_empty():
            for batch_id in range(n_batches):
                for trajectory_id in range(n_trajectories_per_batch):
                    trajectory = trajectory_buffer.next_shuffled_datum()

                    fitness = 0.
                    for experience in trajectory:
                        fitness += experience.reward

                    output = intermediate_critic.eval(trajectory)

                    intermediate_critic.model.zero_grad()
                    loss = (output[:,0,0].sum() - fitness) ** 2
                    loss.backward()

                    with torch.no_grad():
                        for param in intermediate_critic.model.parameters():
                            param -= intermediate_critic.learning_rate * param.grad

            # print("Estimate: ", output[:,0,0].sum())


        system = self.super_system()
        system.prep_for_epoch()



cdef class FinalGruCriticSystem(FitnessCriticSystem):
    cdef public Py_ssize_t n_steps

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        self.n_steps = 0
        init_FitnessCriticSystem(self, super_system, intermediate_critic)

    @cython.locals(current_trajectory = list)
    cpdef void receive_feedback(self, feedback) except *:
        cdef ExperienceDatum experience
        cdef BaseFunctionApproximator intermediate_critic
        current_trajectory: List[Experience]
        cdef BaseSystem system
        cdef object output_np

        system = self.super_system()

        intermediate_critic = self.intermediate_critic()

        experience = new_ExperienceDatum()
        experience.observation = self.current_observation()
        experience.action = self.current_action()
        experience.reward = feedback

        current_trajectory = self.current_trajectory()
        current_trajectory.append(experience)



        if len(current_trajectory) == self.n_steps:
            output_np = intermediate_critic.eval(current_trajectory).detach().numpy()
            system.receive_feedback(output_np[-1, 0, 0])
        else:
            system.receive_feedback(0.)
        pass

    @cython.locals(trajectory = list)
    cpdef void prep_for_epoch(self) except *:
        cdef Py_ssize_t batch_id
        cdef Py_ssize_t trajectory_id
        cdef Py_ssize_t n_trajectories_per_batch
        cdef Py_ssize_t n_batches
        cdef ExperienceDatum experience
        cdef ShuffleBuffer trajectory_buffer
        cdef ShuffleBuffer critic_target_buffer
        trajectory: List[ExperienceDatum]
        cdef GruApproximator intermediate_critic
        cdef BaseSystem system
        cdef double fitness
        cdef object loss
        cdef object output

        n_batches = (
            self.n_critic_update_batches_per_epoch())

        n_trajectories_per_batch = (
            self.n_trajectories_per_critic_update_batch())

        trajectory_buffer = self.trajectory_buffer()
        critic_target_buffer = self.critic_target_buffer()
        intermediate_critic = self.intermediate_critic()

        if not trajectory_buffer.is_empty():
            for batch_id in range(n_batches):
                for trajectory_id in range(n_trajectories_per_batch):
                    trajectory = trajectory_buffer.next_shuffled_datum()

                    fitness = 0.
                    for experience in trajectory:
                        fitness += experience.reward

                    output = intermediate_critic.eval(trajectory)

                    intermediate_critic.model.zero_grad()
                    loss = (output[-1, 0, 0] - fitness) ** 2
                    loss.backward()

                    with torch.no_grad():
                        for param in intermediate_critic.model.parameters():
                            param -= intermediate_critic.learning_rate * param.grad

            # print("Estimate: ", output[-1, 0, 0])


        system = self.super_system()
        system.prep_for_epoch()



cdef class RecordingSumGruCriticSystem(SumGruCriticSystem):
    cdef list estimates

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        FinalGruCriticSystem.__init__(self, super_system, intermediate_critic)
        self.estimates = []

    @cython.locals(trajectory = list)
    cpdef void receive_score(self, double score) except *:


        cdef object output
        cdef ShuffleBuffer trajectory_buffer
        cdef BaseFunctionApproximator intermediate_critic
        trajectory: List[ExperienceDatum]

        intermediate_critic = self.intermediate_critic()
        trajectory_buffer = self.trajectory_buffer()


        if not trajectory_buffer.is_empty():
            trajectory = trajectory_buffer.next_shuffled_datum()

            output = intermediate_critic.eval(trajectory)

        self.estimates.append(output[:,0,0].detach().numpy().sum())

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



cdef class RecordingFinalGruCriticSystem(FinalGruCriticSystem):
    cdef list estimates

    def __init__(
            self,
            BaseSystem super_system,
            BaseFunctionApproximator intermediate_critic):
        FinalGruCriticSystem.__init__(self, super_system, intermediate_critic)
        self.estimates = []

    @cython.locals(trajectory = list)
    cpdef void receive_score(self, double score) except *:


        cdef object output
        cdef ShuffleBuffer trajectory_buffer
        cdef BaseFunctionApproximator intermediate_critic
        trajectory: List[ExperienceDatum]

        intermediate_critic = self.intermediate_critic()
        trajectory_buffer = self.trajectory_buffer()


        if not trajectory_buffer.is_empty():
            trajectory = trajectory_buffer.next_shuffled_datum()

            output = intermediate_critic.eval(trajectory)

        self.estimates.append(output.detach().numpy()[-1, 0, 0])

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

