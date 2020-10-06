# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -mfma -ftree-vectorize

cimport cython
from rockefeg.policyopt.function_approximation cimport BaseFunctionApproximator, TargetEntry
from rockefeg.policyopt.neural_network cimport Rbfn
from rockefeg.policyopt.neural_network cimport normalization_for_DoubleArray
from rockefeg.policyopt.neural_network cimport rbfn_pre_norm_activations_eval
from rockefeg.policyopt.experience cimport ExperienceDatum
from rockefeg.cyutil.typed_list cimport TypedList
from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from rockefeg.policyopt.map cimport BaseDifferentiableMap
from rbfn_random cimport random_normal

from libcpp.vector cimport vector

cdef extern from "<valarray>" namespace "std" nogil:
    cdef cppclass valarray[T]:
        valarray() except +
        void resize (size_t) except +
        size_t size() const
        valarray operator-() const
        valarray operator= (const valarray&)
        T& operator[] (size_t)
        T sum() const

        valarray operator* (const valarray&, const valarray&)
        valarray operator* (const T&, const valarray&)
        valarray operator* (const valarray&, const T&)
        valarray operator/ (const valarray&, const valarray&)
        valarray operator/ (const T&, const valarray&)
        valarray operator/ (const valarray&, const T&)
        valarray operator+ (const valarray&, const valarray&)
        valarray operator+ (const T&, const valarray&)
        valarray operator+ (const valarray&, const T&)
        valarray operator- (const valarray&, const valarray&)
        valarray operator- (const T&, const valarray&)
        valarray operator- (const valarray&, const T&)
    valarray[T] tanh[T] (const valarray[T]&)
    valarray[T] sqrt[T] (const valarray[T]&)


cdef class RbfnApproximator(BaseFunctionApproximator):
    cdef Rbfn rbfn
    cdef valarray[double] counters
    # TODO only work with one dimensional output, extend to multiple dimensions in the future
    cdef valarray[double] uncertainties
    cdef public double scale_multiplier
    cdef public double discount_factor
    cdef public double exploration_incentive_factor
    cdef public double exploration_sampling_factor
    cdef public double process_uncertainty_rate
    cdef public double center_relocalization_rate
    cdef public double epsilon

    def __init__(self, Rbfn rbfn):
        cdef size_t center_id
        cdef size_t i

        if rbfn.transform.size() != 1:
            raise (
                ValueError(
                    "The number of output dimension "
                    "(rbfn.shape()[2] = {n_rbfn_output_dims}) must be 1."
                    .format(**locals()) ))

        if rbfn.scalings_are_fixed is True:
            raise (
                ValueError(
                    "The value of whether the radial basis function network's "
                    "scaling values are fixed "
                    "(rbfn.scalings_are_fixed = {rbfn.scalings_are_fixed}) "
                    "must be False."
                    .format(**locals()) ))

        if rbfn.normalizes_activations is False:
            raise (
                ValueError(
                    "The value of whether the radial basis function network "
                    "normalizes its activations "
                    "(rbfn.scalings_are_fixed = {rbfn.scalings_are_fixed}) "
                    "must be True."
                    .format(**locals()) ))

        self.rbfn = rbfn

        self.counters.resize(rbfn.centers.size())
        assign_valarray_to_double(self.counters, 0.)

        self.uncertainties.resize(rbfn.centers.size())
        assign_valarray_to_double(self.uncertainties, 0.)

        self.scale_multiplier = 1.
        self.discount_factor = 1.
        self.exploration_incentive_factor = 1.
        self.exploration_sampling_factor = 1.
        self.process_uncertainty_rate = 0.
        self.center_relocalization_rate = 1.
        self.epsilon = 1e-9

    def __reduce__(self):
        cdef DoubleArray counters
        cdef DoubleArray uncertainties
        cdef Py_ssize_t center_id

        counters = DoubleArray_from_valarray(self.counters)

        uncertainties = DoubleArray_from_valarray(self.uncertainties)

        return (
            unpickle_RbfnApproximator,
            (
            self.rbfn,
            counters,
            uncertainties,
            self.scale_multiplier,
            self.discount_factor,
            self.exploration_incentive_factor,
            self.exploration_sampling_factor,
            self.process_uncertainty_rate,
            self.center_relocalization_rate,
            self.epsilon
            ) )


    cpdef copy(self, copy_obj = None):
        cdef Rbfn new_approximator
        if copy_obj is None:
            new_approximator = (
                Rbfn.__new__(
                    Rbfn))
        else:
            new_approximator = copy_obj

        # TODO
        raise NotImplementedError("TODO")
        #
        # return new_approximator

    cpdef void set_center_locations(self, locations):
        # TODO locations is a numpy matrix, need to use something faster.

        cdef size_t center_id
        cdef size_t item_id
        cdef double val


        for center_id in range(self.rbfn.centers.size()):
            for item_id in range(self.rbfn.centers[0].size()):
                val = locations[center_id, item_id]
                self.rbfn.centers[center_id][item_id] = val



    cpdef void set_center_scalings(self, scalings):
        # TODO locations is a numpy matrix, need to use something faster.

        cdef size_t center_id
        cdef size_t item_id
        cdef double val

        for center_id in range(self.rbfn.scalings.size()):
            for item_id in range(self.rbfn.scalings[0].size()):
                val =  scalings[center_id, item_id]
                self.rbfn.scalings[center_id][item_id] = val



    cpdef void set_uncertainties(self, DoubleArray uncertainties) except *:
        cdef Py_ssize_t self_n_uncertainties
        cdef Py_ssize_t new_n_uncertainties
        cdef Py_ssize_t id

        self_n_uncertainties = <Py_ssize_t>(self.uncertainties.size())
        new_n_uncertainties = len(uncertainties)

        if self_n_uncertainties != new_n_uncertainties:
            raise (
                ValueError(
                    "The number of uncertainty values provided "
                    "(len(uncertainties) = {new_n_uncertainties}) must "
                    "equal the current number of uncertainty values "
                    "([in cython]self.uncertainties.size() = "
                    "{self_n_uncertainties})."
                    .format(**locals()) ))

        for id in range(self_n_uncertainties):
            self.uncertainties[<size_t>id] = uncertainties.view[id]

    cpdef void set_values(self, DoubleArray values) except *:
        cdef Rbfn rbfn
        cdef Py_ssize_t rbfn_n_values
        cdef Py_ssize_t new_n_values
        cdef Py_ssize_t id

        rbfn = self.rbfn

        rbfn_n_values = <Py_ssize_t>(rbfn.transform[0].size())
        new_n_values = len(values)

        if rbfn_n_values != new_n_values:
            raise (
                ValueError(
                    "The number of center output values provided "
                    "(len(values) = {new_n_values}) must "
                    "equal the current number of center output values "
                    "([in cython]self.values.size() = "
                    "{rbfn_n_values})."
                    .format(**locals()) ))

        for id in range(rbfn_n_values):
            rbfn.transform[0][<size_t>id] = values.view[id]

    cpdef void set_counters(self, DoubleArray counters) except *:
        cdef Py_ssize_t self_n_counters
        cdef Py_ssize_t new_n_counters
        cdef Py_ssize_t id

        self_n_counters = <Py_ssize_t>(self.counters.size())
        new_n_counters = len(counters)

        if self_n_counters != new_n_counters:
            raise (
                ValueError(
                    "The number of counters provided "
                    "(len(counters) = {new_n_counters}) must "
                    "equal the current number of counters "
                    "([in cython]self.counters.size() = "
                    "{self_n_counters})."
                    .format(**locals()) ))

        for id in range(self_n_counters):
            self.counters[<size_t>id] = counters.view[id]


    cpdef parameters(self):
        # TODO
        raise NotImplementedError("TODO")

    cpdef void set_parameters(self, parameters) except *:
        # TODO
        raise NotImplementedError("TODO")

    cpdef Py_ssize_t n_parameters(self) except *:
        # TODO
        raise NotImplementedError("TODO")

    cpdef eval(self, raw_input):
        cdef DoubleArray input
        cdef DoubleArray value_eval
        cdef DoubleArray eval
        cdef valarray[double] normalized_activations
        cdef valarray[double] random_output_deviations
        cdef double uncertainty_eval

        input = concatenate_state_action(raw_input)

        value_eval = self.rbfn.eval(input)

        # Get uncertainty evaluation.
        normalized_activations = (
            valarray_from_DoubleArray(
                normalization_for_DoubleArray(
                    rbfn_pre_norm_activations_eval(
                        self.rbfn,
                        input ) ) ) )
        #
        random_output_deviations = (
            (self.exploration_incentive_factor
                + normal_valarray(self.counters.size())
                * self.exploration_sampling_factor)
            * sqrt(self.uncertainties)
            / (sqrt(self.counters) + self.epsilon) )

        uncertainty_eval = (
            random_output_deviations * normalized_activations).sum()

        eval = value_eval.copy()
        eval.view[0] += uncertainty_eval

        return eval

    cpdef void batch_update(self, entries) except *:
        cdef TypedList cy_entries
        cdef Rbfn rbfn
        cdef double error
        cdef double fitness_estimate
        cdef double fitness_estimate_intermediate
        cdef double trajectory_size
        cdef double activations_sum
        cdef double learning_rate
        cdef TargetEntry entry
        cdef list inputs
        cdef DoubleArray fitness_target
        cdef DoubleArray eval
        cdef valarray[double] values
        cdef valarray[double] center_update
        cdef valarray[double] center_location
        cdef valarray[double] center_variances_update
        cdef valarray[double] center_variances
        cdef valarray[double] values_update
        cdef valarray[double] unnormalized_activations
        cdef valarray[double] local_activations
        cdef valarray[double] center_participations
        cdef valarray[double] center_weak_participations
        cdef valarray[double] local_variances
        cdef valarray[double] separation
        cdef valarray[double] centers_update_partial
        cdef vector[valarray[double]] normalized_activations
        cdef vector[valarray[double]] variances
        cdef vector[valarray[double]] centers_update_sum
        cdef vector[valarray[double]] center_variances_update_sum
        cdef size_t n_centers
        cdef size_t n_input_dims
        cdef size_t center_id
        cdef size_t entry_id

        # TODO error checking

        trajectory_size = <double>len(entries)



        # Get Fitness Estimate.
        fitness_estimate = 0.
        for entry in entries:
            input = concatenate_state_action(entry.input)
            eval = self.rbfn.eval(input)
            fitness_estimate += eval.view[0] / trajectory_size

        # Get fitness target and error.
        cy_entries = entries
        entry = cy_entries.item(0)
        fitness_target = entry.target
        error = (fitness_target.view[0] - fitness_estimate)

        # Initialize local variables and arrays.
        rbfn = self.rbfn
        #
        n_centers = rbfn.centers.size()
        n_input_dims = rbfn.centers[0].size()
        #
        centers_update_sum.resize(n_centers)
        for center_id in range(n_centers):
            centers_update_sum[center_id].resize(n_input_dims)
            assign_valarray_to_double(centers_update_sum[center_id], 0.)

        #
        center_variances_update_sum.resize(n_centers)
        for center_id in range(n_centers):
            center_variances_update_sum[center_id].resize(n_input_dims)
            assign_valarray_to_double(
                center_variances_update_sum[center_id]
                , 0. )
        #

        inputs = [None] * len(entries)

        # Decay counter.
        self.counters = self.discount_factor * self.counters

        # Get inputs
        entry_id = 0
        for entry in entries:
            inputs[<Py_ssize_t> entry_id] = (
                concatenate_state_action(entry.input))
            entry_id += 1

        # Get center participation and normalized activations.
        entry_id = 0
        center_participations.resize(n_centers)
        assign_valarray_to_double(center_participations, 0.)
        center_weak_participations.resize(n_centers)
        assign_valarray_to_double(center_weak_participations, 0.)
        normalized_activations.resize(<size_t>len(entries))
        for entry in entries:
            unnormalized_activations = (
                valarray_from_DoubleArray(
                    rbfn_pre_norm_activations_eval(
                        self.rbfn,
                        inputs[<Py_ssize_t> entry_id] ) ) )
            activations_sum = unnormalized_activations.sum()
            if activations_sum == 0.:
                normalized_activations[entry_id].resize(n_centers)
                assign_valarray_to_double(
                    normalized_activations[entry_id],
                    1. / <double>n_centers )
            else:
                normalized_activations[entry_id] = (
                    unnormalized_activations
                    / activations_sum )
            center_participations = (
                center_participations
                + normalized_activations[entry_id])
            center_weak_participations = (
                center_weak_participations
                + normalized_activations[entry_id])
            entry_id += 1
        #
        center_participations = center_participations / trajectory_size

        center_weak_participations = center_weak_participations / trajectory_size

        # Update Center Values and Uncertainties
        values_update = (
            error
            * center_participations
            / (self.counters + center_participations) )

        values = rbfn.transform[0]
        values = values + values_update
        rbfn.transform[0] = values
        #
        self.uncertainties = (
            (self.counters
                * self.uncertainties
                + error
                * (error - values_update)
                * center_participations )
            / (self.counters + center_participations)
            + self.process_uncertainty_rate)

        # # Store previous variances now, so I can store local scalings when
        # # updating the center locations and variances.
        # variances.resize(n_centers)
        # for center_id in range(n_centers):
        #     variances[center_id] = (
        #         1.
        #         / ((rbfn.scalings[center_id]
        #             * rbfn.scalings[center_id])
        #             + self.epsilon ) )

        # # Update Center Locations and Variances
        # entry_id = 0
        # local_variances.resize(n_input_dims)
        # for entry in entries:
        #     # Get local scaling
        #     assign_valarray_to_double(local_variances, 0.)
        #     for center_id in range(n_centers):
        #         local_variances = (
        #             local_variances
        #             + normalized_activations[entry_id][center_id]
        #             * variances[center_id])
        #
        #     # Set local scalings
        #     for center_id in range(n_centers):
        #         rbfn.scalings[center_id] = (
        #             1.
        #             / (sqrt(local_variances))
        #             + self.epsilon )
        #
        #     # Get local activations
        #     unnormalized_activations = (
        #         valarray_from_DoubleArray(
        #             rbfn_pre_norm_activations_eval(
        #                 self.rbfn,
        #                 inputs[<Py_ssize_t> entry_id] ) ) )
        #     activations_sum = unnormalized_activations.sum()
        #     if activations_sum == 0.:
        #         assign_valarray_to_double(
        #             local_activations,
        #             1. / <double>n_centers )
        #     else:
        #         local_activations = (
        #             unnormalized_activations
        #             / activations_sum )
        #
        #     # Accumulate partial center location and variance updates with
        #     # respect to the current entry.
        #     for center_id in range(n_centers):
        #         center_location = rbfn.centers[center_id]
        #         separation = (
        #             valarray_from_DoubleArray(inputs[<Py_ssize_t> entry_id])
        #             - center_location )
        #
        #         centers_update_partial = (
        #             separation
        #             * local_activations[center_id]
        #             / (local_activations[center_id]
        #                 + self.counters[center_id]) )
        #
        #         centers_update_sum[center_id] = (
        #             centers_update_sum[center_id]
        #             + centers_update_partial)
        #
        #         center_variances_update_sum[center_id] = (
        #             center_variances_update_sum[center_id]
        #             + (self.counters
        #                 * variances[center_id]
        #                 + separation
        #                 * (separation - centers_update_partial)
        #                 * local_activations[center_id] )
        #             - variances[center_id] )
        #
        #     entry_id += 1
        #
        # # Actually update the center variances and locations using the
        # # cumulative batch updates.
        # for center_id in range(n_centers):
        #     learning_rate = self.center_relocalization_rate/trajectory_size
        #
        #     center_location = rbfn.centers[center_id]
        #     center_update = centers_update_sum[center_id]
        #     rbfn.centers[center_id] = (
        #         center_location
        #         + center_update
        #         * learning_rate )
        #
        #     center_variances = variances[center_id]
        #     center_variances_update = center_variances_update_sum[center_id]
        #
        #     variances[center_id]  = (
        #         center_variances
        #         + learning_rate
        #         * center_variances_update)
        #
        #     rbfn.scalings[center_id] = (
        #         1.
        #         / (sqrt(variances[center_id]) + self.epsilon) )

        # Increment counters.
        self.counters = self.counters + center_participations


def unpickle_RbfnApproximator(
        rbfn,
        counters,
        uncertainties,
        scale_multiplier,
        discount_factor,
        exploration_incentive_factor,
        exploration_sampling_factor,
        process_uncertainty_rate,
        center_relocalization_rate,
        epsilon
        ):
    # TODO need error checking here.
    approximator = RbfnApproximator(rbfn)

    approximator.set_counters(counters)
    approximator.set_uncertainties(uncertainties)

    approximator.scale_multiplier = scale_multiplier
    approximator.discount_factor = discount_factor
    approximator.exploration_incentive_factor = exploration_incentive_factor
    approximator.exploration_sampling_factor = exploration_sampling_factor
    approximator.process_uncertainty_rate = process_uncertainty_rate
    approximator.center_relocalization_rate = center_relocalization_rate
    approximator.epsilon = epsilon

    return approximator

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

cdef valarray[double] normal_valarray(size_t size) except *:
    cdef valarray[double] arr
    cdef Py_ssize_t id

    arr.resize(size)

    for id in range(size):
        arr[id] = random_normal()

    return arr

cpdef DoubleArray concatenate_state_action(raw_input):
    cdef ExperienceDatum input
    cdef DoubleArray state
    cdef DoubleArray action
    cdef DoubleArray state_action
    cdef Py_ssize_t n_state_dims
    cdef Py_ssize_t n_action_dims
    cdef Py_ssize_t id

    if isinstance(raw_input, DoubleArray):
        return raw_input

    input = <ExperienceDatum?> raw_input

    state = <DoubleArray?>input.state
    action = <DoubleArray?>input.action

    n_state_dims = len(state)
    n_action_dims = len(action)
    state_action = new_DoubleArray(n_state_dims + n_action_dims)

    for id in range(n_state_dims):
        state_action.view[id] = state.view[id]

    for id in range(n_action_dims):
        state_action.view[id + n_state_dims] = action.view[id]


    return state_action

cdef inline void assign_valarray_to_double(
        valarray[double] arr,
        double val
        ) except *:
    cdef size_t id

    for id in range(arr.size()):
        arr[id] = val


