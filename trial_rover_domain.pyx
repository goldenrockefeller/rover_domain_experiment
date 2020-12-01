cimport cython

from rockefeg.policyopt.domain cimport BaseDomain
from rockefeg.roverdomain.rover_domain cimport RoverDomain
from rockefeg.roverdomain.state cimport State, RoverDatum, PoiDatum
from rockefeg.roverdomain.state cimport new_RoverDatum, new_PoiDatum
from rockefeg.cyutil.array cimport DoubleArray

from rockefeg.roverdomain.rover_domain import RoverDomain

import numpy as np

from typing import List, Sequence

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class TrialRoverDomain(BaseDomain):
    cdef public RoverDomain super_domain
    cdef public double setup_size
    cdef public double poi_init_thickness
    cdef public object poi_value_init_type
    cdef public double base_poi_value

    def __init__(self):
        self.super_domain = RoverDomain()
        self.setup_size = 0.
        self.poi_init_thickness = 0.
        self.poi_value_init_type = "sequential"

    @cython.locals(rover_data=list)
    cpdef void set_n_rovers(self, Py_ssize_t n_rovers) except *:
        cdef State setting_state
        cdef Py_ssize_t i
        rover_data: List[RoverDatum]

        setting_state = self.super_domain.setting_state()

        rover_data = [None] * n_rovers

        for i in range(n_rovers):
            rover_data[i] = new_RoverDatum()


        setting_state.set_rover_data(rover_data)

    @cython.locals(poi_data=list)
    cpdef void set_n_pois(self, Py_ssize_t n_pois) except *:
        cdef State setting_state
        cdef Py_ssize_t i
        poi_data: List[PoiDatum]

        setting_state = self.super_domain.setting_state()

        poi_data = [None] * n_pois

        for i in range(n_pois):
            poi_data[i] = new_PoiDatum()

        setting_state.set_poi_data(poi_data)

    cpdef TrialRoverDomain copy(self, copy_obj = None):
        raise NotImplementedError()

    cpdef randomize_setting_state(self):
        cdef State setting_state
        cdef Py_ssize_t poi_id
        cdef RoverDatum rover_datum
        cdef PoiDatum poi_datum
        cdef double angle, min_radius, max_radius, radius
        cdef object poi_position, rover_position

        setting_state = self.super_domain.setting_state()

        # Place POIs radially.
        poi_id = 0
        for poi_datum in setting_state.poi_data():
            poi_position = np.zeros((2))
            angle = np.random.uniform(-np.pi, np.pi)
            min_radius = (1. - self.poi_init_thickness) * 0.5 * self.setup_size
            max_radius =  0.5 * self.setup_size
            radius = np.random.uniform(min_radius, max_radius)
            poi_position[0] = radius*np.cos(angle) + 0.5*self.setup_size
            poi_position[1] = radius*np.sin(angle) + 0.5*self.setup_size
            poi_datum.set_position_x(poi_position[0])
            poi_datum.set_position_y(poi_position[1])

            # Set POI values as the 1, 2, 3, ... sequence.
            if self.poi_value_init_type  == "sequential":
                poi_datum.set_value(self.base_poi_value * (poi_id + 1.))
            elif self.poi_value_init_type == "same":
                poi_datum.set_value(self.base_poi_value)
            else:
                raise ValueError(
                    "POI value initialization type (poi_value_init_type) not "
                    + "recognized. A value of %s was received."
                    % self.poi_value_init_type)
            poi_id += 1


        for rover_datum in setting_state.rover_data():
            # Place rovers radially.
            rover_position = np.zeros((2))
            angle = np.random.uniform(-np.pi, np.pi)
            radius = np.random.uniform(0., .05 * self.setup_size)
            rover_position[0] = radius*np.cos(angle) + 0.5*self.setup_size
            rover_position[1] = radius*np.sin(angle) + 0.5*self.setup_size
            rover_datum.set_position_x(rover_position[0])
            rover_datum.set_position_y(rover_position[1])


            # Orient rovers randomly.
            angle = np.random.uniform(-np.pi, np.pi)
            rover_datum.set_direction(angle)

    cpdef void prep_for_epoch(self) except *:
        self.randomize_setting_state()
        self.super_domain.reset()

    cpdef void reset_for_training(self) except *:
        self.super_domain.reset()

    @cython.locals(joint_observation = list)
    cpdef observation(self):
        joint_observation: List[DoubleArray]

        joint_observation = self.super_domain.rover_observations()

        return joint_observation.copy()

    cpdef void step(self, actions: Sequence[DoubleArray]) except *:
        self.super_domain.step(actions)

    cpdef feedback(self):
        cdef list feedback
        cdef DoubleArray rover_evals
        cdef Py_ssize_t rover_id

        rover_evals = self.super_domain.rover_evals()

        feedback = [None] * len(rover_evals)

        for rover_id in range(len(rover_evals)):
            feedback[rover_id] = rover_evals.view[rover_id]

        return feedback

    cpdef void reset_for_evaluation(self) except *:
        self.randomize_setting_state()
        self.super_domain.reset()

    cpdef bint episode_is_done(self) except *:
        return self.super_domain.episode_is_done()

    cpdef double score(self) except *:
        return self.super_domain.eval()

    cpdef void output_final_log(self, log_dirname, datetime_str) except *:
        pass