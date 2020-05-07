cimport cython

from rockefeg.policyopt.evolution cimport DefaultPhenotype
from rockefeg.policyopt.evolution cimport init_DefaultPhenotype
from rockefeg.policyopt.evolution cimport DefaultEvolvingSystem
from rockefeg.policyopt.evolution cimport init_DefaultEvolvingSystem

from rockefeg.cyutil.array cimport DoubleArray, new_DoubleArray
from rockefeg.cyutil.array import DoubleArray

from libc.math cimport INFINITY, sqrt

import numpy as np
import random

cdef object random_shuffle = random.shuffle
cdef object np_random_uniform = np.random.uniform
cdef object np_random_normal = np.random.normal
cdef object np_linalg_norm = np.linalg.norm

cdef double DIFF_RANDOM_SPREAD_FACTOR = 0.1

cpdef enum FeedbackType:
    fitness
    novelty_score

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class PhenotypeWithNoveltyScore(DefaultPhenotype):
    cdef public double novelty_score

    def __init__(self, policy):
        init_DefaultPhenotype(self, policy)
        self.novelty_score = -INFINITY

    cpdef copy(self, copy_obj = None):
        cdef PhenotypeWithNoveltyScore new_phenotype

        if copy_obj is None:
            new_phenotype = (
                PhenotypeWithNoveltyScore.__new__(
                    PhenotypeWithNoveltyScore))
        else:
            new_phenotype = copy_obj

        DefaultPhenotype.copy(self, new_phenotype)
        new_phenotype.novelty_score = -INFINITY

        return new_phenotype

    cpdef void receive_feedback(self, feedback) except *:
        cdef tuple cy_feedback = feedback
        cdef FeedbackType feedback_type = cy_feedback[0]
        cdef double feedback_value = cy_feedback[1]

        if feedback_type is FeedbackType.fitness:
            DefaultPhenotype.receive_feedback(self, feedback_value)

        elif feedback_type is FeedbackType.novelty_score:
            self.novelty_score = max(self.novelty_score, feedback_value)

    cpdef void prep_for_epoch(self) except *:
        DefaultPhenotype.prep_for_epoch(self)
        self.novelty_score = -INFINITY

@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class NoveltyNode:
    cdef public DoubleArray loc
    cdef public Py_ssize_t n_nearby_state_actions
    cdef public NoveltyNode closest_other_node
    cdef public double sqr_dist_to_closest_other_node

    def __init__(self, loc):
        cdef DoubleArray cy_loc = <DoubleArray?> loc
        self.loc = cy_loc.copy()
        self.n_nearby_state_actions = 0
        self.closest_other_node = None

    cpdef copy(self, copy_obj = None):
        cdef NoveltyNode new_node

        if copy_obj is None:
            new_node = NoveltyNode.__new__(NoveltyNode)
        else:
            new_node = copy_obj

        new_node.loc = self.loc.copy()
        new_node.n_nearby_state_actions = 0
        new_node.closest_other_node = None

        return new_node



@cython.warn.undeclared(True)
@cython.auto_pickle(True)
cdef class EvolvingSystemWithNoveltySearch(DefaultEvolvingSystem):
    cdef public list novelty_nodes
    cdef public bint using_action

    def __init__(self, n_state_action_dims, n_novelty_nodes, using_action):
        cdef Py_ssize_t cy_n_novelty_nodes = n_novelty_nodes
        cdef Py_ssize_t node_id

        self.using_action = using_action

        self.novelty_nodes = [None] * cy_n_novelty_nodes

        for node_id in range(cy_n_novelty_nodes):
            self.novelty_nodes[node_id] = (
                NoveltyNode(
                    DoubleArray(
                        np_random_uniform(-1., 1., n_state_action_dims) )))

        init_DefaultEvolvingSystem(self)


    cpdef copy(self, copy_obj = None):
        cdef EvolvingSystemWithNoveltySearch new_system

        if copy_obj is None:
            new_system = (
                EvolvingSystemWithNoveltySearch.__new__(
                    EvolvingSystemWithNoveltySearch))
        else:
            new_system = copy_obj

        raise RuntimeError("Don't need this right now")
        # return new_system


    cpdef void receive_feedback(self, feedback) except *:
        cdef tuple cy_feedback

        cy_feedback = (FeedbackType.fitness, feedback)

        DefaultEvolvingSystem.receive_feedback(self, cy_feedback)

    cpdef void operate(self) except *:
        update_phenotypes(self)
        update_novelty_nodes(self)
        unmark_novelty_nodes(self)

    cpdef action(self, observation):
        cdef DoubleArray cy_observation
        cdef DoubleArray action
        cdef DoubleArray state_action
        cdef Py_ssize_t i
        cdef Py_ssize_t observation_size
        cdef double novelty_score
        cdef NoveltyNode closest_node
        cdef tuple feedback

        action = DefaultEvolvingSystem.action(self, observation)
        cy_observation = <DoubleArray?>observation
        observation_size = len(cy_observation)

        # Append state and action
        state_action = new_DoubleArray(observation_size + len(action))
        #
        for i in range(observation_size):
            state_action.view[i] = cy_observation.view[i]
        #
        for i in range(len(action)):
            if self.using_action:
                state_action.view[i + observation_size] = action.view[i]
            else:
                state_action.view[i + observation_size] = 0.0

        closest_node = (
            impl_closest_novelty_node(state_action, self.novelty_nodes, None))

        novelty_score = -impl_sqr_dist(state_action, closest_node.loc)

        closest_node.n_nearby_state_actions += 1

        feedback = (FeedbackType.novelty_score, novelty_score)

        (<PhenotypeWithNoveltyScore?>self.acting_phenotype()).receive_feedback(
            feedback)

        return action


@cython.warn.undeclared(True)
cpdef NoveltyNode impl_closest_novelty_node(
        DoubleArray x,
        list nodes,
        NoveltyNode excluded_node):
    cdef NoveltyNode closest_node
    cdef NoveltyNode node
    cdef double sqr_dist
    cdef double closest_sqr_dist

    closest_sqr_dist = INFINITY
    closest_node = None

    for node in nodes:
        if node is not excluded_node:
            sqr_dist = impl_sqr_dist(x, node.loc)
            if sqr_dist < closest_sqr_dist:
                closest_sqr_dist = sqr_dist
                closest_node = node

    if closest_node is None:
        raise RuntimeError("Something went wrong")

    return closest_node

@cython.warn.undeclared(True)
cpdef double impl_sqr_dist(DoubleArray x, DoubleArray y) except *:
    cdef double sqr_dist
    cdef double diff
    cdef Py_ssize_t i

    sqr_dist = 0.

    for i in range(len(x)):
        diff = x.view[i] - y.view[i]
        sqr_dist += diff * diff

    return sqr_dist

@cython.warn.undeclared(True)
cpdef void update_phenotypes(
        EvolvingSystemWithNoveltySearch system
        ) except *:
    cdef Py_ssize_t n_phenotypes
    cdef Py_ssize_t n_extra_phenotypes
    cdef Py_ssize_t phenotype_id
    cdef list extra_phenotypes
    cdef list phenotypes
    cdef list survivors
    cdef list super_fit_children
    cdef list novel_phenotypes
    cdef list fit_novel_children
    cdef Py_ssize_t match_id
    cdef PhenotypeWithNoveltyScore extra_phenotype
    cdef PhenotypeWithNoveltyScore contender_a
    cdef PhenotypeWithNoveltyScore contender_b
    cdef double fitness_a
    cdef double fitness_b
    cdef double novelty_score_a
    cdef double novelty_score_b

    phenotypes = system.phenotypes_shallow_copy()

    # Reduce copy of phenotype list size until the size is divible by 4
    # (for practical reasons). Save the extra phenotypes.
    extra_phenotypes = []
    n_extra_phenotypes = len(phenotypes) % 4
    for phenotype_id in range(n_extra_phenotypes):
        extra_phenotype = phenotypes.pop()
        extra_phenotypes.append(extra_phenotype)

    # TODO Optimize random shuffle with non-python random shuffle.
    random_shuffle(phenotypes)

    # Get population size.
    n_phenotypes = len(phenotypes)

    # Get survivors.
    survivors = [None] * (n_phenotypes//2)

    for match_id in range(n_phenotypes// 2):
        # Find the match winner amongst contenders.
        contender_a = phenotypes[2 * match_id]
        contender_b = phenotypes[2 * match_id + 1]
        fitness_a = contender_a.fitness()
        fitness_b = contender_b.fitness()
        if fitness_a > fitness_b:
            survivors[match_id] = contender_a
        else:
            survivors[match_id] = contender_b

    # Get fittest children from survivors.
    super_fit_children = [None] * (n_phenotypes//4)

    for match_id in range(n_phenotypes//4):
        # Find the match winner amongst contenders.
        contender_a = survivors[2 * match_id]
        contender_b = survivors[2 * match_id + 1]
        fitness_a = contender_a.fitness()
        fitness_b = contender_b.fitness()
        if fitness_a > fitness_b:
            super_fit_children[match_id] = contender_a.child()
        else:
            super_fit_children[match_id] = contender_b.child()


    # Get novel phenotypes.
    # TODO Optimize random shuffle with non-python random shuffle.
    random_shuffle(phenotypes)

    novel_phenotypes = [None] * (n_phenotypes//2)

    for match_id in range(n_phenotypes// 2):
        # Find the match winner amongst contenders.
        contender_a = phenotypes[2 * match_id]
        contender_b = phenotypes[2 * match_id + 1]
        novelty_score_a = contender_a.novelty_score
        novelty_score_b = contender_b.novelty_score
        if novelty_score_a > novelty_score_b:
            novel_phenotypes[match_id] = contender_a
        else:
            novel_phenotypes[match_id] = contender_b

    # Get fit novel children.
    fit_novel_children = [None] * (n_phenotypes//4)

    for match_id in range(n_phenotypes//4):
        # Find the match winner amongst contenders.
        contender_a = novel_phenotypes[2 * match_id]
        contender_b = novel_phenotypes[2 * match_id + 1]
        fitness_a = contender_a.fitness()
        fitness_b = contender_b.fitness()
        if fitness_a > fitness_b:
            fit_novel_children[match_id] = contender_a.child()
        else:
            fit_novel_children[match_id] = contender_b.child()

    # Grow list back to population size and then shuffle
    # We shuffle so that it isn't predictable what the phenotype is good at.
    phenotypes = []
    phenotypes.extend(survivors)
    phenotypes.extend(fit_novel_children)
    phenotypes.extend(super_fit_children)
    phenotypes.extend(extra_phenotypes)

    # TODO Optimize random shuffle with non-python random shuffle.
    random_shuffle(phenotypes)

    if len(phenotypes) != n_phenotypes + n_extra_phenotypes:
        raise RuntimeError("Something went wrong")

    # Set population.
    system.set_phenotypes(phenotypes)

@cython.warn.undeclared(True)
def n_nearby_state_actions_for_NoveltyNode(NoveltyNode node):
    return node.n_nearby_state_actions

@cython.warn.undeclared(True)
def sqr_dist_to_closest_other_node_for_NoveltyNode(NoveltyNode node):
    return node.sqr_dist_to_closest_other_node

@cython.warn.undeclared(True)
cpdef void update_novelty_nodes(
        EvolvingSystemWithNoveltySearch system
        ) except *:
    cdef list new_nodes
    cdef list parent_nodes
    cdef NoveltyNode node
    cdef NoveltyNode closest_node
    cdef NoveltyNode non_surviving_node
    cdef NoveltyNode parent_a
    cdef NoveltyNode parent_b
    cdef NoveltyNode parent_c
    cdef Py_ssize_t n_nodes
    cdef Py_ssize_t node_id


    n_nodes = len(system.novelty_nodes)

    new_nodes  = [None] * n_nodes
    for node_id in range(n_nodes):
        new_nodes[node_id] = system.novelty_nodes[node_id]


    # Nodes must be near experience state-action data.
    for node_id in range(n_nodes//2):
        new_nodes.sort(
            reverse = True,
            key = n_nearby_state_actions_for_NoveltyNode)
        non_surviving_node = new_nodes.pop()
        closest_node = (
            impl_closest_novelty_node(
                non_surviving_node.loc,
                new_nodes,
                non_surviving_node))
        closest_node.n_nearby_state_actions += (
            non_surviving_node.n_nearby_state_actions)


    # Nodes must be far away from other nodes.
    for node in new_nodes:
        closest_node = impl_closest_novelty_node(node.loc, new_nodes, node)
        node.closest_other_node = closest_node
        node.sqr_dist_to_closest_other_node = (
            impl_sqr_dist(node.loc, closest_node.loc))
    #
    for node_id in range(n_nodes//4):
        new_nodes.sort(
            reverse = True,
            key = sqr_dist_to_closest_other_node_for_NoveltyNode)
        non_surviving_node = new_nodes.pop()
        for node in new_nodes:
            if node.closest_other_node is non_surviving_node:
                closest_node = (
                    impl_closest_novelty_node(node.loc, new_nodes, node))
                node.closest_other_node = closest_node
                node.sqr_dist_to_closest_other_node = (
                    impl_sqr_dist(node.loc, closest_node.loc))

    # Repopulate Nodes
    parent_nodes = [None] * len(new_nodes)
    for node_id in range(len(new_nodes)):
        parent_nodes[node_id] = new_nodes[node_id]
    #
    for node_id in range(n_nodes - n_nodes//4):
        random_shuffle(parent_nodes)
        parent_a = parent_nodes[0]
        parent_b = parent_nodes[1]
        parent_c = parent_nodes[2]
        new_nodes.append(child_of_NoveltyNodes(parent_a, parent_b, parent_c))

@cython.warn.undeclared(True)
cpdef NoveltyNode child_of_NoveltyNodes(
        NoveltyNode parent_a,
        NoveltyNode parent_b,
        NoveltyNode parent_c):
    cdef double random_n
    cdef double random_n2
    cdef object random_dir_py
    cdef DoubleArray child_loc
    cdef DoubleArray random_dir
    cdef Py_ssize_t i

    random_n = np_random_uniform(0., 2.)


    child_loc = parent_a.loc.copy()

    for i in range(len(parent_a.loc)):
        child_loc.view[i] += (
            random_n
            * (parent_b.loc.view[i] - parent_c.loc.view[i]) )

    random_n2 = np_random_uniform(
        0.,
        DIFF_RANDOM_SPREAD_FACTOR * sqrt(impl_sqr_dist(child_loc, child_loc)) )
    random_dir_py = np_random_normal(0.0, 1.0, (len(parent_a.loc),) )
    random_dir_py /= np_linalg_norm(random_dir_py)
    random_dir = DoubleArray(random_dir_py)

    for i in range(len(parent_a.loc)):
        child_loc.view[i] += random_n2 * random_dir.view[i]

    return NoveltyNode(child_loc)

@cython.warn.undeclared(True)
cpdef void unmark_novelty_nodes(
        EvolvingSystemWithNoveltySearch system
        ) except *:
    cdef NoveltyNode node

    for node in system.novelty_nodes:
        node.n_nearby_state_actions = 0
        node.closest_other_node = None


