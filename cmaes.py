from goldenrockefeller.policyopt.evolution import BaseEvolvingSystem
import numpy as np
from goldenrockefeller.cyutil.array import DoubleArray
import random

def ndarray_from_DoubleArray(d_arr):
    return np.asarray(d_arr.view).copy()

class CmaesSystem(BaseEvolvingSystem):
    def __init__(self, n_parameters = 1):
        BaseEvolvingSystem.__init__(self)
        self.step_size = 1.
        self._slow_down = 1.
        CmaesSystem.reinit_system(self, n_parameters)


    @staticmethod
    def reinit_system(system, n_parameters):
        slow_down = system._slow_down

        system.mean = np.zeros(n_parameters)

        system.covariance = np.ones(n_parameters)
        system.isotropic_path = np.zeros(n_parameters)
        system.anisotropic_path = np.zeros(n_parameters)
        system.epsilon = 1.e-9

        # learning rates (lr)
        system.mean_lr = 1. / slow_down
        system.isotropic_lr = 1. / (1. + n_parameters/3.) / slow_down
        system.anisotropic_lr = 1. / (1. + n_parameters/4.) / slow_down
        system.covariance_path_lr = 1. / (2. + n_parameters**2/2.) / slow_down
        system.step_size_damping = 1.

        system.expected_step_size = (
            np.sqrt(n_parameters)
            * (1 - 1/(4*n_parameters) + 1 / (21 * n_parameters ** 2))
        )

    def copy(self, copy_obj = None):
        raise NotImplementedError("Not implemented yet.")

    def operate(self):



        n_parameters = len(self.mean)

        phenotypes = self.phenotypes().copy()
        phenotypes.sort(
            reverse = True,
            key = lambda phenotype : phenotype.fitness()
        )

        new_n_parameters = phenotypes[0].policy().n_parameters()

        if n_parameters != new_n_parameters:
            CmaesSystem.reinit_system(self, new_n_parameters)
            n_parameters = new_n_parameters

        n_phenotypes = len(phenotypes)
        n_kept_phenotypes =  n_phenotypes // 4
        n_selected_phenotypes = n_phenotypes // 2

        weights = float(n_selected_phenotypes) - np.arange(n_selected_phenotypes)
        weights /= weights.sum()

        selection_mass = 1. / (weights ** 2).sum()

        covariance_paramaters_lr = (
            1. / (2. + n_parameters**2/selection_mass) / self._slow_down
        )

        prev_mean = self.mean.copy()
        prev_step_size = self.step_size


        displacement = np.zeros(n_parameters)

        for phenotype_id in range(n_selected_phenotypes):
            phenotype = phenotypes[phenotype_id]
            weight = weights[phenotype_id]
            parameter_vector = (
                ndarray_from_DoubleArray(
                    phenotype.policy().parameters()
                )
            )

            displacement += weight * (parameter_vector - prev_mean)

        self.mean += self.mean_lr * displacement

        self.isotropic_path = (
            (1. - self.isotropic_lr) * self.isotropic_path
            + np.sqrt(1 - (1. - self.isotropic_lr) ** 2)
            * np.sqrt(selection_mass)
            * displacement
            / (prev_step_size * np.sqrt(self.covariance) + self.epsilon)
        )

        self.step_size = (
            prev_step_size
            * np.exp(
                self.isotropic_lr / self.step_size_damping
                * (
                    np.linalg.norm(self.isotropic_path)
                    / self.expected_step_size
                    - 1.
                )
            )
        )

        self.anisotropic_path = (
            (1. - self.anisotropic_lr) * self.anisotropic_path
            + np.sqrt(1 - (1. - self.anisotropic_lr) ** 2)
            * np.sqrt(selection_mass)
            * displacement
            / (prev_step_size + self.epsilon)
        )

        delta_covariance = np.zeros(n_parameters)

        for phenotype_id in range(n_selected_phenotypes):
            phenotype = phenotypes[phenotype_id]
            weight = weights[phenotype_id]
            parameter_vector = (
                ndarray_from_DoubleArray(
                    phenotype.policy().parameters()
                )
            )

            delta_covariance += (
                weight
                * (
                    (parameter_vector - prev_mean)
                    / (prev_step_size + self.epsilon)
                ) ** 2
            )

        self.covariance = (
            (1. - self.covariance_path_lr - covariance_paramaters_lr)
            * self.covariance
            +  self.covariance_path_lr * self.anisotropic_path ** 2
            + covariance_paramaters_lr * delta_covariance
        )

        # Sample new population
        phenotypes = phenotypes[:n_kept_phenotypes]

        for phenotype_id in range(n_phenotypes - len(phenotypes)):
            new_phenotype = phenotypes[0].copy()
            parameters = (
                self.mean
                + self.step_size
                * np.sqrt(self.covariance)
                * np.random.normal(size = (n_parameters,))
            )
            new_phenotype.policy().set_parameters(DoubleArray(parameters))


            phenotypes.append(new_phenotype)

        random.shuffle(phenotypes)
        self.set_phenotypes(phenotypes)
        print(f"{self.step_size=}, {np.sum(self.mean**2)}")





