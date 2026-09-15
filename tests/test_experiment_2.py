import unittest

import numpy as np

from experiments import experiment_2


class Experiment2Tests(unittest.TestCase):
    def test_exact_design_gram_matrix(self):
        for rho in (0.0, 0.5, 0.9, 0.99):
            design = experiment_2.build_exact_design(100, rho)
            expected = 100 * np.array([[1.0, rho], [rho, 1.0]])
            np.testing.assert_allclose(design.T @ design, expected, atol=1e-10)

    def test_mean_field_variance_is_inverse_precision_diagonal(self):
        precision = np.array([[2.0, 0.8], [0.8, 2.0]])
        mean = np.zeros(2)
        _, covariance = experiment_2.gaussian_mf_forward_kl_optimum(mean, precision)
        np.testing.assert_allclose(np.diag(covariance), [0.5, 0.5])

    def test_collinearity_increases_condition_number(self):
        kappas = []
        for rho in (0.0, 0.5, 0.9, 0.99):
            design = experiment_2.build_exact_design(100, rho)
            precision = design.T @ design + np.eye(2)
            eigvals = np.linalg.eigvalsh(precision)
            kappas.append(eigvals[-1] / eigvals[0])
        self.assertTrue(np.all(np.diff(kappas) > 0))


if __name__ == "__main__":
    unittest.main()
