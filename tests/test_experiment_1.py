import unittest

import numpy as np

from experiments import experiment_1


class Experiment1Tests(unittest.TestCase):
    def test_mean_field_closed_form(self):
        rho = 0.9
        expected = (1.0 - rho**2) * np.eye(2)
        np.testing.assert_allclose(experiment_1.mean_field_covariance(rho), expected)

    def test_kl_closed_form(self):
        rho = 0.5
        self.assertTrue(
            np.isclose(experiment_1.optimal_mean_field_kl(rho), -0.5 * np.log(0.75))
        )

    def test_step_size_is_stable(self):
        for rho in (0.0, 0.5, 0.9, 0.99):
            precision = experiment_1.precision_matrix(rho)
            eta = experiment_1.ula_step_size(rho, 0.8)
            self.assertLess(eta * np.linalg.eigvalsh(precision)[-1], 2.0)


if __name__ == "__main__":
    unittest.main()
