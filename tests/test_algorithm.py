import unittest
import numpy as np
from anytime_infw import inner_infw, anytime_infw, AnytimeINFWConfig, InnerLimit

class Quadratic:
    first_oracle_call = 0
    def f(self, x): return float(x @ x / 2)
    def oracle_grad(self, x, direction=None):
        self.first_oracle_call += 1
        return x.copy()

class Tests(unittest.TestCase):
    def test_certificate_precedes_descent(self):
        x = np.array([0.1])
        y, g = inner_infw(x, 0.2, 1., Quadratic(), first_order=False)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, g)
    def test_descent(self):
        x = np.array([2.])
        o = Quadratic()
        y, g = inner_infw(x, .1, .1, o, first_order=False)
        self.assertGreaterEqual(o.f(x)-o.f(y), .1*np.linalg.norm(g)/2)
    def test_zero_gradient(self):
        r = anytime_infw(np.zeros(2), Quadratic())
        self.assertEqual(r.status, 'stationary')
        self.assertEqual(r.certificate[2], 0)
    def test_anytime(self):
        o = Quadratic()
        r = anytime_infw(np.array([2.,1.]), o, AnytimeINFWConfig(T=10, seed=3))
        self.assertEqual(r.status, 'outer_limit')
        self.assertTrue(np.all(np.diff(r.history['f_t']) <= 1e-12))
        self.assertTrue(np.all(np.diff(r.history['best_eps']) <= 0))
        self.assertLessEqual(np.linalg.norm(r.certificate[1]), r.certificate[2])
    def test_stochastic_rejected(self):
        o = Quadratic(); o.stochastic = True
        with self.assertRaises(ValueError): anytime_infw(np.ones(1), o)
    def test_limit_is_not_certificate(self):
        with self.assertRaises(InnerLimit):
            inner_infw(np.ones(1), .01, .1, Quadratic(), max_iters=0)

if __name__ == '__main__': unittest.main()
