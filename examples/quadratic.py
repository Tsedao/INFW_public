import numpy as np
from anytime_infw import anytime_infw, AnytimeINFWConfig

class Quadratic:
    first_oracle_call = 0
    def f(self, x): return float(x @ x / 2)
    def oracle_grad(self, x, direction=None):
        self.first_oracle_call += 1
        return x.copy()

result = anytime_infw(np.array([2., 1.]), Quadratic(), AnytimeINFWConfig(T=10, seed=3))
print('Status:', result.status)
print('Final point:', result.x_T)
print('Best certificate radius:', result.certificate[2])
