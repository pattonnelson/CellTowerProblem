import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population, Individual

class CustomMixedSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        individuals = []
        for _ in range(n_samples):
            x = {}
            for k, var in problem.vars.items():
                lb, ub = var.bounds
                if var.type == 'real':
                    x[k] = np.random.uniform(lb, ub)
                elif var.type == 'int':
                    x[k] = np.random.randint(lb, ub + 1)
                else:
                    raise ValueError(f"Unsupported variable type: {var.type}")
            individuals.append(Individual(X=x))
        return Population.create(*individuals)
