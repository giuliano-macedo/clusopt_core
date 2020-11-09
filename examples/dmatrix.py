from clusopt_core.metrics import DistanceMatrix
import numpy as np

x = np.random.rand(5, 2)

dist_matrix = DistanceMatrix(5)

print("input:")
print(x)

print("dist matrix")
print(dist_matrix.compute(x))
