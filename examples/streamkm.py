from clusopt_core.cluster import Streamkm
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

k = 32

dataset, _ = make_blobs(n_samples=64000, centers=k, random_state=42, cluster_std=0.1)
model = Streamkm(
    coresetsize=k * 10,
    length=64000,
    seed=42,
)

chunks = np.split(dataset, len(dataset) / 4000)

for chunk in chunks:
    model.partial_fit(chunk)

clusters, _ = model.get_final_clusters(k, seed=42)

plt.scatter(*dataset.T, marker=",", label="datapoints")

plt.scatter(
    *model.get_partial_cluster_centers().T, marker=".", label="StreaKM++ coresets"
)
plt.scatter(*clusters.T, marker="x", label="StreaKM++ final clusters", color="black")

plt.legend()
plt.show()