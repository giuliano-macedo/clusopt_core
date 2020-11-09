from clusopt_core.cluster import CluStream
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

k = 32

dataset, _ = make_blobs(n_samples=64000, centers=k, random_state=42, cluster_std=0.1)

model = CluStream(
    m=k * 10,  # no microclusters
    h=64000,  # horizon
    t=2,  # radius factor
)

chunks = np.split(dataset, len(dataset) / 4000)

model.init_offline(chunks.pop(0), seed=42)

for chunk in chunks:
    model.partial_fit(chunk)

clusters, _ = model.get_macro_clusters(k, seed=42)

plt.scatter(*dataset.T, marker=",", label="datapoints")

plt.scatter(*model.get_partial_cluster_centers().T, marker=".", label="microclusters")

plt.scatter(*clusters.T, marker="x", label="macro clusters", color="black")

plt.legend()
plt.show()