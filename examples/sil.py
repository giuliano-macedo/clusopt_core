from sklearn.datasets import make_blobs
from clusopt_core.metrics import DistanceMatrix, Silhouette

k = 10

dataset, labels = make_blobs(centers=k, n_samples=100)

distances = DistanceMatrix(100).compute(dataset)

sil = Silhouette(k).get_score(distances, labels)

print(f"Silhouette score : {sil:.2f}")
