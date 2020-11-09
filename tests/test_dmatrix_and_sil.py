from clusopt_core.metrics import DistanceMatrix, Silhouette
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.datasets import make_blobs
import numpy as np

SEED = 42
REPETITIONS = 100
np.random.seed(SEED)


def test_dmatrix_and_sil():
    dist_matrix = DistanceMatrix(100)

    for _ in range(REPETITIONS):
        dim = np.random.randint(2, 20)
        k = np.random.randint(2, 20)

        dataset, labels = make_blobs(
            n_features=dim, centers=k, n_samples=100, random_state=SEED
        )

        gt_matrix = pairwise_distances(dataset)

        assert np.allclose(
            dist_matrix.compute(dataset),
            gt_matrix,
        )

        assert np.isclose(
            Silhouette(k).get_score(gt_matrix, labels),
            silhouette_score(gt_matrix, labels, metric="precomputed"),
        )
