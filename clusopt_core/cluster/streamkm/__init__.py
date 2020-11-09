from sklearn.cluster import KMeans
from .streamkm import Streamkm as Streamkm_


class Streamkm(Streamkm_):
    def get_final_clusters(self, k, seed=0, n_init=3, max_iter=300):
        """clusters coresets with given k using Kmeans++

        Args:
            k (int): number of clusters
            seed (int, optional): seed for the random number generator.
            n_init (int, optional): number of times to repeat kmeans. Defaults to 3.
            max_iter (int, optional): maximum number of kmeans iterations. Defaults to 300.
        """
        model = KMeans(
            init="k-means++",
            random_state=seed,
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = model.fit_predict(self.get_streaming_coreset_centers())
        centers = model.cluster_centers_
        return centers, labels