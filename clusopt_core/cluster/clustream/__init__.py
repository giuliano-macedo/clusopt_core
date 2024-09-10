from typing import Tuple
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from .clustream import __CluStream__


class CluStream(__CluStream__):
    def init_offline(self, init_points:NDArray, seed:int=0, n_init:int=3, max_iter:int=300) -> None:
        cluster_centers = (
            KMeans(
                n_clusters=self.m,
                init="k-means++",
                random_state=seed,
                n_init=n_init,
                max_iter=max_iter,
            )
            .fit(init_points)
            .cluster_centers_
        )
        self.init_kernels_offline(cluster_centers, init_points)

    def get_macro_clusters(self, k:int, seed:int=0, n_init:int=3, max_iter:int=300) -> Tuple[NDArray, NDArray]:
        model = KMeans(
            init="k-means++",
            random_state=seed,
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
        )
        labels = model.fit_predict(self.get_kernel_centers())
        centers = model.cluster_centers_
        return centers, labels
