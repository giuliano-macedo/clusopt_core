from numpy.typing import NDArray


class DistanceMatrix:
    def __init__(self, max_size: int, no_threads: int=0) -> None:
        ...

    def compute(self, matrix: NDArray) -> NDArray:
        ...

class Silhouette:
    def __init__(self, n_clusters: int) -> None:
        ...
    
    def get_score(self, dist_table: NDArray, labels: NDArray) -> float:
        ...
