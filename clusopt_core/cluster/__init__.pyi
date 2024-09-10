from typing import Tuple
from numpy.typing import NDArray

class CluStream:
    """CluStream data stream clustering algorithm implementation"""
    
    @property
    def m(self) -> int:
        """Maximum number of micro kernels to use"""

    @property
    def time_window(self) -> int:
        ...
    
    @property
    def t(self) -> int:
        """Multiplier for the kernel radius"""

    @property
    def timestamp(self) -> int:
        ...

    @property
    def points_fitted(self) -> int:
        ...

    @property
    def points_fitted(self) -> int:
        ...

    @property
    def points_forgot(self) -> int:
        ...

    @property
    def points_merged(self) -> int:
        ...


    def __init__(self, h: int=100, m: int=1000, t:int=2) -> None:
        """
        Args:
            h (int, optional): Range of the window. Defaults to 100.
            m (int, optional): Maximum number of micro kernels to use. Defaults to 1000.
            t (int, optional): Multiplier for the kernel radius. Defaults to 2.
        """

    def batch_online_cluster(self, batch: NDArray) -> None:
        """Process a chunk of datapoints all at once"""
    
    def partial_fit(self, batch: NDArray) -> None:
        """Process a chunk of datapoints all at once"""

    def get_kernel_centers(self)->NDArray:
        """Get current microclusters centroids"""

    def get_partial_cluster_centers(self)->NDArray:
        """Get current microclusters centroids"""

    def init_kernels_offline(self, cluster_centers: NDArray, initpoints: NDArray) -> None:
        """initialize m kernels with its coresponding initpoints and clustering centers

        Args:
            cluster_centers (NDArray): the offline clustering of the initpoints
            initpoints (NDArray): datapoints to initialize
        """


    def init_offline(self, init_points:NDArray, seed:int=0, n_init:int=3, max_iter:int=300) -> None:
        """
        initialize microclusters using kmeans++

        Args:
            init_points (ndarray): points to initialize
            seed (int):random number generator seed
            n_init (int): number of kmeans runs
            max_iter (int): max number of kmeans iterations
        """

    def get_macro_clusters(self, k:int, seed:int =0, n_init:int=3, max_iter:int=300) -> Tuple[NDArray, NDArray]:
        """
        clusters microclusters and returns macroclusters centers

        Args:
            k (int): number of centers
            seed (int): random number generator seed
            n_init (int): number of kmeans runs
            max_iter (int): max number of kmeans iterations

        Returns:
                centers,labels
        """

class Streamkm:
    """StreaKM++ data stream clustering algorithm implementation"""
    @property
    def coresetsize(self)->int:
        """Number of coresets to use"""
    
    @property
    def length(self) -> int:
        """Total length of the dataset"""

    def __init__(self, coresetsize: int, length: int, seed: int)->None:
        """StreaKM++ data stream clustering algorithm implementation

        Args:
            coresetsize (int): Number of coresets to use
            length (int): Total length of the dataset
            seed (int): Random number generator seed
        """

    def batch_online_cluster(self, batch: NDArray)->None:
        """Process a chunk of datapoints all at once"""

    def partial_fit(self, batch: NDArray)->None:
        """Process a chunk of datapoints all at once"""

    def get_streaming_coreset_centers(self) -> NDArray:
        """Get current streaming coreset centers"""
        
    def get_partial_cluster_centers(self) -> NDArray:
        """Get current streaming coreset centers"""


    def get_final_clusters(self, k:int, seed:int=0, n_init:int=3, max_iter:int=300) -> Tuple[NDArray, NDArray]:
        """clusters coresets with given k using Kmeans++

        Args:
            k (int): number of clusters
            seed (int, optional): seed for the random number generator.
            n_init (int, optional): number of times to repeat kmeans. Defaults to 3.
            max_iter (int, optional): maximum number of kmeans iterations. Defaults to 300.

        Returns:
            centers,labels
        """
