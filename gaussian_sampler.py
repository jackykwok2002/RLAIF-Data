import numpy as np
from typing import Optional, Union, Tuple, List
from itertools import combinations


class MultivariateGaussianSampler:
    """
    A class for generating samples from a multivariate Gaussian distribution
    with independent components (diagonal covariance matrix).
    """

    def __init__(
        self,
        means: Union[list, np.ndarray],
        variances: Union[list, np.ndarray],
        random_seed: Optional[int] = None
    ):
        """
        Initialize the sampler with means and variances.

        Parameters:
        -----------
        means : array-like
            Mean values for each dimension
        variances : array-like
            Variance values for each dimension
        random_seed : int, optional
            Seed for random number generation

        Raises:
        -------
        ValueError
            If means and variances have different lengths or contain invalid values
        """
        self.means = np.array(means, dtype=np.float64)
        self.variances = np.array(variances, dtype=np.float64)

        if len(self.means) != len(self.variances):
            raise ValueError("Means and variances must have the same length")

        if np.any(self.variances < 0):
            raise ValueError("Variances must be non-negative")

        self.std_devs = np.sqrt(self.variances)
        self.dimensions = len(means)
        self.rng = np.random.default_rng(seed=random_seed)

    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the specified multivariate Gaussian distribution.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate

        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, dimensions) containing the samples

        Raises:
        -------
        ValueError
            If n_samples is less than 1
        """
        if n_samples < 1:
            raise ValueError("Number of samples must be positive")

        samples = np.zeros((n_samples, self.dimensions))
        for i in range(self.dimensions):
            samples[:, i] = self.rng.normal(
                loc=self.means[i],
                scale=self.std_devs[i],
                size=n_samples
            )
        return samples

    def get_pairwise_comparisons(self, samples: np.ndarray):
        """
        Generate all possible pairwise combinations of samples without repetition.

        Parameters:
        -----------
        samples : numpy.ndarray
            Array of shape (n_samples, dimensions) containing the samples

        Returns:
        --------
        list
            List of tuples, each containing:
            - First sample vector
            - Second sample vector
            - Index of first sample
            - Index of second sample

        Example:
        --------
        If samples has shape (3, 2), indicating 3 samples of 2 dimensions each:
        [[1, 2], [3, 4], [5, 6]]

        The function returns:
        [
            (array([1, 2]), array([3, 4]), 0, 1),
            (array([1, 2]), array([5, 6]), 0, 2),
            (array([3, 4]), array([5, 6]), 1, 2)
        ]
        """
        n_samples = len(samples)
        pairs = []

        # Generate all possible pairs of indices
        for i, j in combinations(range(n_samples), 2):
            # Store both samples and their indices
            pairs.append((
                samples[i],
                samples[j],
                i,
                j
            ))

        return pairs

    def get_statistics(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate sample statistics (means and variances).

        Parameters:
        -----------
        samples : numpy.ndarray
            Array of samples to analyze

        Returns:
        --------
        tuple
            (sample_means, sample_variances)
        """
        sample_means = np.mean(samples, axis=0)
        sample_variances = np.var(samples, axis=0)
        return sample_means, sample_variances

    @property
    def parameters(self) -> dict:
        """
        Get the distribution parameters.

        Returns:
        --------
        dict
            Dictionary containing the means and variances
        """
        return {
            'means': self.means,
            'variances': self.variances,
            'dimensions': self.dimensions
        }


# # Define your means and variances
# means = np.array([
#     0.00042034, 0.00028339, 0.00044667, 0.00031542,
#     -0.00144568, 0.00062363, 0.63334810
# ])

# variances = np.array([
#     0.00010464, 0.00024932, 0.00017878, 0.00062988,
#     0.00095615, 0.00290656, 0.22948318
# ])

# # Create sampler instance
# sampler = MultivariateGaussianSampler(means, variances, random_seed=42)

# # Generate samples
# samples = sampler.generate_samples(500)

# # Get statistics
# sample_means, sample_variances = sampler.get_statistics(samples)

# # Print results
# print("Sample means:", sample_means)
# print("Sample variances:", sample_variances)

# # Get distribution parameters
# params = sampler.parameters
# print("Distribution parameters:", params)
