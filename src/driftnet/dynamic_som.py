import numpy as np

class DynamicSOM:
    """
    A Self-Organizing Map (SOM) that can dynamically grow by adding new nodes.
    Unlike a standard SOM with a fixed grid, this SOM's topology is flexible.
    """
    def __init__(self, input_dim: int, initial_nodes: int = 4, learning_rate: float = 0.1, sigma: float = 1.0):
        """
        Initializes the DynamicSOM.

        Args:
            input_dim (int): The dimensionality of the input vectors.
            initial_nodes (int): The number of nodes to start with.
            learning_rate (float): The initial learning rate for weight updates.
            sigma (float): The initial radius of the neighborhood function.
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma = sigma

        # The codebook stores the weight vector for each neuron.
        # We use a list to allow for dynamic growth.
        self.codebook = [np.random.rand(input_dim) for _ in range(initial_nodes)]

        # Each neuron also has a position in a 2D latent space for topological relationships.
        # We initialize them in a small grid.
        self.positions = [np.array([i % 2, i // 2], dtype=float) for i in range(initial_nodes)]

        self.num_nodes = initial_nodes

    def _calculate_distance(self, x: np.array, w: np.array) -> float:
        """Calculates the Euclidean distance between two vectors."""
        return np.linalg.norm(x - w)

    def find_bmu(self, x: np.array) -> int:
        """
        Finds the Best Matching Unit (BMU) for a given input vector x.

        Returns:
            int: The index of the winning neuron in the codebook.
        """
        distances = [self._calculate_distance(x, w) for w in self.codebook]
        return np.argmin(distances)

    def add_node(self, activation_neuron_idx: int):
        """
        Adds a new node to the SOM, near an existing neuron that is frequently activated.

        Args:
            activation_neuron_idx (int): The index of the neuron to spawn a new node near.
        """
        # Initialize the new node's weight vector to be similar to the activation neuron's weight.
        new_weight = self.codebook[activation_neuron_idx].copy() + np.random.normal(0, 0.1, self.input_dim)
        self.codebook.append(new_weight)

        # Place the new node near the activation neuron in the 2D latent space.
        new_position = self.positions[activation_neuron_idx].copy() + np.random.normal(0, 0.2, 2)
        self.positions.append(new_position)

        self.num_nodes += 1
        print(f"Node added. Total nodes: {self.num_nodes}")
        return self.num_nodes - 1

    def update(self, x: np.array, bmu_idx: int, epoch: int, max_epochs: int):
        """
        Updates the weights of the BMU and its neighbors.

        Args:
            x (np.array): The input vector.
            bmu_idx (int): The index of the Best Matching Unit.
            epoch (int): The current training epoch, used for decaying learning rate and sigma.
            max_epochs (int): The total number of training epochs.
        """
        # Decay learning rate and sigma over time
        lr_decay = np.exp(-epoch / max_epochs)
        sigma_decay = np.exp(-epoch / max_epochs)

        current_lr = self.learning_rate * lr_decay
        current_sigma = self.sigma * sigma_decay

        bmu_position = self.positions[bmu_idx]

        for i in range(self.num_nodes):
            node_position = self.positions[i]
            distance_to_bmu = np.linalg.norm(bmu_position - node_position)

            if distance_to_bmu < current_sigma:
                # Calculate neighborhood influence (Gaussian function)
                influence = np.exp(-distance_to_bmu**2 / (2 * current_sigma**2))

                # Update weight vector
                self.codebook[i] += influence * current_lr * (x - self.codebook[i])
