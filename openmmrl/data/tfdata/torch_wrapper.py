import torch
import tensorflow as tf
from torch.utils.data import IterableDataset

class TFDatasetToTorch(IterableDataset):
    """
    A PyTorch `IterableDataset` wrapper for a TensorFlow `tf.data.Dataset`.
    """
    def __init__(self, tf_dataset: tf.data.Dataset):
        """
        Initializes the wrapper.

        Args:
            tf_dataset: The TensorFlow dataset to wrap.
        """
        super().__init__()
        self.tf_dataset = tf_dataset

    def __iter__(self):
        """
        Iterates over the TensorFlow dataset and yields PyTorch tensors.
        """
        for example in self.tf_dataset.as_numpy_iterator():
            # Convert numpy arrays to torch tensors
            torch_example = {}
            for key, value in example.items():
                if isinstance(value, dict):
                    torch_example[key] = {
                        sub_key: torch.from_numpy(sub_value)
                        for sub_key, sub_value in value.items()
                    }
                else:
                    torch_example[key] = torch.from_numpy(value)
            yield torch_example

    def __len__(self):
        """
        Returns the number of elements in the dataset.
        Note: For `tf.data.Dataset`, cardinality can be infinite, unknown, or finite.
        This may not be accurate for all datasets.
        """
        return self.tf_dataset.cardinality().numpy() 