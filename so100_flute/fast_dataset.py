"""Cache ACT target actions without changing LeRobot's sample construction.

HF column indexing with a custom transform can decode every column, including RGB, for all 100 future actions; instead if you select the action column, you aviod redundant work."""

import logging

import numpy as np
import torch
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class CachedActionReader(DatasetReader):
    def __init__(self, reader):
        # Preserve LeRobot's current image transforms, episode subset, and mapping.
        self.__dict__.update(reader.__dict__)
        actions = self.hf_dataset.select_columns(["action"]).with_format("numpy")[:]["action"]
        self._cached_actions = torch.from_numpy(np.asarray(actions, dtype=np.float32).copy())
        logging.info(
            "Cached %s action rows (%.2f MiB); future-action lookups skip image decoding",
            len(actions),
            self._cached_actions.numel() * self._cached_actions.element_size() / 2**20,
        )

    def _query_hf_dataset(self, query_indices):
        result = super()._query_hf_dataset(
            {k: v for k, v in query_indices.items() if k != "action"}
        )
        if "action" in query_indices:
            indices = query_indices["action"]
            if self._absolute_to_relative_idx is not None:
                indices = [self._absolute_to_relative_idx[i] for i in indices]
            # Advanced indexing returns a fresh tensor, protecting cached data.
            result["action"] = self._cached_actions[indices]
        return result


def cache_action_targets(dataset):
    """Keep the native dataset/sampler interface, including spawn-worker support."""
    if dataset is None:
        return dataset
    if not isinstance(dataset, LeRobotDataset):
        raise TypeError("Action caching requires a local, non-streaming LeRobotDataset")
    reader = dataset._ensure_reader()
    if not isinstance(reader, CachedActionReader):
        if reader.hf_dataset is None:
            reader.load_and_activate()
        if reader.delta_indices and "action" in reader.delta_indices:
            dataset.reader = CachedActionReader(reader)
    return dataset
