from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterator


class ResolutionBatchSampler:
    """
    Batches indices so every batch shares the same spatial shape (H, W).
    Required when the manifest mixes resolutions (e.g. 64 and 96).
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def _hw_for_index(self, idx: int) -> tuple[int, int]:
        case_id, _ = self.dataset.samples[idx]
        v, _ = self.dataset._load_case(case_id)
        return int(v.shape[1]), int(v.shape[2])

    def _build_batches_at_epoch(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + int(epoch))
        hw_to_idx: dict[tuple[int, int], list[int]] = defaultdict(list)
        n = len(self.dataset)
        for idx in range(n):
            hw_to_idx[self._hw_for_index(idx)].append(idx)
        batches: list[list[int]] = []
        for indices in hw_to_idx.values():
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)
            else:
                idxs.sort()
            step = self.batch_size
            for i in range(0, len(idxs), step):
                chunk = idxs[i : i + step]
                if self.drop_last and len(chunk) < step:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        self._epoch += 1
        yield from self._build_batches_at_epoch(self._epoch)

    def __len__(self) -> int:
        return len(self._build_batches_at_epoch(self._epoch + 1))
