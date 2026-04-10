from __future__ import annotations

"""
Indexes (case_id, t) for local rejector training; FNO forward happens in train loop.
"""

from datasets.ep_operator_dataset import EPOneStepDataset


class LocalRejectorDataset(EPOneStepDataset):
    """Same samples as one-step dataset; backbone builds multi-channel input in trainer."""

    pass
