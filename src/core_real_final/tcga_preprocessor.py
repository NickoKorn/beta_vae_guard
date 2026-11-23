"""Split paired TCGA samples by participant before fitting split-dependent transforms."""

import re

import numpy as np
from autoencodix.data._datapackage_splitter import DataPackageSplitter
from autoencodix.data._datasplitter import DataSplitter
from autoencodix.data.general_preprocessor import GeneralPreprocessor


class TCGAPreprocessor(GeneralPreprocessor):
    def _split_data_package(self, data_package):
        sample_ids = data_package.multi_bulk["RNA"].index
        if not sample_ids.is_unique or not all(
            re.fullmatch(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}(?:-.+)?", sid)
            for sid in sample_ids
        ):
            raise ValueError("Expected unique TCGA barcodes for paired RNA samples")

        participants = np.unique(sample_ids.str.slice(0, 12))
        splits = DataSplitter(self.config).split(len(participants))
        indices = {}
        for key, frame in data_package:
            if not frame.index.is_unique or not frame.index.sort_values().equals(sample_ids.sort_values()):
                raise ValueError(f"Paired TCGA sample IDs differ in {key}")
            category, name = key.split(".", 1)
            patients = frame.index.str.slice(0, 12)
            indices.setdefault(category, {})[name] = {
                split: np.flatnonzero(patients.isin(participants[positions]))
                for split, positions in splits.items()
            }
        return DataPackageSplitter(data_package, self.config, indices).split(), indices
