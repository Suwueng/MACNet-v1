import os
import sys
import torch
import numpy as np

print(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RawDataProcessing.GalaxyData import GalDataSet
from RawDataProcessing.ParseLogFile import load_config


class CachePt:
    def __init__(self, dataset: GalDataSet = None):
        self._dataset = dataset

    # ------------ Properties -----------

    @property
    def dataset(self):
        return self._dataset

    # ------------ Dunder Methods ------------

    def __len__(self):
        return len(self._dataset.raw_data["time"])

    def __add__(self, other: "CachePt") -> "CachePt":
        if not isinstance(other, CachePt):
            raise TypeError("Can only add CachePt to another CachePt")
        combined_dataset = self._dataset + other._dataset
        return CachePt(combined_dataset)

    # ------------ Methods ------------

    def load(self, path: str):
        self._dataset = GalDataSet().load_data(path)
        self._dataset.drop_invalid()
        return self

    def split(
        self,
        train_size: float,
        validation_size: float,
        stratify: str | np.ndarray | None = None,
    ):
        train_set, val_set, test_set = self._dataset.split(
            train_size=train_size, validation_size=validation_size, stratify=stratify
        )
        return CachePt(train_set), CachePt(val_set), CachePt(test_set)

    def process(
        self, threshold: float = None, mirror: bool = False, balance: str = None
    ):
        data = self._dataset
        if threshold is not None:
            data.filter(threshold)
        if mirror:
            data.mirror_data()
        if balance is not None:
            if balance not in ["oversample", "undersample"]:
                raise ValueError("Balance method must be 'oversample' or 'undersample'")
            data.balance_groups(method=balance)
        return CachePt(data)

    def to_pt(
        self,
        save_path: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ):
        data = self._dataset
        if mean is not None and std is not None:
            x = data.standardize(mean, std)
        else:
            x, mean, std = data.standardize()
        torch.save(
            (
                x,
                data.coordinate,
                data.y,
                data.mbh,
                data.y_baseline,
                data.groups,
                data.n_groups,
            ),
            save_path,
        )
        return mean, std


def get_folder_paths(configs_path, gal_type, resolution):
    configs = load_config(configs_path)
    data_dir = configs["BaseConfig"]["data_dir"]
    raw_data_config = configs["RawDataConfig"]

    folder_abs_paths = []
    for gal_nickname in raw_data_config[gal_type]:
        gal_name = f"{gal_type}_{gal_nickname}"
        folder_abs_path = os.path.join(data_dir, gal_name, resolution)
        folder_abs_paths.append([folder_abs_path, gal_name])
    return folder_abs_paths


# Define global paths for external imports (e.g. by notebooks)
try:
    _resolution = "coarse"
    _configs_dir = ".config"
    Exp_eg_folder_paths = get_folder_paths(_configs_dir, "elliptical_galaxy", _resolution)
    Exp_dg_folder_paths = get_folder_paths(_configs_dir, "disk_galaxy", _resolution)
except Exception as e:
    # This might happen if CWD is not project root during import
    Exp_eg_folder_paths = []
    Exp_dg_folder_paths = []


if __name__ == "__main__":
    resolution = "coarse"
    configs_dir = ".config"

    # Use the global variables (renamed to match notebook import expectation)
    exp_eg_folder_paths = Exp_eg_folder_paths
    exp_dg_folder_paths = Exp_dg_folder_paths
    
    exp_all_folder_paths = exp_eg_folder_paths + exp_dg_folder_paths

    # Create in- and out-datasets for each galaxy type
    exp_eg_folder_paths_in = exp_eg_folder_paths.copy()
    exp_dg_folder_paths_in = exp_dg_folder_paths.copy()
    exp_eg_folder_paths_in.pop(2)
    exp_dg_folder_paths_in.pop(2)
    exp_eg_folder_paths_out = [exp_eg_folder_paths[2]]
    exp_dg_folder_paths_out = [exp_dg_folder_paths[2]]

    for i, (in_paths, out_path) in enumerate(
        [
            (exp_eg_folder_paths_in, exp_eg_folder_paths_out),
            (exp_dg_folder_paths_in, exp_dg_folder_paths_out),
        ]
    ):
        print(f"Processing Experiment {i+4}...")
        save_path = os.path.join(".cache", f"Exp{i+4}_")

        cache_in = CachePt().load(in_paths)
        train_cache, val_cache, test_cache = cache_in.split(
            train_size=0.6, validation_size=0.2, stratify="groups"
        )

        train_cache = train_cache.process(threshold=-5, balance="oversample")
        val_cache = val_cache.process(threshold=-5)
        test_cache = test_cache.process(threshold=-5)
        print(f"  Training set size:   {len(train_cache)}")
        print(f"  Validation set size: {len(val_cache)}")
        print(f"  Testing set size:    {len(test_cache)}")

        mean, std = train_cache.to_pt(save_path + "train.pt")
        val_cache.to_pt(save_path + "val.pt", mean, std)
        test_cache.to_pt(save_path + "test.pt", mean, std)

        cache_out = CachePt().load(out_path)

        cache_out = cache_out.process()
        print(f"  Out-set size: {len(cache_out)}")

        cache_out.to_pt(save_path + "out.pt", mean, std)