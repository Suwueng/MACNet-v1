import os
import sys
import torch
import numpy as np

print(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RawDataProcessing.GalaxyData import GalDataSet


def cache_pt(
    data_path: str,
    cache_path: str,
    threshold: float = -5,
    train_size: float = 0.6,
    validation_size: float = 0.2,
    balance: str = None,
    mirror: bool = False,
    stratify: str | np.ndarray | None = None,
):
    dataset = GalDataSet().load_data(data_path)
    dataset.drop_invalid().filter(threshold)
    if mirror:
        dataset.mirror_data()

    # Display data counts for each group
    unique_groups, counts = np.unique(dataset.groups, axis=0, return_counts=True)
    print("Data counts per group after filtering:")
    for group, count in zip(unique_groups, counts):
        print(f"  Group {group}: {count}")
    
    if balance is not None:
        if balance not in ["oversample", "undersample"]:
            raise ValueError("Balance method must be 'oversample' or 'undersample'")
        dataset.balance_groups(method=balance)

    train_set, val_set, test_set = dataset.split(
        train_size=train_size, validation_size=validation_size, stratify=stratify
    )
    print(f"Training set size:   {len(train_set.raw_data['time'])}")
    print(f"Validation set size: {len(val_set.raw_data['time'])}")
    print(f"Testing set size:    {len(test_set.raw_data['time'])}")

    x_train, mean, std = train_set.standardize()
    x_val = val_set.standardize(mean, std)
    x_test = test_set.standardize(mean, std)

    torch.save(
        (
            x_train,
            train_set.coordinate,
            train_set.y,
            train_set.mbh,
            train_set.y_baseline,
            train_set.groups,
            train_set.n_groups,
        ),
        cache_path + "train.pt",
    )
    torch.save(
        (x_val, val_set.coordinate, val_set.y, val_set.mbh, val_set.y_baseline, val_set.groups, val_set.n_groups),
        cache_path + "val.pt",
    )
    torch.save(
        (
            x_test,
            test_set.coordinate,
            test_set.y,
            test_set.mbh,
            test_set.y_baseline,
            test_set.groups,
            test_set.n_groups,
        ),
        cache_path + "test.pt",
    )

    return mean, std


from RawDataProcessing.ParseLogFile import load_config

configs = load_config(".config")
workspace_root = configs["BaseConfig"]["workspace_root"]

resolution = "coarse"  # 'coarse' or 'fine'
Exp_eg_folder_paths = [
    (os.path.join(workspace_root, "Data", "elliptical_galaxy_fiducial", resolution), np.array([1, 1])),
    (os.path.join(workspace_root, "Data", "elliptical_galaxy_0dot1", resolution), np.array([1, 2])),
    (os.path.join(workspace_root, "Data", "elliptical_galaxy_pgc", resolution), np.array([1, 3])),
]
Exp_dg_folder_paths = [
    (os.path.join(workspace_root, "Data", "disk_galaxy_fiducial", resolution), np.array([2, 1])),
    (os.path.join(workspace_root, "Data", "disk_galaxy_supplement", resolution), np.array([2, 1])),
    (os.path.join(workspace_root, "Data", "disk_galaxy_fiducial_4", resolution), np.array([2, 2])),
    (os.path.join(workspace_root, "Data", "disk_galaxy_fiducial_7", resolution), np.array([2, 3])),
    (os.path.join(workspace_root, "Data", "disk_galaxy_fiducial_10", resolution), np.array([2, 4])),
    (os.path.join(workspace_root, "Data", "disk_galaxy_low", resolution), np.array([2, 5])),
]

Exp_all_folder_paths = Exp_eg_folder_paths + Exp_dg_folder_paths


if __name__ == "__main__":

    cache_path_exp1 = os.path.join(".cache", "Exp1_")
    cache_path_exp2 = os.path.join(".cache", "Exp2_")
    cache_path_exp3 = os.path.join(".cache", "Exp3_")

    # cache_pt(Exp_eg_folder_paths, cache_path_exp1, balance="oversample")
    cache_pt(Exp_dg_folder_paths, cache_path_exp2, stratify="groups", balance="oversample")
    cache_pt(Exp_all_folder_paths, cache_path_exp3, balance="oversample")

    # for i, folder_paths in enumerate([Exp_eg_folder_paths, Exp_dg_folder_paths, Exp_all_folder_paths]):
    #     cache_path = os.path.join(".cache", f"Exp{i+1}_")
    #     cache_pt(folder_paths, cache_path, balance="oversample")
