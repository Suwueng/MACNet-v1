# Galaxy Data Processing Module
# This module provides classes and functions to read, process, and manage galaxy simulation data.
# It includes functionality to read snapshot data, compute derived quantities,
# rescale data, and manage datasets for machine learning applications.
# Author: Peng Cheng

import os
import warnings

import h5py
import numpy as np

from typing import Callable, List, Tuple
from pyhdf.SD import SD


def _validate_file_path(file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    return file_path


# =============== Galaxy Data Class ===============


class GalData:
    def __init__(self, ndim: int = 2, coordinate_mode: str = "polar"):
        """
        Initializes the GalData object.
        Args:
            ndim (int): Dimensionality of the data, either 2 or 3. Default is 2.
            coordinate_mode (str): Coordinate system mode, either 'polar'('p') or 'cartesian'('c'). Default is 'polar'.
        """
        coordinate_mode = coordinate_mode.lower()
        self._coordinate_mode = {
            "polar": "polar",
            "p": "polar",
            "cartesian": "cartesian",
            "c": "cartesian",
        }.get(coordinate_mode)
        if self._coordinate_mode is None:
            raise ValueError("Invalid coordinate mode. Choose 'polar' or 'cartesian'.")

        if ndim not in [2, 3]:
            raise ValueError("ndim must be either 2 or 3.")
        self.ndim = ndim
        self.time = None
        self._data_hdfra = None
        self._data_log = None
        self._corr = None
        self._axes = None
        self._coord = None

    # -------- Properties --------

    @property
    def snapshot(self) -> dict:
        if isinstance(self._data_hdfra, dict):
            return self._data_hdfra

        data = {}
        for original_name, new_name in self._corr:
            arr = self._data_hdfra.select(original_name)[:]
            data[new_name] = arr[0, ...]
        data["temperature"] = 93 * data["gas_energy"] / data["density"]
        return data

    @property
    def snapshot_array(self) -> np.ndarray:
        return np.array(list(self.snapshot.values()))

    @property
    def raw_axes(self) -> list:
        self.__validate_coords()
        return self._axes

    @property
    def broadcast_coords(self) -> np.ndarray:
        self.__validate_coords()
        return self._coord

    @property
    def coordinate(self) -> Tuple[np.ndarray, ...]:
        self.__validate_coords()
        return np.meshgrid(*self._axes)

    def axes(self) -> list:
        self.__validate_coords()
        return self._axes

    @property
    def r(self) -> np.ndarray:
        self.__validate_coords()
        if self._coordinate_mode == "polar":
            return self._coord[0]
        else:  # Cartesian -> spherical radius
            return np.sqrt(sum(c**2 for c in self._coord))

    @property
    def theta(self) -> np.ndarray:
        self.__validate_coords()
        if self._coordinate_mode == "polar":
            return self._coord[1]
        else:
            if self.ndim == 2:
                return np.arctan2(self._coord[1], self._coord[0])
            else:
                return np.arccos(self._coord[2] / self.r)

    @property
    def phi(self) -> np.ndarray:
        if self.ndim == 2:
            raise ValueError("Phi coordinate not available in 2D data.")
        self.__validate_coords()
        if self._coordinate_mode == "polar":
            return self._coord[2]
        else:
            return np.arctan2(self._coord[1], self._coord[0])

    @property
    def x(self) -> np.ndarray:
        self.__validate_coords()
        if self._coordinate_mode == "cartesian":
            return self._coord[0]
        if self.ndim == 2:
            return self.r * np.sin(self.theta)
        else:
            return self.r * np.sin(self.theta) * np.cos(self.phi)

    @property
    def y(self) -> np.ndarray:
        self.__validate_coords()
        if self._coordinate_mode == "cartesian":
            return self._coord[1]
        if self.ndim == 2:
            return self.r * np.cos(self.theta)
        else:
            return self.r * np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self) -> np.ndarray:
        if self.ndim == 2:
            raise ValueError("Z coordinate not available in 2D data.")
        self.__validate_coords()
        if self._coordinate_mode == "cartesian":
            return self._coord[2]
        else:
            return self.r * np.cos(self.theta)

    @property
    def grid_volume(self) -> np.ndarray:
        self.__validate_coords()

        spacings = []
        for ax in self._axes:
            edges = 0.5 * (ax[1:] + ax[:-1])
            first_edge = ax[0] - (edges[0] - ax[0])
            last_edge = ax[-1] + (ax[-1] - edges[-1])
            edges = np.concatenate(([first_edge], edges, [last_edge]))
            d = np.diff(edges)
            spacings.append(d)

        spacings_mesh = np.meshgrid(*spacings)

        dV = np.prod(spacings_mesh, axis=0)
        if self._coordinate_mode == "cartesian":
            return dV
        else:  # polar/spherical
            r = spacings_mesh[0]
            dV *= r ** (self.ndim - 1)

            if self.ndim == 3:
                theta = spacings_mesh[1]
                dV *= np.sin(theta)
            return dV

    @property
    def gas_mass(self) -> float:
        return self.snapshot["density"] * self.grid_volume

    @property
    def mbh(self) -> float:
        return self._data_log["mbh"]

    @property
    def mdot_macer(self) -> float:
        return self._data_log["mdot_macer"]

    @property
    def mdot_edd(self) -> float:
        return self._data_log["mdot_edd"]

    # -------- Calculation Methods --------

    def mdot_bondi(self, r_acc: float = 1.0, G: float = 112.0, gamma: float = 5 / 3) -> float:
        """
        Calculates the Bondi accretion rate.
        Args:
            r_acc (float): Accretion radius within which to compute the Bondi rate (kpc). Default is 1.0 kpc.
            G (float): Gravitational constant in the simulation units. Default is 112.  (kpc^3 / (2.5e7 Msun * Gyr^2))
            gamma (float): Adiabatic index of the gas. Default is 5/3
        """
        gamma1 = gamma - 1

        # lambda_c stable calculation
        if abs(gamma - 5 / 3) < 1e-8:
            lambda_c = 0.25
        else:
            denom = 5.0 - 3.0 * gamma
            if abs(denom) < 1e-12:
                raise ValueError("gamma too close to 5/3 causing numerical instability in lambda_c.")
            exponent = (5.0 - 3.0 * gamma) / (2.0 * gamma1)
            lambda_c = 0.25 * (2.0 / denom) ** exponent

        mask = self.r < r_acc
        effective_r_acc = r_acc
        if not np.any(mask):
            r_values = np.sort(np.unique(self.r.reshape(-1)))
            idx = np.searchsorted(r_values, r_acc)
            if idx >= len(r_values):
                idx = len(r_values) - 1
            effective_r_acc = float(r_values[max(idx, 0)])
            mask = self.r <= effective_r_acc

            if not np.any(mask):
                raise ValueError("No cells found within r_acc even after expanding to the smallest available radius.")

            warnings.warn(
                (
                    "No grid cells lie within the requested r_acc=%.3f kpc; "
                    "using the innermost available radius at %.3f kpc instead."
                )
                % (r_acc, effective_r_acc),
                RuntimeWarning,
            )

        # Broadcast mask to match the shape of density and other snapshot data
        density_shape = self.snapshot["density"].shape
        mask_broadcasted = np.broadcast_to(mask, density_shape)

        rho = self.snapshot["density"][mask_broadcasted]
        e = self.snapshot["gas_energy"][mask_broadcasted]
        cs = np.sqrt((gamma * gamma1) * e / rho)
        m = self.gas_mass[mask_broadcasted]
        M_bh = self.mbh

        # Weighted averages (default: mass-weighted)
        rho_inf = np.sum(m * rho) / np.sum(m)
        cs_inf = np.sum(m * cs) / np.sum(m)

        return 4 * np.pi * lambda_c * G**2 * M_bh**2 * rho_inf / cs_inf**3

    # -------- Public Methods --------

    def read_hdfra(
        self,
        hdfra_path: str,
        time: float,
        log_path: str,
        func_parse_log: Callable,
        scope: float,
    ) -> "GalData":
        """
        Reads snapshot data and associated BH accretion rates from specified paths.
        Args:
            hdfra_path (str): Path to the hdfra.xxxxx file including the simulation snapshot.
            time (float): The simulation time corresponding to the snapshot to be read (Gyr).
            log_path (str): Path to the log file including the BH accretion accretion rate.
            func_parse_log (Callable): Function to parse the log file into readable data.
            scope (float): Time scope for filtering log data. If provided, only entries within [time - scope, time + scope] are retained.
        """
        self._load_snapshot_hdfra(hdfra_path, time)._load_log(log_path, func_parse_log, scope)
        return self

    def rescale(self, new_shape: Tuple[int, ...], weights: List[np.ndarray | None] = None) -> "GalData":
        """
        Returns a new GalData instance with snapshot data rescaled to new_shape.

        Args:
            new_shape (Tuple[int, ...]): Desired shape for each snapshot variable.
            weights (List[np.ndarray | None], optional): List of weight arrays for each variable. If None, sum will be used for that variable. If not provided, defaults to None for all variables. If existing keys in snapshot are not in weights, they will default to mass-weighted averaging.
                Length of weights must match the number of variables in the snapshot.

        Returns:
            GalyData: A new instance with rescaled snapshot data and corresponding axes.
        """
        self.__validate_coords()
        if len(new_shape) != self.ndim:
            raise ValueError(f"New shape must have {self.ndim} dimensions, got {len(new_shape)}.")

        new_instance = GalData(ndim=self.ndim, coordinate_mode=self._coordinate_mode)
        new_instance.time = self.time
        new_instance._corr = self._corr.copy()
        new_instance._data_log = self._data_log.copy() if self._data_log is not None else None

        new_snapshot = {}
        for i, (name, data) in enumerate(self.snapshot.items()):
            weight = weights[name] if name in weights else self.gas_mass
            new_snapshot[name] = self._scale_transform(data, new_shape, weight)
        new_snapshot["temperature"] = 93 * new_snapshot["gas_energy"] / new_snapshot["density"]
        new_instance._data_hdfra = new_snapshot

        new_axes = []
        for old_axis, n in zip(self._axes, new_shape):
            edges = 0.5 * (old_axis[1:] + old_axis[:-1])
            first_edge = old_axis[0] - (edges[0] - old_axis[0])
            last_edge = old_axis[-1] + (old_axis[-1] - edges[-1])
            edges = np.concatenate(([first_edge], edges, [last_edge]))

            factor = len(old_axis) // n
            new_edges = edges[::factor]
            if len(new_edges) < n + 1:
                new_edges = np.append(new_edges, edges[-1])

            new_axis = 0.5 * (new_edges[1:] + new_edges[:-1])
            new_axes.append(new_axis)

        new_instance._axes = new_axes
        new_instance._coord = np.ix_(*new_axes)

        return new_instance

    def set_corr(self, corr: List[Tuple[str, str]] | Tuple[str, str]) -> "GalData":
        """
        Sets the correlation mapping for hdfra file data to new names.
        Note: This class requires the presence of 'density' and 'gas_energy' in the snapshot data.
        Args:
            corr (List[Tuple[str, str]] | Tuple[str, str]): List of tuples or a single tuple where each tuple contains
                the original dataset name and the new variable name.
        """
        self._corr = [corr] if isinstance(corr, tuple) else corr
        return self

    def set_coord(self, coords: List[str] | str) -> "GalData":
        """
        Sets the correlation mapping for coordinate data in hdfra file to new names.
        Args:
            coords (List[str] | str): List of coordinate names or a single coordinate name.
        Note:
            the order of coordinates in the list must match the order of the coordinate system.
            2D polar: ["r", "theta"]
            2D cartesian: ["x", "y"]
            3D polar: ["r", "theta", "phi"]
            3D cartesian: ["x", "y", "z"]
            "theta" is the polar angle in 2D/3D, and "phi" is the azimuthal angle in 3D.
        """
        if isinstance(coords, str):
            coords = [coords]
        if len(coords) != self.ndim:
            raise ValueError(f"Coordinate count mismatch: expected {self.ndim}, got {len(coords)}.")
        self._axes = [self._data_hdfra.select(c)[:] for c in coords]
        self._coord = np.ix_(*self._axes)
        return self

    def save_h5(self, file_path: str) -> None:
        """
        Saves core data of GalData instance to an HDF5 file.
        Args:
            file_path (str): Path to the output HDF5 file.
        """
        with h5py.File(file_path, "w") as f:
            snap_grp = f.create_group("snapshot")
            for name, data in self.snapshot.items():
                snap_grp.create_dataset(name, data=data)

            coord_grp = f.create_group("coordinates")
            for i, ax in enumerate(self._axes):
                coord_grp.create_dataset(f"axis_{i}", data=ax)
            coord_grp.attrs["coordinate_mode"] = self._coordinate_mode
            coord_grp.attrs["ndim"] = self.ndim

            log_grp = f.create_group("log_data")
            for key, value in self._data_log.items():
                log_grp.create_dataset(key, data=value)

            f.attrs["time"] = self.time

            corr_grp = f.create_group("correlation")
            corr_grp.attrs["length"] = len(self._corr)
            for i, (orig, new) in enumerate(self._corr):
                corr_grp.attrs[f"{i}_orig"] = orig
                corr_grp.attrs[f"{i}_new"] = new

            # print(f"GalData saved to HDF5: {file_path}")

    def load_h5(self, file_path: str) -> "GalData":
        """

        Loads GalData instance from an HDF5 file.
        Args:
            file_path (str): Path to the input HDF5 file.
        """
        with h5py.File(file_path, "r") as f:
            ndim = f["coordinates"].attrs["ndim"]
            coordinate_mode = f["coordinates"].attrs["coordinate_mode"]
            self.ndim = ndim
            self.coordinate_mode = coordinate_mode

            self.time = f.attrs["time"]

            self._data_hdfra = {}
            for name in f["snapshot"]:
                self._data_hdfra[name] = f["snapshot"][name][()]

            self._axes = [f["coordinates"][ax][()] for ax in f["coordinates"] if ax.startswith("axis_")]
            self._coord = np.ix_(*self._axes)

            if "log_data" in f:
                self._data_log = {k: f["log_data"][k][()] for k in f["log_data"]}

            corr_grp = f["correlation"]
            self._corr = [
                (corr_grp.attrs[f"{i}_orig"], corr_grp.attrs[f"{i}_new"]) for i in range(corr_grp.attrs["length"])
            ]

        # print(f"GalData loaded from HDF5: {file_path}")
        return self

    # -------- Private Methods --------

    def _load_snapshot_hdfra(self, path: str, time: float) -> "GalData":
        """
        Reads snapshot data from the specified path.
        Args:
            path (str): Path to the hdfra.xxxxx file including the simulation snapshot.
            time (float): The simulation time corresponding to the snapshot to be read (Gyr).
        """
        # Validate file path
        path = _validate_file_path(path)
        self.time = time
        self._data_hdfra = SD(path)
        return self

    def _load_log(self, path: str, f: Callable, scope: float) -> "GalData":
        """
        Reads BH accretion rates from the specified log file.
        Args:
            path (str): Path to the log file including the BH accretion accretion rate.
            f (Callable): Function to parse the log file into readable data.
                - Accept parameters (path: str, time: float, scope: float).
                - Return a structured data object (e.g., pandas DataFrame) containing at least the columns 'time', 'mbh', 'mdot_macer', and 'mdot_edd'.
            scope (float, optional): Time scope for filtering log data. If provided, only entries within [time - scope, time + scope] are retained.
        """
        path = _validate_file_path(path)
        if self.time is None:
            raise ValueError("Simulation time not set. Please call load_snapshot_hdfra() first.")
        self._data_log = f(path, time=self.time, scope=scope)
        return self

    def _scale_transform(self, data: np.ndarray, new_shape: Tuple[int, ...], weights: np.ndarray = None) -> np.ndarray:
        """
        Rescales the input data array to a new shape using specified method.
        Args:
            data (np.ndarray): Input data array to be rescaled.
            new_shape (Tuple[int, ...]): Desired shape of the output array.
            weights (np.ndarray, optional): Weights for weighted averaging. If None, sum will be used.
        """
        old_shape = np.array(data.shape)
        new_shape = np.array(new_shape)

        if len(old_shape) != len(new_shape):
            raise ValueError("Old and new shapes must have the same number of dimensions.")

        if not np.all(old_shape % new_shape == 0):
            raise ValueError("Current function only supports rescaling where new shape must divide old shape exactly.")

        factors = (old_shape // new_shape).astype(int)

        reshaped = data.reshape(*[n for pair in zip(new_shape, factors) for n in pair])

        axes = tuple(range(1, 2 * len(new_shape), 2))
        if weights is None:
            reshaped = reshaped.sum(axis=axes)
        else:
            w_reshaped = weights.reshape(*[n for pair in zip(new_shape, factors) for n in pair])
            reshaped = (reshaped * w_reshaped).sum(axis=axes) / w_reshaped.sum(axis=axes)

        return reshaped

    # ------- Validation Methods -------

    def __validate_coords(self) -> None:
        if self._coord is None:
            raise ValueError("Coordinates not set. Please call set_coord() first.")


# =============== Galaxy DataSet Class ===============


class GalDataSet:
    def __init__(self):
        """
        Initializes the GalDataSet object for managing multiple GalData instances.
        """
        self._raw_data = None
        self._group_labels = None
        self.folder_path = None
        self._x_keys = None
        self._coord_keys = None

    # --------- Properties --------

    @property
    def raw_data(self) -> dict:
        return self._raw_data

    @property
    def x(self) -> np.ndarray:
        self.__verify_data_loaded()
        return np.moveaxis(np.array([self._raw_data[k] for k in self._x_keys]), 0, 1)

    @property
    def y(self) -> np.ndarray:
        self.__verify_data_loaded()
        mdot_macer = np.array(self._raw_data["mdot_macer"])
        mdot_edd = np.array(self._raw_data["mdot_edd"])
        return np.log10(mdot_macer / mdot_edd)

    @property
    def mbh(self) -> np.ndarray:
        self.__verify_data_loaded()
        return np.array(self._raw_data["mbh"])

    @property
    def coordinate(self) -> np.ndarray:
        self.__verify_data_loaded()
        return np.moveaxis(np.array([self._raw_data[k] for k in self._coord_keys]), 0, 1)

    @property
    def time(self) -> np.ndarray:
        self.__verify_data_loaded()
        return np.array(self._raw_data["time"])

    @property
    def groups(self) -> np.ndarray:
        self.__verify_data_loaded()
        if self._group_labels is None:
            raise ValueError("Group labels not initialized. Please call load_data() first.")
        return np.asarray(self._group_labels, dtype=int)

    @property
    def n_groups(self) -> int:
        self.__verify_data_loaded()
        return len(np.unique(self.groups))

    @property
    def y_baseline(self) -> np.ndarray:
        self.__verify_data_loaded()
        mdot_bondi = np.array(self._raw_data["mdot_bondi"])
        mdot_edd = np.array(self._raw_data["mdot_edd"])
        return np.log10(mdot_bondi / mdot_edd)

    # --------- Dunder Methods --------

    def __len__(self):
        self.__verify_data_loaded()
        return len(self._raw_data[self._x_keys[0]])

    # -------- Public Methods --------

    def load_data(self, folder_path) -> "GalDataSet":
        """
        Loads data from the specified folder path(s).
        Args:
            folder_path (str | list[str]): Path to the folder(s) containing the data files.
        """
        # Handle folder paths
        folder_path = folder_path if isinstance(folder_path, list) else [folder_path]
        self.folder_path = [fp[0] for fp in folder_path]

        # Initialize raw data dictionary
        self._group_labels = []
        data = GalData()
        data_path = os.path.join(self.folder_path[0], os.listdir(self.folder_path[0])[0])
        data.load_h5(data_path)
        self._x_keys = list(data.snapshot.keys())
        self._coord_keys = [f"coordinate_{i}" for i in range(len(data.coordinate))]
        self._x_keys += ["gas_mass", "grid_volume"]
        keys = (
            self._x_keys
            + self._coord_keys
            + [
                "mbh",
                "mdot_macer",
                "mdot_edd",
                "time",
                "mdot_bondi",
            ]
        )
        self._raw_data = {k: [] for k in keys}

        # Load data from all specified folder paths
        for fp, group_idx in folder_path:
            for file_name in os.listdir(fp):
                file_path = os.path.join(fp, file_name)
                data = GalData().load_h5(file_path)
                for key, value in data.snapshot.items():
                    self._raw_data[key].append(value)
                for i, coord in enumerate(data.coordinate):
                    self._raw_data[f"coordinate_{i}"].append(coord)
                self._raw_data["gas_mass"].append(data.gas_mass)
                self._raw_data["grid_volume"].append(data.grid_volume)
                self._raw_data["mbh"].append(data.mbh)
                self._raw_data["mdot_macer"].append(data.mdot_macer)
                self._raw_data["mdot_edd"].append(data.mdot_edd)
                self._raw_data["time"].append(data.time)
                self._raw_data["mdot_bondi"].append(data.mdot_bondi())
                self._group_labels.append(group_idx)

        return self

    def drop_invalid(
        self,
        *,
        require_positive_targets: bool = True,
        require_positive_mbh: bool = True,
        require_positive_bondi: bool = True,
    ) -> "GalDataSet":
        """
        Removes samples whose target-related scalars contain NaN/Inf (or non-positive values if requested).
        Args:
            require_positive_targets (bool): Enforce mdot_macer and mdot_edd to be strictly positive.
            require_positive_mbh (bool): Enforce mbh to be strictly positive.
        """
        self.__verify_data_loaded()
        mdot_macer = np.asarray(self._raw_data["mdot_macer"], dtype=float)
        mdot_edd = np.asarray(self._raw_data["mdot_edd"], dtype=float)
        mdot_bondi = np.asarray(self._raw_data["mdot_bondi"], dtype=float)
        mbh = np.asarray(self._raw_data["mbh"], dtype=float)

        mask = np.isfinite(mdot_macer) & np.isfinite(mdot_edd) & np.isfinite(mdot_bondi) & np.isfinite(mbh)
        if require_positive_targets:
            mask &= (mdot_macer > 0.0) & (mdot_edd > 0.0)
        if require_positive_mbh:
            mask &= mbh > 0.0
        if require_positive_bondi:
            mask &= mdot_bondi > 0.0

        removed = np.count_nonzero(~mask)
        if removed:
            if removed == len(mask):
                raise ValueError("All samples are invalid after filtering; check the raw dataset.")
            self.__subset(np.flatnonzero(mask))
        return self

    def filter(self, threshold: float) -> "GalDataSet":
        """
        Filters out data points where the target variable y is below the specified threshold.
        Args:
            threshold (float): Threshold value for filtering y.
        """
        self.__verify_data_loaded()
        mask = self.y >= threshold
        indices = np.flatnonzero(mask)
        # If no samples pass the threshold, retain an empty dataset consistently
        return self.__subset(indices)

    def mirror_data(self) -> "GalDataSet":
        """
        Mirrors the data along the first coordinate axis (e.g., x-axis).
        """
        self.__verify_data_loaded()

        for key in self._x_keys:
            self._raw_data[key] = [np.hstack([np.fliplr(x), x]) for x in self._raw_data[key]]

        for key in self._coord_keys:
            self._raw_data[key] = [np.hstack([np.fliplr(x), x]) for x in self._raw_data[key]]
        return self

    def split(
        self,
        train_size: float,
        validation_size: float = 0.0,
        stratify: str | np.ndarray | None = None,
        *,
        n_bins: int = 10,
        random_state: int | None = None,
        shuffle: bool = True,
    ):
        """
        Split dataset into train / validation / test sets.

        stratify options:
            None        : random split
            "groups"    : stratify by self._group_labels
            "y"         : stratify by y (equal-frequency bins)
            array_like  : external 1D labels
        """
        self.__verify_data_loaded()

        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")
        if validation_size < 0 or train_size + validation_size >= 1:
            raise ValueError("validation_size must be non-negative and train_size + validation_size < 1.")

        rng = np.random.default_rng(random_state)
        indices = np.arange(len(self))

        labels = self._build_stratified_labels(stratify, n_bins)

        train_idx, val_idx, test_idx = [], [], []
        if labels is None:
            tr, va, te = self._split_group(indices, train_size, validation_size, rng)
            train_idx.append(tr)
            val_idx.append(va)
            test_idx.append(te)
        else:
            for lv in np.unique(labels):
                group_idx = indices[labels == lv]
                tr, va, te = self._split_group(group_idx, train_size, validation_size, rng)
                if tr.size:
                    train_idx.append(tr)
                if validation_size > 0 and va.size:
                    val_idx.append(va)
                if te.size:
                    test_idx.append(te)

        train_idx = np.concatenate(train_idx) if train_idx else np.empty(0, dtype=int)
        val_idx = np.concatenate(val_idx) if validation_size > 0 and val_idx else np.empty(0, dtype=int)
        test_idx = np.concatenate(test_idx) if test_idx else np.empty(0, dtype=int)

        if shuffle:
            rng.shuffle(train_idx)
            if validation_size > 0:
                rng.shuffle(val_idx)
            rng.shuffle(test_idx)

        train_set = self.__subset_by_idx(train_idx)
        val_set = self.__subset_by_idx(val_idx) if validation_size > 0 else None
        test_set = self.__subset_by_idx(test_idx)

        return (train_set, val_set, test_set) if validation_size > 0 else (train_set, test_set)

    def standardize(
        self, mean: np.ndarray = None, std: np.ndarray = None, log_transform: bool = True
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardizes the input features x to have zero mean and unit variance.
        Args:
            mean (np.ndarray, optional): Precomputed mean for each feature. If None, computed from data.
            std (np.ndarray, optional): Precomputed standard deviation for each feature. If None, computed from data.
            log_transform (bool): Whether to apply log1p transformation before standardization.
        """
        self.__verify_data_loaded()
        x = self.x
        if log_transform:
            x = np.log1p(x)

        if mean is None or std is None:
            mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            std = np.std(x, axis=(0, 2, 3), keepdims=True)
            std_safe = np.where(std == 0, 1.0, std)
            return (x - mean) / std_safe, mean, std_safe
        elif mean is not None and std is not None:
            std_safe = np.where(std == 0, 1, std)
            return (x - mean) / std_safe
        else:
            raise ValueError("Both mean and std must be provided for standardization.")

    def balance_groups(self, method: str = "oversample") -> "GalDataSet":
        """
        Balance samples across groups by random over- or under-sampling.

        Args:
            method: "oversample" (with replacement) or "undersample" (without replacement)
        """
        self.__verify_data_loaded()

        if method not in {"oversample", "undersample"}:
            raise ValueError("method must be 'oversample' or 'undersample'")

        # Collect indices by normalized group label
        def normalize_label(label):
            arr = np.asarray(label)
            return arr.item() if arr.ndim == 0 else tuple(arr.tolist())

        indices_by_group = {}
        for i, label in enumerate(self._group_labels):
            key = normalize_label(label)
            indices_by_group.setdefault(key, []).append(i)

        groups = list(indices_by_group.keys())
        group_sizes = np.array([len(indices_by_group[g]) for g in groups], dtype=int)

        # Determine target size per group
        target_size = group_sizes.max() if method == "oversample" else group_sizes.min()
        replace = method == "oversample"

        # Resample each group to target size
        sampled_indices = [np.random.choice(indices_by_group[g], target_size, replace=replace) for g in groups]

        all_indices = np.concatenate(sampled_indices)
        np.random.shuffle(all_indices)

        return self.__subset(all_indices)

    # ------- Private Methods -------

    def _build_stratified_labels(self, stratify: str | np.ndarray | None, n_bins: int) -> np.ndarray | None:
        """
        Constructs stratification labels based on the specified method.
        """
        if stratify is None:
            return None

        if isinstance(stratify, str):
            if stratify == "groups":
                return np.array(
                    [
                        tuple(np.asarray(g).tolist()) if np.asarray(g).ndim > 0 else np.asarray(g).item()
                        for g in self._group_labels
                    ],
                    dtype=object,
                )

            if stratify == "y":
                y = self.y
                quantiles = np.linspace(0, 1, n_bins + 1)
                edges = np.unique(np.quantile(y, quantiles))
                if edges.size < 2:
                    edges = np.array([y.min() - 1e-9, y.max() + 1e-9])
                return np.digitize(y, edges[1:-1], right=False)

            raise ValueError("stratify must be None, 'groups', 'y', or a 1D array.")

        labels = np.asarray(stratify)
        if labels.shape != (len(self),):
            raise ValueError("Provided stratify array must be 1D and match dataset length.")
        return labels

    def _split_group(self, group_idx: np.ndarray, train_size: float, validation_size: float, rng: np.random.Generator):
        """
        Splits the provided group indices into train, validation, and test sets.
        """
        if self.shuffle:
            rng.shuffle(group_idx)

        n = len(group_idx)
        n_train = int(train_size * n)
        n_val = int(validation_size * n) if validation_size > 0 else 0

        # guard against rounding overflow
        excess = n_train + n_val - n
        if excess > 0:
            reduce_val = min(n_val, excess)
            n_val -= reduce_val
            n_train = max(0, n_train - (excess - reduce_val))

        train = group_idx[:n_train]
        val = group_idx[n_train : n_train + n_val]
        test = group_idx[n_train + n_val :]
        return train, val, test

    # ------- Validation Methods -------

    def __verify_data_loaded(self) -> None:
        if self._raw_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

    def __subset(self, indices: np.ndarray) -> "GalDataSet":
        """
        Retains only the samples specified by indices across all stored fields.
        """
        keep = indices.tolist()
        for key, values in self._raw_data.items():
            self._raw_data[key] = [values[i] for i in keep]
        if self._group_labels is not None:
            self._group_labels = [self._group_labels[i] for i in keep]
        return self

    def __subset_by_idx(self, idx: np.ndarray) -> "GalDataSet":
        """
        Returns a new GalDataSet instance containing only the samples specified by idx.
        """
        ds = GalDataSet()
        ds._x_keys = self._x_keys.copy()
        ds._coord_keys = self._coord_keys.copy()
        ds._raw_data = {k: [self._raw_data[k][i] for i in idx] for k in self._raw_data}
        ds._group_labels = [self._group_labels[i] for i in idx]
        ds.folder_path = self.folder_path.copy() if self.folder_path is not None else None
        return ds
