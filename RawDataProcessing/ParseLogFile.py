import os
import re
import numpy as np
import pandas as pd
from typing import List, Iterable


# ======== Low-level utilities ========
_FORTRAN_EXP_RE = re.compile(r"(\d+(?:\.\d+)?)([+-]\d+)")


def fix_fortran_number(token: str) -> str:
    """Convert Fortran-style scientific notation to standard float notation."""
    token = token.replace("D", "E")
    return _FORTRAN_EXP_RE.sub(r"\1E\2", token)


def detect_fortran_style(path: str, max_lines: int = 100) -> bool:
    """Detect whether a file uses Fortran-style scientific notation."""
    with open(path) as f:
        for _, line in zip(range(max_lines), f):
            if "D" in line or _FORTRAN_EXP_RE.search(line):
                return True
    return False


# ======== Parsing ========


def _parse_fortran_log(path: str, colnames: List[str]) -> pd.DataFrame:
    rows: list[list[float]] = []

    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            payload = line.split("=", 1)[-1]
            tokens = payload.split()
            values = [float(fix_fortran_number(tok)) for tok in tokens]
            rows.append(values)

    if not rows:
        raise ValueError(f"No valid data rows found in {path}")

    ncols = max(len(r) for r in rows)
    columns = _normalize_colnames(colnames, ncols)

    return pd.DataFrame(rows, columns=columns)


def _parse_plain_log(path: str, colnames: List[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        names=colnames,
        comment="#",
        engine="python",
    )


def _normalize_colnames(colnames: List[str], ncols: int) -> List[str]:
    if len(colnames) < ncols:
        return colnames + [f"col{i}" for i in range(len(colnames), ncols)]
    return colnames[:ncols]


def parse_logfile(
    path: str, colnames: List[str], save_path: str | None = None, force_parse: bool = False
) -> pd.DataFrame:
    """
    Parse a log file into a DataFrame, optionally cached as parquet.
    """
    if save_path and os.path.exists(save_path) and not force_parse:
        return pd.read_parquet(save_path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    is_fortran = detect_fortran_style(path)
    parser = _parse_fortran_log if is_fortran else _parse_plain_log
    df = parser(path, colnames)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)

    return df


# ======== Time-window Analysis ========


def mean_around_time(df: pd.DataFrame, time: float, scope: float, *, time_col: str = "time") -> pd.Series:
    if time_col not in df.columns:
        raise KeyError(f"Missing required column '{time_col}'")

    mask = (df[time_col] >= time - scope) & (df[time_col] <= time + scope)
    if not mask.any():
        raise ValueError("No data points in the specified time window")

    avg = df.loc[mask].mean()
    avg[time_col] = time
    return avg


def _require_columns(data: pd.Series | pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in data]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# ======== Physics-specific Helpers ========


def ll_eg(path: str, time: float, scope: float) -> pd.Series:
    df = pd.read_parquet(path)

    avg = mean_around_time(df, time, scope)
    _require_columns(avg, ["mdot_edd", "mdot_ratio"])

    avg["mdot_macer"] = avg["mdot_edd"] * avg["mdot_ratio"]
    return avg


def ll_dg(path: str, time: float, scope: float) -> pd.Series:
    df = pd.read_parquet(path)

    avg = mean_around_time(df, time, scope)
    _require_columns(avg, ["mdot_bh", "mbh"])

    avg["mdot_macer"] = avg["mdot_bh"]

    # constants
    G = 112
    m_p = 3.365e-65
    eta = 0.1
    c = 3.07e5
    sigma_t = 6.99e-68

    avg["mdot_edd"] = (4 * np.pi * G * avg["mbh"] * m_p) / (eta * c * sigma_t)

    return avg
