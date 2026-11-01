#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Support-matched allometric calibration for TropiSAR–Paracou.

Scientific purpose
------------------
This module replaces the previous 3×3-pixel CHM sampling with a spatially
matched extraction over each complete 50×50 m inventory cell. It uses the
canonical CHM reconstructed by the current TCCT pipeline:

    Dados_Processados/LiDAR_CHM/
    CHM_Paracou_2009_DSM_menos_DTM_validado.tif

The field inventory contains 475 cells distributed among 16 plots
(plots 1–15: 25 cells each; plot 16: 100 cells). Biomass reported in tonnes
per 50×50 m cell is converted to Mg/ha by multiplying by four.

Primary model
-------------
    AGB = a * H_mean ** b

where H_mean is the mean canonical CHM over the full 50×50 m cell. The model
is fitted by uncertainty-weighted nonlinear least squares. Generalization is
measured by leave-one-plot-out (LOPO) validation, so cells from the evaluated
plot never participate in calibration.

Important scale rule
--------------------
The fitted equation is support-specific. AGB maps are therefore produced only
after aggregating CHM to a regular 50 m support. The equation is never applied
directly to individual 1 m pixels.

Usage
-----
1. Place this file in SIBGRAPI_REV/Codigo.
2. Place CSV_50.zip (or the extracted CSV_50 folder) anywhere inside
   SIBGRAPI_REV. The script discovers it automatically.
3. Run:

       python TCCT_AGB_Allometry_SupportMatched.py

Optional predicted CHM map:

       python TCCT_AGB_Allometry_SupportMatched.py \
           --predicted-chm <LOCAL_PATH>

The predicted CHM is aggregated to the same 50 m grid before AGB conversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds
from scipy.optimize import curve_fit


SCRIPT_VERSION = "1.0.0-support-matched"
CANONICAL_CHM_RELATIVE = Path(
    "Dados_Processados/LiDAR_CHM/"
    "CHM_Paracou_2009_DSM_menos_DTM_validado.tif"
)
DEFAULT_OUTPUT_RELATIVE = Path(
    "Resultados/TCCT_v3_2_SIBGRAPI/AGB_Allometry_SupportMatched"
)
EXPECTED_PLOTS = tuple(range(1, 17))
EXPECTED_TOTAL_CELLS = 475
PRIMARY_SUPPORT_M = 50.0
MIN_VALID_FRACTION = 0.80
MIN_MEAN_CHM_M = 2.0
MAX_VALID_CHM_M = 60.0
PREVIOUS_A = 162.17268
PREVIOUS_B = 0.24862


@dataclass(frozen=True)
class InventoryCell:
    plot_id: int
    cell_id: int
    source_file: str
    x: float
    y: float
    biomass_tons_cell: float
    biomass_sd_tons_cell: float
    quantile_2_5_tons_cell: float | None
    quantile_upper_tons_cell: float | None

    @property
    def agb_mg_ha(self) -> float:
        return self.biomass_tons_cell * 4.0

    @property
    def agb_sd_mg_ha(self) -> float:
        return self.biomass_sd_tons_cell * 4.0


@dataclass(frozen=True)
class GridBasis:
    u: np.ndarray
    v: np.ndarray
    length_u: float
    length_v: float
    angle_deg: float
    area_m2: float


@dataclass(frozen=True)
class FitResult:
    a: float
    b: float
    covariance: np.ndarray


class CalibrationError(RuntimeError):
    """Raised when an audit condition prevents a defensible calibration."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_project_root(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_root = os.environ.get("TCCT_SIBGRAPI_REV_DIR")
    if env_root:
        candidates.append(Path(env_root))

    script_path = Path(__file__).resolve()
    candidates.extend([script_path.parent, script_path.parent.parent])

    for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        candidates.append(Path(f"{drive}:/SIBGRAPI_REV"))
    candidates.extend(
        [
            Path("<PROJECT_ROOT>"),
            Path.cwd(),
            Path.cwd().parent,
        ]
    )

    checked: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        checked.append(str(resolved))
        if resolved.name == "SIBGRAPI_REV" and resolved.exists():
            return resolved
        nested = resolved / "SIBGRAPI_REV"
        if nested.exists() and nested.is_dir():
            return nested.resolve()

    raise FileNotFoundError(
        "SIBGRAPI_REV was not found. Set TCCT_SIBGRAPI_REV_DIR or pass "
        f"--project-root. Checked: {checked[:12]}"
    )


def discover_inventory_source(root: Path, explicit: str | None = None) -> Path:
    if explicit:
        source = Path(explicit).expanduser().resolve()
        if source.exists():
            return source
        raise FileNotFoundError(source)

    preferred = [
        root / "Auditoria/Inventario_TropiSAR/CSV_50.zip",
        root / "Auditoria/Inventario_TropiSAR/CSV_50",
        root / "Dados_Brutos/Inventario_TropiSAR/CSV_50.zip",
        root / "Dados_Brutos/Inventario_TropiSAR/CSV_50",
        root / "CSV_50.zip",
        root / "CSV_50",
    ]
    for path in preferred:
        if path.exists():
            return path.resolve()

    zip_matches = sorted(root.rglob("CSV_50.zip"))
    if len(zip_matches) == 1:
        return zip_matches[0].resolve()
    if len(zip_matches) > 1:
        raise CalibrationError(
            "More than one CSV_50.zip was found. Pass --inventory explicitly: "
            + ", ".join(str(p) for p in zip_matches)
        )

    folders = sorted(
        {
            p.parent.resolve()
            for p in root.rglob("Biomass_P*_50.csv")
            if p.is_file()
        }
    )
    if len(folders) == 1:
        return folders[0]
    if len(folders) > 1:
        raise CalibrationError(
            "Biomass CSV files were found in multiple folders. Pass --inventory: "
            + ", ".join(str(p) for p in folders)
        )

    raise FileNotFoundError(
        "CSV_50.zip or an extracted CSV_50 folder was not found inside "
        f"{root}."
    )


def _safe_float(value: object, name: str, source: str) -> float:
    try:
        result = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise CalibrationError(
            f"Invalid numeric value in {source}, column {name}: {value!r}"
        ) from exc
    if not np.isfinite(result):
        raise CalibrationError(
            f"Non-finite value in {source}, column {name}: {value!r}"
        )
    return result


def _parse_inventory_csv(text: str, source_name: str) -> list[InventoryCell]:
    match = re.search(r"Biomass_P(\d+)_50\.csv$", Path(source_name).name)
    if not match:
        return []
    plot_id = int(match.group(1))

    reader = csv.DictReader(io.StringIO(text))
    fields = list(reader.fieldnames or [])
    required = {"X", "Y", "Biomass_(tons)", "SD_Biomass"}
    missing = required - set(fields)
    if missing:
        raise CalibrationError(
            f"Missing columns in {source_name}: {sorted(missing)}. "
            f"Available columns: {fields}"
        )

    upper_candidates = [
        col for col in fields if col.lower().startswith("quant_") and col != "Quant_2.5"
    ]
    upper_column = upper_candidates[0] if upper_candidates else None

    rows: list[InventoryCell] = []
    for cell_id, row in enumerate(reader, start=1):
        q25 = row.get("Quant_2.5")
        qupper = row.get(upper_column) if upper_column else None
        rows.append(
            InventoryCell(
                plot_id=plot_id,
                cell_id=cell_id,
                source_file=Path(source_name).name,
                x=_safe_float(row["X"], "X", source_name),
                y=_safe_float(row["Y"], "Y", source_name),
                biomass_tons_cell=_safe_float(
                    row["Biomass_(tons)"], "Biomass_(tons)", source_name
                ),
                biomass_sd_tons_cell=_safe_float(
                    row["SD_Biomass"], "SD_Biomass", source_name
                ),
                quantile_2_5_tons_cell=(
                    _safe_float(q25, "Quant_2.5", source_name)
                    if q25 not in (None, "")
                    else None
                ),
                quantile_upper_tons_cell=(
                    _safe_float(qupper, upper_column or "upper quantile", source_name)
                    if qupper not in (None, "")
                    else None
                ),
            )
        )
    return rows


def load_inventory(source: Path) -> tuple[list[InventoryCell], dict]:
    cells: list[InventoryCell] = []
    provenance: dict = {
        "source": str(source),
        "source_type": "zip" if source.suffix.lower() == ".zip" else "folder",
    }

    if source.is_file() and source.suffix.lower() == ".zip":
        provenance["sha256"] = sha256_file(source)
        with zipfile.ZipFile(source) as archive:
            names = sorted(
                name
                for name in archive.namelist()
                if re.search(r"Biomass_P\d+_50\.csv$", Path(name).name)
            )
            for name in names:
                raw = archive.read(name)
                text = raw.decode("utf-8-sig")
                cells.extend(_parse_inventory_csv(text, name))
        provenance["files"] = names
    elif source.is_dir():
        files = sorted(source.rglob("Biomass_P*_50.csv"))
        for path in files:
            cells.extend(
                _parse_inventory_csv(
                    path.read_text(encoding="utf-8-sig"), str(path)
                )
            )
        provenance["files"] = [str(p) for p in files]
        provenance["sha256_by_file"] = {
            str(p): sha256_file(p) for p in files
        }
    else:
        raise CalibrationError(
            f"Inventory source must be CSV_50.zip or a folder: {source}"
        )

    if not cells:
        raise CalibrationError(f"No Biomass_P*_50.csv records were read from {source}")

    plots = sorted({cell.plot_id for cell in cells})
    counts = {plot: sum(cell.plot_id == plot for cell in cells) for plot in plots}
    if tuple(plots) != EXPECTED_PLOTS:
        raise CalibrationError(
            f"Expected plots 1–16, found {plots}. Counts: {counts}"
        )
    if len(cells) != EXPECTED_TOTAL_CELLS:
        raise CalibrationError(
            f"Expected {EXPECTED_TOTAL_CELLS} inventory cells, found {len(cells)}. "
            f"Counts: {counts}"
        )
    expected_counts = {plot: (100 if plot == 16 else 25) for plot in EXPECTED_PLOTS}
    if counts != expected_counts:
        raise CalibrationError(
            f"Unexpected number of cells by plot. Expected {expected_counts}; found {counts}."
        )

    coordinates = [(round(c.x, 6), round(c.y, 6)) for c in cells]
    if len(coordinates) != len(set(coordinates)):
        raise CalibrationError("Duplicated inventory cell centres were found.")

    provenance.update(
        {
            "n_cells": len(cells),
            "plots": plots,
            "cells_by_plot": counts,
            "agb_conversion": "Biomass_(tons) per 0.25 ha cell multiplied by 4",
        }
    )
    return cells, provenance


def _canonicalize_angle(theta: np.ndarray) -> np.ndarray:
    return np.mod(theta, np.pi)


def infer_grid_basis(points: np.ndarray, support_m: float = PRIMARY_SUPPORT_M) -> GridBasis:
    """Infers the two local grid vectors from adjacent 50 m cell centres."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise CalibrationError("At least four 2-D points are required to infer a grid.")

    vectors: list[np.ndarray] = []
    lower = support_m * 0.80
    upper = support_m * 1.20
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            vector = points[j] - points[i]
            distance = float(np.linalg.norm(vector))
            if lower <= distance <= upper:
                vectors.append(vector)
    if len(vectors) < 4:
        raise CalibrationError(
            f"Too few adjacent-centre vectors near {support_m} m: {len(vectors)}"
        )

    vectors_array = np.vstack(vectors)
    lengths = np.linalg.norm(vectors_array, axis=1)
    angles = _canonicalize_angle(
        np.arctan2(vectors_array[:, 1], vectors_array[:, 0])
    )
    doubled = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)])

    # Deterministic two-cluster k-means on doubled-angle unit vectors.
    centre_1 = doubled[0].copy()
    centre_2 = doubled[np.argmin(doubled @ centre_1)].copy()
    labels = np.zeros(len(doubled), dtype=int)
    for _ in range(50):
        d1 = np.sum((doubled - centre_1) ** 2, axis=1)
        d2 = np.sum((doubled - centre_2) ** 2, axis=1)
        new_labels = (d2 < d1).astype(int)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        if np.any(labels == 0):
            centre_1 = doubled[labels == 0].mean(axis=0)
            centre_1 /= np.linalg.norm(centre_1)
        if np.any(labels == 1):
            centre_2 = doubled[labels == 1].mean(axis=0)
            centre_2 /= np.linalg.norm(centre_2)

    basis_vectors: list[np.ndarray] = []
    basis_lengths: list[float] = []
    for cluster in (0, 1):
        mask = labels == cluster
        if mask.sum() < 2:
            raise CalibrationError("Grid direction clustering failed.")
        mean_doubled = doubled[mask].mean(axis=0)
        theta = 0.5 * math.atan2(mean_doubled[1], mean_doubled[0])
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        aligned_lengths = np.abs(vectors_array[mask] @ direction)
        length = float(np.median(aligned_lengths))
        basis_vectors.append(direction * length)
        basis_lengths.append(length)

    u, v = basis_vectors
    # Ensure v has the orientation closest to 90 degrees from u.
    angle = math.degrees(
        math.acos(
            float(
                np.clip(
                    abs(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)),
                    0.0,
                    1.0,
                )
            )
        )
    )
    angle_between = 90.0 - angle
    if angle_between > 8.0:
        raise CalibrationError(
            f"Inferred grid axes are not sufficiently orthogonal: {90-angle_between:.2f}°"
        )

    cross_uv = float(u[0] * v[1] - u[1] * v[0])
    if cross_uv < 0:
        v = -v

    length_u = float(np.linalg.norm(u))
    length_v = float(np.linalg.norm(v))
    if not (support_m * 0.90 <= length_u <= support_m * 1.10):
        raise CalibrationError(f"Unexpected grid spacing u={length_u:.3f} m")
    if not (support_m * 0.90 <= length_v <= support_m * 1.10):
        raise CalibrationError(f"Unexpected grid spacing v={length_v:.3f} m")

    actual_angle = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(u, v) / (length_u * length_v), -1.0, 1.0
                )
            )
        )
    )
    area = float(abs(u[0] * v[1] - u[1] * v[0]))
    if not (support_m**2 * 0.85 <= area <= support_m**2 * 1.15):
        raise CalibrationError(f"Unexpected inferred cell area: {area:.2f} m²")

    return GridBasis(
        u=u,
        v=v,
        length_u=length_u,
        length_v=length_v,
        angle_deg=actual_angle,
        area_m2=area,
    )


def cell_polygon(x: float, y: float, basis: GridBasis) -> dict:
    centre = np.array([x, y], dtype=float)
    corners = [
        centre - 0.5 * basis.u - 0.5 * basis.v,
        centre + 0.5 * basis.u - 0.5 * basis.v,
        centre + 0.5 * basis.u + 0.5 * basis.v,
        centre - 0.5 * basis.u + 0.5 * basis.v,
    ]
    coordinates = [[float(p[0]), float(p[1])] for p in corners]
    coordinates.append(coordinates[0])
    return {"type": "Polygon", "coordinates": [coordinates]}


def _bounded_window(dataset: rasterio.io.DatasetReader, polygon: dict) -> Window:
    coords = polygon["coordinates"][0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    window = from_bounds(min(xs), min(ys), max(xs), max(ys), dataset.transform)
    window = window.round_offsets().round_lengths()
    full = Window(0, 0, dataset.width, dataset.height)
    try:
        window = window.intersection(full)
    except rasterio.errors.WindowError as exc:
        raise CalibrationError("Inventory cell does not intersect the CHM raster.") from exc
    if window.width <= 0 or window.height <= 0:
        raise CalibrationError("Inventory cell has an empty CHM window.")
    return window


def extract_polygon_chm(
    dataset: rasterio.io.DatasetReader,
    polygon: dict,
) -> dict[str, float | int]:
    window = _bounded_window(dataset, polygon)
    data = dataset.read(1, window=window, masked=True).filled(np.nan).astype(float)
    transform = dataset.window_transform(window)
    inside = geometry_mask(
        [polygon],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    support_pixels = int(inside.sum())
    values = data[inside]
    valid = np.isfinite(values) & (values >= 0.0) & (values <= MAX_VALID_CHM_M)
    valid_values = values[valid]
    valid_fraction = float(valid.sum() / max(1, support_pixels))

    result: dict[str, float | int] = {
        "support_pixels": support_pixels,
        "valid_pixels": int(valid.sum()),
        "valid_fraction": valid_fraction,
    }
    if valid_values.size:
        result.update(
            {
                "chm_mean_m": float(np.mean(valid_values)),
                "chm_median_m": float(np.median(valid_values)),
                "chm_std_m": float(np.std(valid_values, ddof=1))
                if valid_values.size > 1
                else 0.0,
                "chm_p10_m": float(np.percentile(valid_values, 10)),
                "chm_p90_m": float(np.percentile(valid_values, 90)),
                "chm_min_m": float(np.min(valid_values)),
                "chm_max_m": float(np.max(valid_values)),
            }
        )
    else:
        result.update(
            {
                "chm_mean_m": math.nan,
                "chm_median_m": math.nan,
                "chm_std_m": math.nan,
                "chm_p10_m": math.nan,
                "chm_p90_m": math.nan,
                "chm_min_m": math.nan,
                "chm_max_m": math.nan,
            }
        )
    return result


def build_support_matched_table(
    cells: Sequence[InventoryCell],
    chm_path: Path,
) -> tuple[list[dict], dict[int, GridBasis], dict]:
    if not chm_path.exists():
        raise FileNotFoundError(chm_path)

    cells_by_plot: dict[int, list[InventoryCell]] = {}
    for cell in cells:
        cells_by_plot.setdefault(cell.plot_id, []).append(cell)

    bases: dict[int, GridBasis] = {}
    for plot_id, plot_cells in cells_by_plot.items():
        points = np.array([[c.x, c.y] for c in plot_cells], dtype=float)
        bases[plot_id] = infer_grid_basis(points)

    records: list[dict] = []
    with rasterio.open(chm_path) as chm:
        if chm.crs is None:
            raise CalibrationError("The canonical CHM has no CRS.")
        if chm.transform.is_identity:
            raise CalibrationError("The canonical CHM has an identity transform.")

        for cell in cells:
            basis = bases[cell.plot_id]
            polygon = cell_polygon(cell.x, cell.y, basis)
            stats = extract_polygon_chm(chm, polygon)
            accepted = (
                stats["valid_fraction"] >= MIN_VALID_FRACTION
                and np.isfinite(stats["chm_mean_m"])
                and stats["chm_mean_m"] >= MIN_MEAN_CHM_M
                and cell.agb_mg_ha > 0
                and cell.agb_sd_mg_ha > 0
            )
            records.append(
                {
                    "plot_id": cell.plot_id,
                    "cell_id": cell.cell_id,
                    "source_file": cell.source_file,
                    "x_center": cell.x,
                    "y_center": cell.y,
                    "biomass_tons_cell": cell.biomass_tons_cell,
                    "biomass_sd_tons_cell": cell.biomass_sd_tons_cell,
                    "agb_observed_mg_ha": cell.agb_mg_ha,
                    "agb_sd_mg_ha": cell.agb_sd_mg_ha,
                    "grid_u_x": float(basis.u[0]),
                    "grid_u_y": float(basis.u[1]),
                    "grid_v_x": float(basis.v[0]),
                    "grid_v_y": float(basis.v[1]),
                    "cell_area_inferred_m2": basis.area_m2,
                    **stats,
                    "accepted_for_calibration": bool(accepted),
                }
            )

        raster_info = {
            "path": str(chm_path),
            "sha256": sha256_file(chm_path),
            "crs": str(chm.crs),
            "transform": list(chm.transform)[:6],
            "shape": [chm.height, chm.width],
            "resolution_m": [abs(chm.transform.a), abs(chm.transform.e)],
            "bounds": [chm.bounds.left, chm.bounds.bottom, chm.bounds.right, chm.bounds.top],
            "nodata": chm.nodata,
        }

    accepted_n = sum(bool(r["accepted_for_calibration"]) for r in records)
    if accepted_n < 0.90 * len(records):
        raise CalibrationError(
            f"Only {accepted_n}/{len(records)} cells passed support/CHM checks. "
            "Inspect CRS, cell-centre interpretation and CHM coverage before fitting."
        )
    return records, bases, raster_info


def power_model(height: np.ndarray | float, a: float, b: float) -> np.ndarray:
    h = np.asarray(height, dtype=float)
    return a * np.power(np.clip(h, 1.0e-6, None), b)


def fit_power_model(
    height: np.ndarray,
    agb: np.ndarray,
    agb_sd: np.ndarray,
) -> FitResult:
    height = np.asarray(height, dtype=float)
    agb = np.asarray(agb, dtype=float)
    agb_sd = np.asarray(agb_sd, dtype=float)
    mask = (
        np.isfinite(height)
        & np.isfinite(agb)
        & np.isfinite(agb_sd)
        & (height > 0)
        & (agb > 0)
        & (agb_sd > 0)
    )
    x, y, sigma = height[mask], agb[mask], agb_sd[mask]
    if len(x) < 20:
        raise CalibrationError(f"Too few valid calibration cells: {len(x)}")

    # Log-linear initial values only; final optimization is in the original scale.
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    p0 = [float(np.exp(intercept)), float(slope)]
    p0[0] = float(np.clip(p0[0], 1.0e-3, 1.0e4))
    p0[1] = float(np.clip(p0[1], 0.02, 4.0))

    parameters, covariance = curve_fit(
        power_model,
        x,
        y,
        p0=p0,
        sigma=np.maximum(sigma, 1.0),
        absolute_sigma=True,
        bounds=([1.0e-8, 0.01], [1.0e5, 5.0]),
        maxfev=200_000,
    )
    return FitResult(
        a=float(parameters[0]),
        b=float(parameters[1]),
        covariance=np.asarray(covariance, dtype=float),
    )


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    y, p = observed[mask], predicted[mask]
    if len(y) == 0:
        raise CalibrationError("No finite observations were available for metrics.")
    residual = p - y
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "n": int(len(y)),
        "rmse_mg_ha": float(np.sqrt(np.mean(residual**2))),
        "rrmse_pct": float(100.0 * np.sqrt(np.mean(residual**2)) / np.mean(y)),
        "mae_mg_ha": float(np.mean(np.abs(residual))),
        "bias_mg_ha": float(np.mean(residual)),
        "r2": float(1.0 - sse / sst) if sst > 0 else math.nan,
        "observed_mean_mg_ha": float(np.mean(y)),
        "observed_std_mg_ha": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "predicted_std_mg_ha": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
    }


def leave_one_plot_out(records: Sequence[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    accepted = [r for r in records if r["accepted_for_calibration"]]
    plots = sorted({int(r["plot_id"]) for r in accepted})
    predictions: list[dict] = []
    coefficients: list[dict] = []
    per_plot_metrics: list[dict] = []

    for held_out in plots:
        train = [r for r in accepted if int(r["plot_id"]) != held_out]
        test = [r for r in accepted if int(r["plot_id"]) == held_out]
        fit = fit_power_model(
            np.array([r["chm_mean_m"] for r in train]),
            np.array([r["agb_observed_mg_ha"] for r in train]),
            np.array([r["agb_sd_mg_ha"] for r in train]),
        )
        y_test = np.array([r["agb_observed_mg_ha"] for r in test])
        pred_test = power_model(
            np.array([r["chm_mean_m"] for r in test]), fit.a, fit.b
        )
        coefficients.append(
            {
                "held_out_plot": held_out,
                "a": fit.a,
                "b": fit.b,
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        for row, pred in zip(test, pred_test):
            predictions.append(
                {
                    "plot_id": int(row["plot_id"]),
                    "cell_id": int(row["cell_id"]),
                    "chm_mean_m": float(row["chm_mean_m"]),
                    "agb_observed_mg_ha": float(row["agb_observed_mg_ha"]),
                    "agb_predicted_lopo_mg_ha": float(pred),
                    "residual_mg_ha": float(pred - row["agb_observed_mg_ha"]),
                    "a_without_plot": fit.a,
                    "b_without_plot": fit.b,
                }
            )
        metrics = regression_metrics(y_test, pred_test)
        per_plot_metrics.append({"plot_id": held_out, **metrics})

    return predictions, coefficients, per_plot_metrics


def cluster_bootstrap_coefficients(
    records: Sequence[dict],
    iterations: int = 500,
    seed: int = 2026,
) -> np.ndarray:
    accepted = [r for r in records if r["accepted_for_calibration"]]
    plots = sorted({int(r["plot_id"]) for r in accepted})
    by_plot = {plot: [r for r in accepted if int(r["plot_id"]) == plot] for plot in plots}
    rng = np.random.default_rng(seed)
    coefficients: list[tuple[float, float]] = []

    for _ in range(iterations):
        sampled_plots = rng.choice(plots, size=len(plots), replace=True)
        sample: list[dict] = []
        for plot in sampled_plots:
            sample.extend(by_plot[int(plot)])
        try:
            fit = fit_power_model(
                np.array([r["chm_mean_m"] for r in sample]),
                np.array([r["agb_observed_mg_ha"] for r in sample]),
                np.array([r["agb_sd_mg_ha"] for r in sample]),
            )
            if np.isfinite(fit.a) and np.isfinite(fit.b):
                coefficients.append((fit.a, fit.b))
        except (CalibrationError, RuntimeError, ValueError):
            continue

    if len(coefficients) < max(100, int(iterations * 0.70)):
        raise CalibrationError(
            f"Only {len(coefficients)}/{iterations} cluster bootstrap fits succeeded."
        )
    return np.asarray(coefficients, dtype=float)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise CalibrationError(f"Cannot write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_allometric_calibration(
    records: Sequence[dict],
    fit: FitResult,
    bootstrap: np.ndarray,
    lopo_metrics: dict,
    output: Path,
) -> None:
    accepted = [r for r in records if r["accepted_for_calibration"]]
    x = np.array([r["chm_mean_m"] for r in accepted])
    y = np.array([r["agb_observed_mg_ha"] for r in accepted])
    plots = np.array([r["plot_id"] for r in accepted])
    h_grid = np.linspace(max(0.1, x.min() - 1.0), x.max() + 1.0, 300)
    curves = np.array([power_model(h_grid, a, b) for a, b in bootstrap])
    lower, upper = np.percentile(curves, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    scatter = ax.scatter(
        x,
        y,
        c=plots,
        cmap="turbo",
        s=34,
        alpha=0.68,
        edgecolors="black",
        linewidths=0.25,
        rasterized=True,
    )
    ax.fill_between(
        h_grid,
        lower,
        upper,
        color="#8B1A1A",
        alpha=0.16,
        label="95% plot-cluster bootstrap interval",
    )
    ax.plot(
        h_grid,
        power_model(h_grid, fit.a, fit.b),
        color="#8B0000",
        linewidth=3.0,
        label="Final support-matched calibration",
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("Inventory plot ID")
    ax.set_title("Support-Matched Allometric Calibration — TropiSAR Paracou")
    ax.set_xlabel("Mean LiDAR-derived CHM over the 50×50 m cell (m)")
    ax.set_ylabel("Observed aboveground biomass (Mg/ha)")
    ax.grid(True, linestyle="--", alpha=0.32)
    annotation = (
        rf"$AGB={fit.a:.3f}\,H^{{{fit.b:.4f}}}$" "\n"
        f"LOPO RMSE = {lopo_metrics['rmse_mg_ha']:.2f} Mg/ha\n"
        f"LOPO $R^2$ = {lopo_metrics['r2']:.3f}; "
        f"rRMSE = {lopo_metrics['rrmse_pct']:.2f}%"
    )
    ax.text(
        0.025,
        0.975,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    _save_figure(fig, output)


def plot_lopo_observed_predicted(
    predictions: Sequence[dict],
    metrics: dict,
    output: Path,
) -> None:
    observed = np.array([r["agb_observed_mg_ha"] for r in predictions])
    predicted = np.array([r["agb_predicted_lopo_mg_ha"] for r in predictions])
    plots = np.array([r["plot_id"] for r in predictions])
    limits = [
        float(min(observed.min(), predicted.min()) - 15.0),
        float(max(observed.max(), predicted.max()) + 15.0),
    ]

    fig, ax = plt.subplots(figsize=(8.3, 7.4))
    scatter = ax.scatter(
        observed,
        predicted,
        c=plots,
        cmap="turbo",
        s=32,
        alpha=0.68,
        edgecolors="black",
        linewidths=0.25,
        rasterized=True,
    )
    ax.plot(limits, limits, "k--", linewidth=2.0, label="Identity line (1:1)")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Plot-Held-Out AGB Validation")
    ax.set_xlabel("Observed AGB (Mg/ha)")
    ax.set_ylabel("LOPO-predicted AGB (Mg/ha)")
    ax.grid(True, linestyle="--", alpha=0.32)
    fig.colorbar(scatter, ax=ax, pad=0.015, label="Held-out plot ID")
    text = (
        f"RMSE = {metrics['rmse_mg_ha']:.2f} Mg/ha\n"
        f"rRMSE = {metrics['rrmse_pct']:.2f}%\n"
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"Bias = {metrics['bias_mg_ha']:.2f} Mg/ha"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    _save_figure(fig, output)


def plot_previous_equation_audit(
    records: Sequence[dict],
    final_fit: FitResult,
    output: Path,
) -> None:
    accepted = [r for r in records if r["accepted_for_calibration"]]
    x = np.array([r["chm_mean_m"] for r in accepted])
    y = np.array([r["agb_observed_mg_ha"] for r in accepted])
    grid = np.linspace(max(0.1, x.min() - 1), x.max() + 1, 300)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(
        x,
        y,
        s=28,
        alpha=0.42,
        color="#20B26B",
        edgecolors="black",
        linewidths=0.2,
        rasterized=True,
        label="50×50 m inventory cells",
    )
    ax.plot(
        grid,
        power_model(grid, final_fit.a, final_fit.b),
        linewidth=3,
        color="#8B0000",
        label="Current support-matched equation",
    )
    ax.plot(
        grid,
        power_model(grid, PREVIOUS_A, PREVIOUS_B),
        linewidth=2.2,
        linestyle="--",
        color="#2B2B2B",
        label="Previous equation (historical; 3×3-pixel implementation)",
    )
    ax.set_title("Audit Comparison with the Previous Allometric Equation")
    ax.set_xlabel("Mean CHM over the 50×50 m cell (m)")
    ax.set_ylabel("AGB (Mg/ha)")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    _save_figure(fig, output)


def aggregate_chm_to_regular_support(
    source_path: Path,
    support_m: float = PRIMARY_SUPPORT_M,
    target_grid: dict | None = None,
) -> tuple[np.ndarray, dict]:
    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise CalibrationError(f"Raster has no CRS: {source_path}")
        left, bottom, right, top = src.bounds
        if target_grid is None:
            aligned_left = math.floor(left / support_m) * support_m
            aligned_bottom = math.floor(bottom / support_m) * support_m
            aligned_right = math.ceil(right / support_m) * support_m
            aligned_top = math.ceil(top / support_m) * support_m
            width = int(round((aligned_right - aligned_left) / support_m))
            height = int(round((aligned_top - aligned_bottom) / support_m))
            transform = from_origin(aligned_left, aligned_top, support_m, support_m)
            crs = src.crs
        else:
            width = int(target_grid["width"])
            height = int(target_grid["height"])
            transform = rasterio.Affine(*target_grid["transform"])
            crs = rasterio.crs.CRS.from_string(target_grid["crs"])

        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    grid = {
        "width": width,
        "height": height,
        "transform": list(transform)[:6],
        "crs": str(crs),
        "support_m": support_m,
    }
    return destination, grid


def save_agb_raster(
    mean_chm: np.ndarray,
    grid: dict,
    fit: FitResult,
    output: Path,
) -> Path:
    agb = np.full(mean_chm.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(mean_chm) & (mean_chm >= MIN_MEAN_CHM_M)
    agb[valid] = power_model(mean_chm[valid], fit.a, fit.b).astype(np.float32)
    transform = rasterio.Affine(*grid["transform"])
    profile = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(agb), agb, -9999.0).astype(np.float32), 1)
        dst.update_tags(
            support="50 m mean CHM",
            allometry=f"AGB={fit.a:.10g}*H^{fit.b:.10g}",
            units="Mg/ha",
            script_version=SCRIPT_VERSION,
        )
    return output


def plot_agb_map(raster_path: Path, title: str, output: Path) -> None:
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True).filled(np.nan).astype(float)
        bounds = src.bounds
    valid = data[np.isfinite(data)]
    if not valid.size:
        raise CalibrationError(f"No finite AGB values in {raster_path}")
    vmin, vmax = np.percentile(valid, [2, 98])
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(
        data,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="AGB (Mg/ha)")
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_figure(fig, output)


def run(args: argparse.Namespace) -> dict:
    root = find_project_root(args.project_root)
    inventory_source = discover_inventory_source(root, args.inventory)
    chm_path = (
        Path(args.chm).expanduser().resolve()
        if args.chm
        else (root / CANONICAL_CHM_RELATIVE).resolve()
    )
    expected_chm = (root / CANONICAL_CHM_RELATIVE).resolve()
    if chm_path != expected_chm:
        raise CalibrationError(
            "A non-canonical CHM was requested. This revision is intentionally "
            f"locked to the TCCT pipeline CHM: {expected_chm}."
        )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("TCCT AGB — SUPPORT-MATCHED ALLOMETRIC CALIBRATION")
    print(f"Version: {SCRIPT_VERSION}")
    print(f"Project root: {root}")
    print(f"Canonical CHM: {chm_path}")
    print(f"Inventory source: {inventory_source}")
    print(f"Output: {output}")
    print("=" * 88, flush=True)

    cells, inventory_provenance = load_inventory(inventory_source)
    records, bases, chm_provenance = build_support_matched_table(cells, chm_path)
    accepted = [r for r in records if r["accepted_for_calibration"]]

    final_fit = fit_power_model(
        np.array([r["chm_mean_m"] for r in accepted]),
        np.array([r["agb_observed_mg_ha"] for r in accepted]),
        np.array([r["agb_sd_mg_ha"] for r in accepted]),
    )
    lopo_predictions, lopo_coefficients, per_plot_metrics = leave_one_plot_out(records)
    lopo_metrics = regression_metrics(
        np.array([r["agb_observed_mg_ha"] for r in lopo_predictions]),
        np.array([r["agb_predicted_lopo_mg_ha"] for r in lopo_predictions]),
    )
    bootstrap = cluster_bootstrap_coefficients(
        records,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    a_ci = np.percentile(bootstrap[:, 0], [2.5, 50, 97.5])
    b_ci = np.percentile(bootstrap[:, 1], [2.5, 50, 97.5])

    write_csv(output / "inventory_50m_support_matched.csv", records)
    write_csv(output / "lopo_agb_predictions.csv", lopo_predictions)
    write_csv(output / "lopo_coefficients.csv", lopo_coefficients)
    write_csv(output / "lopo_metrics_by_plot.csv", per_plot_metrics)
    write_csv(
        output / "plot_grid_geometry.csv",
        [
            {
                "plot_id": plot,
                "u_x": float(basis.u[0]),
                "u_y": float(basis.u[1]),
                "v_x": float(basis.v[0]),
                "v_y": float(basis.v[1]),
                "spacing_u_m": basis.length_u,
                "spacing_v_m": basis.length_v,
                "angle_deg": basis.angle_deg,
                "cell_area_m2": basis.area_m2,
            }
            for plot, basis in sorted(bases.items())
        ],
    )

    coefficient_summary = {
        "model": "AGB = a * mean_CHM_50m^b",
        "height_definition": "mean canonical LiDAR CHM over complete 50x50 m inventory cell",
        "fit_method": "uncertainty-weighted nonlinear least squares in original AGB scale",
        "a": final_fit.a,
        "b": final_fit.b,
        "a_cluster_bootstrap_95pct": [float(a_ci[0]), float(a_ci[2])],
        "b_cluster_bootstrap_95pct": [float(b_ci[0]), float(b_ci[2])],
        "bootstrap_successful_iterations": int(len(bootstrap)),
        "lopo_metrics": lopo_metrics,
        "n_inventory_cells_total": len(records),
        "n_inventory_cells_used": len(accepted),
        "support_m": PRIMARY_SUPPORT_M,
        "minimum_valid_chm_fraction": MIN_VALID_FRACTION,
        "previous_equation_audit_only": {
            "a": PREVIOUS_A,
            "b": PREVIOUS_B,
            "status": "historical; derived from the former 3x3-pixel extraction and not reused",
        },
    }
    (output / "allometry_coefficients_and_metrics.json").write_text(
        json.dumps(coefficient_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_allometric_calibration(
        records,
        final_fit,
        bootstrap,
        lopo_metrics,
        output / "allometric_calibration_support_matched.png",
    )
    plot_lopo_observed_predicted(
        lopo_predictions,
        lopo_metrics,
        output / "agb_lopo_observed_vs_predicted.png",
    )
    plot_previous_equation_audit(
        records,
        final_fit,
        output / "audit_previous_vs_support_matched_equation.png",
    )

    reference_mean_chm, grid = aggregate_chm_to_regular_support(chm_path)
    reference_agb_path = save_agb_raster(
        reference_mean_chm,
        grid,
        final_fit,
        output / "AGB_from_canonical_LiDAR_CHM_50m.tif",
    )
    plot_agb_map(
        reference_agb_path,
        "AGB Derived from the Canonical LiDAR CHM — 50 m Support",
        output / "AGB_from_canonical_LiDAR_CHM_50m.png",
    )

    predicted_outputs: dict | None = None
    if args.predicted_chm:
        predicted_chm = Path(args.predicted_chm).expanduser().resolve()
        if not predicted_chm.exists():
            raise FileNotFoundError(predicted_chm)
        predicted_mean_chm, predicted_grid = aggregate_chm_to_regular_support(
            predicted_chm,
            target_grid=grid,
        )
        predicted_agb_path = save_agb_raster(
            predicted_mean_chm,
            predicted_grid,
            final_fit,
            output / "AGB_from_TCCT_predicted_CHM_50m.tif",
        )
        plot_agb_map(
            predicted_agb_path,
            "AGB Derived from TCCT-Predicted CHM — 50 m Support",
            output / "AGB_from_TCCT_predicted_CHM_50m.png",
        )
        difference = np.full(reference_mean_chm.shape, np.nan, dtype=np.float32)
        valid = np.isfinite(reference_mean_chm) & np.isfinite(predicted_mean_chm)
        difference[valid] = (
            power_model(predicted_mean_chm[valid], final_fit.a, final_fit.b)
            - power_model(reference_mean_chm[valid], final_fit.a, final_fit.b)
        ).astype(np.float32)
        diff_profile = {
            "driver": "GTiff",
            "height": grid["height"],
            "width": grid["width"],
            "count": 1,
            "dtype": "float32",
            "crs": grid["crs"],
            "transform": rasterio.Affine(*grid["transform"]),
            "nodata": -9999.0,
            "compress": "deflate",
            "predictor": 3,
        }
        diff_path = output / "AGB_TCCT_minus_LiDAR_50m.tif"
        with rasterio.open(diff_path, "w", **diff_profile) as dst:
            dst.write(np.where(np.isfinite(difference), difference, -9999.0), 1)
        predicted_outputs = {
            "predicted_chm": str(predicted_chm),
            "predicted_chm_sha256": sha256_file(predicted_chm),
            "predicted_agb_50m": str(predicted_agb_path),
            "agb_difference_50m": str(diff_path),
        }

    provenance = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "project_root": str(root),
        "inventory": inventory_provenance,
        "canonical_chm": chm_provenance,
        "scientific_constraints": {
            "cell_support": "complete inferred 50x50 m polygon",
            "old_3x3_pixel_sampling_used": False,
            "cell_coordinates_interpreted_as": "cell centres",
            "equation_applied_to_1m_pixels": False,
            "map_support": "regular 50 m mean-CHM grid",
            "validation": "leave-one-inventory-plot-out",
            "test_plot_used_for_calibration": False,
        },
        "results": coefficient_summary,
        "reference_agb_50m": str(reference_agb_path),
        "predicted_outputs": predicted_outputs,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFinal support-matched equation:")
    print(f"AGB = {final_fit.a:.8f} * H^{final_fit.b:.8f}")
    print(
        f"LOPO: RMSE={lopo_metrics['rmse_mg_ha']:.3f} Mg/ha | "
        f"rRMSE={lopo_metrics['rrmse_pct']:.3f}% | "
        f"R²={lopo_metrics['r2']:.4f} | "
        f"bias={lopo_metrics['bias_mg_ha']:.3f} Mg/ha"
    )
    print(f"Accepted cells: {len(accepted)}/{len(records)}")
    print(f"Results saved in: {output}")
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Support-matched AGB allometry for TropiSAR–Paracou."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--inventory",
        default=None,
        help="Path to CSV_50.zip or extracted CSV_50 folder.",
    )
    parser.add_argument(
        "--chm",
        default=None,
        help=(
            "Canonical CHM path. A non-canonical path is intentionally rejected "
            "to prevent reuse of the former Gemini CHM."
        ),
    )
    parser.add_argument("--predicted-chm", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print("\n" + "!" * 88, file=sys.stderr)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("!" * 88, file=sys.stderr)
        raise
