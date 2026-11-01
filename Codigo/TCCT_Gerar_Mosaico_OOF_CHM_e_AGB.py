#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a strict Phase-A out-of-fold (OOF) TCCT CHM mosaic and derive a
support-matched AGB map.

Scientific protocol
-------------------
1. Rebuild the same five spatial folds used in the audited TCCT experiment.
2. For each fold, load the Phase-A checkpoint for seeds 42, 314 and 2026.
3. Predict only the validation samples of that fold (never training samples).
4. Average pixel-level predictions across the three seeds and merge all folds
   on the canonical 1 m LiDAR grid.
5. Aggregate the OOF CHM to the exact 50 m grid used by the support-matched
   allometric calibration.
6. Apply the already-calibrated equation stored in
   allometry_coefficients_and_metrics.json.
7. Evaluate the full chain against field AGB with two independent safeguards:
   spatial OOF CHM predictions and plot-level leave-one-plot-out allometry.

No training is performed. The script only loads frozen checkpoints and runs
inference. Per-seed/fold pixel predictions are cached, allowing safe resume.

Usage (Windows)
---------------
<LOCAL_PATH> ^
  <LOCAL_PATH> ^
  --project-root <LOCAL_PATH>

Outputs
-------
Resultados/TCCT_v3_2_SIBGRAPI/OOF_CHM_AGB_SupportMatched/
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


SCRIPT_VERSION = "1.0.0-oof-phaseA-support-matched-agb"
PIPELINE_FILENAME = (
    "Pipeline_TCCT_v3_2_1_SIBGRAPI_REV_GRILLE3D_RETOMAVEL.py"
)
ALLOMETRY_FILENAME = "TCCT_AGB_Allometry_SupportMatched.py"
DEFAULT_OUTPUT_RELATIVE = Path(
    "Resultados/TCCT_v3_2_SIBGRAPI/OOF_CHM_AGB_SupportMatched"
)
ALLOMETRY_OUTPUT_RELATIVE = Path(
    "Resultados/TCCT_v3_2_SIBGRAPI/AGB_Allometry_SupportMatched"
)
SEEDS_EXPECTED = (42, 314, 2026)
N_FOLDS_EXPECTED = 5
PHYSICAL_CHM_MIN_M = 0.0
PHYSICAL_CHM_MAX_M = 60.0
NODATA = -9999.0


class OOFMappingError(RuntimeError):
    """Raised when an audit condition prevents a defensible OOF map."""


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
    candidates.extend([Path.cwd(), Path.cwd().parent])
    for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        candidates.append(Path(f"{drive}:/SIBGRAPI_REV"))
    candidates.append(Path("<PROJECT_ROOT>"))

    checked: list[str] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        checked.append(str(resolved))
        if (
            resolved.name == "SIBGRAPI_REV"
            and (resolved / "Codigo").is_dir()
            and (resolved / "Dados_Processados").is_dir()
        ):
            return resolved
        nested = resolved / "SIBGRAPI_REV"
        if (
            nested.is_dir()
            and (nested / "Codigo").is_dir()
            and (nested / "Dados_Processados").is_dir()
        ):
            return nested.resolve()

    raise FileNotFoundError(
        "SIBGRAPI_REV was not found. Pass --project-root explicitly. "
        f"Checked: {checked[:12]}"
    )


def import_module_from_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def choose_device(requested: str, pipeline) -> torch.device:
    requested = requested.lower().strip()
    if requested == "auto":
        return pipeline.DEVICE
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise OOFMappingError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise OOFMappingError("MPS was requested but is not available.")
        return torch.device("mps")
    raise ValueError(f"Unsupported device: {requested}")


def robust_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "sim"})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def save_float_raster(
    output: Path,
    data: np.ndarray,
    reference_profile: dict,
    tags: dict[str, str] | None = None,
) -> Path:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(
            np.where(np.isfinite(data), data, NODATA).astype(np.float32),
            1,
        )
        if tags:
            dst.update_tags(**tags)
    return output


def save_uint16_raster(
    output: Path,
    data: np.ndarray,
    reference_profile: dict,
    tags: dict[str, str] | None = None,
) -> Path:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        height=int(data.shape[0]),
        width=int(data.shape[1]),
        count=1,
        dtype="uint16",
        nodata=0,
        compress="deflate",
        predictor=2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(data.astype(np.uint16), 1)
        if tags:
            dst.update_tags(**tags)
    return output


def plot_raster(
    raster_path: Path,
    title: str,
    colorbar_label: str,
    output: Path,
    cmap: str = "viridis",
    symmetric: bool = False,
) -> Path:
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True).filled(np.nan).astype(float)
        bounds = src.bounds
    valid = data[np.isfinite(data)]
    if not valid.size:
        raise OOFMappingError(f"No finite values in {raster_path}")
    if symmetric:
        limit = float(np.percentile(np.abs(valid), 98.0))
        vmin, vmax = -limit, limit
    else:
        vmin, vmax = map(float, np.percentile(valid, [2.0, 98.0]))
        if math.isclose(vmin, vmax):
            vmin, vmax = float(valid.min()), float(valid.max())

    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    image = ax.imshow(
        data,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        raise OOFMappingError("No finite values were available for metrics.")
    residual = p - y
    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "n": int(len(y)),
        "rmse": rmse,
        "rrmse_pct": float(100.0 * rmse / np.mean(y)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "r2": float(1.0 - sse / sst) if sst > 0 else math.nan,
        "observed_mean": float(np.mean(y)),
        "observed_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "predicted_mean": float(np.mean(p)),
        "predicted_std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
        "std_ratio": (
            float(np.std(p, ddof=1) / np.std(y, ddof=1))
            if len(y) > 1 and np.std(y, ddof=1) > 0
            else math.nan
        ),
    }


def load_checkpoint(path: Path, pipeline, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(path)
    if hasattr(pipeline, "_torch_load_local"):
        payload = pipeline._torch_load_local(path, map_location="cpu")
    else:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    stats = pipeline.StatsTreino(**payload["stats"])
    model = pipeline.criar_modelo("tcct", stats).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, stats, payload


def checkpoint_path(
    pipeline,
    seed: int,
    fold: int,
) -> Path:
    execution_dir = pipeline._diretorio_execucao_multiseed(
        "tcct", "completo", int(seed), int(fold)
    )
    return execution_dir / "fase_A_supervisionada" / "melhor_checkpoint.pth"


def infer_seed_fold_pixels(
    pipeline,
    dataset,
    validation_indices: np.ndarray,
    seed: int,
    fold: int,
    device: torch.device,
    cache_path: Path,
    overwrite_cache: bool,
) -> dict:
    indices = np.sort(np.asarray(validation_indices, dtype=np.int64))
    expected_checkpoint = checkpoint_path(pipeline, seed, fold)

    if cache_path.exists() and not overwrite_cache:
        cached = np.load(cache_path, allow_pickle=False)
        cached_indices = cached["dataset_index"].astype(np.int64)
        predictions = cached["prediction_30x30"].astype(np.float32)
        windows = cached["chm_window"].astype(np.int32)
        if not np.array_equal(cached_indices, indices):
            raise OOFMappingError(
                f"Cached indices do not match fold={fold}, seed={seed}: {cache_path}"
            )
        if predictions.shape != (len(indices), 30, 30):
            raise OOFMappingError(
                f"Unexpected cached prediction shape {predictions.shape}: {cache_path}"
            )
        return {
            "dataset_index": cached_indices,
            "prediction_30x30": predictions,
            "chm_window": windows,
            "checkpoint": expected_checkpoint,
            "reused_cache": True,
        }

    model, stats, _ = load_checkpoint(expected_checkpoint, pipeline, device)
    loader = DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=int(pipeline.CFG.treino.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    predictions: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"OOF pixel inference — seed {seed}, fold {fold}",
            leave=False,
        ):
            optical, sar_l, sar_p, quality, _, _, _ = pipeline.preparar_batch_v2(
                batch,
                stats,
                device,
                ablation="completo",
            )
            pred_m, _, _ = model(sar_l, sar_p, optical, quality)
            pred_np = pred_m.detach().cpu().numpy().astype(np.float32)
            if pred_np.ndim != 4 or pred_np.shape[1:] != (1, 30, 30):
                raise OOFMappingError(
                    f"Unexpected model output shape: {pred_np.shape}"
                )
            predictions.append(pred_np[:, 0])
            windows.append(batch["chm_window"].detach().cpu().numpy().astype(np.int32))

    prediction_array = np.concatenate(predictions, axis=0)
    window_array = np.concatenate(windows, axis=0)
    if len(prediction_array) != len(indices):
        raise OOFMappingError(
            f"Prediction count mismatch for seed={seed}, fold={fold}: "
            f"{len(prediction_array)} vs {len(indices)}"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        dataset_index=indices,
        prediction_30x30=prediction_array,
        chm_window=window_array,
    )
    metadata = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "seed": int(seed),
        "fold": int(fold),
        "n_validation": int(len(indices)),
        "checkpoint": str(expected_checkpoint),
        "checkpoint_sha256": sha256_file(expected_checkpoint),
        "cache": str(cache_path),
        "prediction_shape": list(prediction_array.shape),
    }
    write_json(cache_path.with_suffix(".json"), metadata)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset_index": indices,
        "prediction_30x30": prediction_array,
        "chm_window": window_array,
        "checkpoint": expected_checkpoint,
        "reused_cache": False,
    }


def audit_folds(dataset, folds: list[tuple[np.ndarray, np.ndarray]]) -> None:
    validation_sets = [np.asarray(v, dtype=int) for _, v in folds]
    concatenated = np.concatenate(validation_sets)
    if len(np.unique(concatenated)) != len(concatenated):
        raise OOFMappingError("Validation folds contain duplicated dataset indices.")
    if not np.array_equal(np.sort(concatenated), np.arange(len(dataset))):
        missing = sorted(set(range(len(dataset))) - set(map(int, concatenated)))
        extra = sorted(set(map(int, concatenated)) - set(range(len(dataset))))
        raise OOFMappingError(
            f"Validation folds do not cover the dataset exactly. "
            f"Missing={missing[:10]}, extra={extra[:10]}"
        )


def reconstruct_oof_mosaic(
    execution_caches: Iterable[dict],
    height: int,
    width: int,
    reference_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sum_map = np.zeros((height, width), dtype=np.float64)
    count_map = np.zeros((height, width), dtype=np.uint16)

    for execution in tqdm(
        list(execution_caches),
        desc="Reconstructing strict Phase-A OOF CHM mosaic",
    ):
        predictions = execution["prediction_30x30"]
        windows = execution["chm_window"]
        for prediction, window in zip(predictions, windows):
            r0, c0, h, w = map(int, window)
            if r0 < 0 or c0 < 0 or h <= 0 or w <= 0:
                raise OOFMappingError(f"Invalid CHM window: {window.tolist()}")
            if r0 + h > height or c0 + w > width:
                raise OOFMappingError(
                    f"CHM window exceeds reference grid: {window.tolist()}"
                )
            if (h, w) == (30, 30):
                resized = prediction.astype(np.float32, copy=False)
            else:
                tensor = torch.from_numpy(prediction)[None, None]
                resized = (
                    F.interpolate(
                        tensor,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    .numpy()
                    .astype(np.float32)
                )
            valid_prediction = np.isfinite(resized)
            target_sum = sum_map[r0 : r0 + h, c0 : c0 + w]
            target_count = count_map[r0 : r0 + h, c0 : c0 + w]
            target_sum[valid_prediction] += resized[valid_prediction]
            target_count[valid_prediction] += 1

    raw = np.divide(
        sum_map,
        count_map,
        out=np.full((height, width), np.nan, dtype=np.float64),
        where=count_map > 0,
    ).astype(np.float32)
    raw[~reference_valid_mask] = np.nan
    physical = raw.copy()
    finite = np.isfinite(physical)
    physical[finite] = np.clip(
        physical[finite],
        PHYSICAL_CHM_MIN_M,
        PHYSICAL_CHM_MAX_M,
    )
    return raw, physical, count_map


def save_regular_grid_raster(
    path: Path,
    data: np.ndarray,
    grid: dict,
    tags: dict[str, str],
) -> Path:
    transform = rasterio.Affine(*grid["transform"])
    profile = {
        "driver": "GTiff",
        "height": int(grid["height"]),
        "width": int(grid["width"]),
        "count": 1,
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(
            np.where(np.isfinite(data), data, NODATA).astype(np.float32),
            1,
        )
        dst.update_tags(**tags)
    return path


def build_grid_basis(allometry, row: pd.Series):
    u = np.array([float(row["grid_u_x"]), float(row["grid_u_y"])])
    v = np.array([float(row["grid_v_x"]), float(row["grid_v_y"])])
    length_u = float(np.linalg.norm(u))
    length_v = float(np.linalg.norm(v))
    angle = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(u, v) / (length_u * length_v),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    area = float(abs(u[0] * v[1] - u[1] * v[0]))
    return allometry.GridBasis(
        u=u,
        v=v,
        length_u=length_u,
        length_v=length_v,
        angle_deg=angle,
        area_m2=area,
    )


def evaluate_end_to_end_field_agb(
    predicted_chm_path: Path,
    allometry_output: Path,
    allometry,
    output: Path,
) -> tuple[pd.DataFrame, dict]:
    inventory_path = allometry_output / "inventory_50m_support_matched.csv"
    coefficients_path = allometry_output / "lopo_coefficients.csv"
    if not inventory_path.exists():
        raise FileNotFoundError(inventory_path)
    if not coefficients_path.exists():
        raise FileNotFoundError(coefficients_path)

    inventory = pd.read_csv(inventory_path)
    coefficients = pd.read_csv(coefficients_path)
    required_inventory = {
        "plot_id",
        "cell_id",
        "x_center",
        "y_center",
        "agb_observed_mg_ha",
        "grid_u_x",
        "grid_u_y",
        "grid_v_x",
        "grid_v_y",
        "accepted_for_calibration",
    }
    missing = required_inventory - set(inventory.columns)
    if missing:
        raise KeyError(f"Missing inventory columns: {sorted(missing)}")
    required_coefficients = {"held_out_plot", "a", "b"}
    missing = required_coefficients - set(coefficients.columns)
    if missing:
        raise KeyError(f"Missing LOPO coefficient columns: {sorted(missing)}")

    inventory = inventory.loc[robust_bool(inventory["accepted_for_calibration"])].copy()
    coefficient_lookup = {
        int(row["held_out_plot"]): (float(row["a"]), float(row["b"]))
        for _, row in coefficients.iterrows()
    }

    records: list[dict] = []
    with rasterio.open(predicted_chm_path) as predicted_chm:
        for _, row in tqdm(
            inventory.iterrows(),
            total=len(inventory),
            desc="End-to-end field AGB validation",
            leave=False,
        ):
            plot_id = int(row["plot_id"])
            if plot_id not in coefficient_lookup:
                raise OOFMappingError(
                    f"No LOPO allometric coefficients for plot {plot_id}"
                )
            basis = build_grid_basis(allometry, row)
            polygon = allometry.cell_polygon(
                float(row["x_center"]),
                float(row["y_center"]),
                basis,
            )
            try:
                stats = allometry.extract_polygon_chm(predicted_chm, polygon)
            except Exception as exc:
                stats = {
                    "valid_fraction": 0.0,
                    "chm_mean_m": math.nan,
                    "chm_std_m": math.nan,
                }
                error_message = f"{type(exc).__name__}: {exc}"
            else:
                error_message = ""

            accepted = (
                float(stats.get("valid_fraction", 0.0))
                >= float(allometry.MIN_VALID_FRACTION)
                and np.isfinite(float(stats.get("chm_mean_m", math.nan)))
                and float(stats["chm_mean_m"]) >= float(allometry.MIN_MEAN_CHM_M)
            )
            a, b = coefficient_lookup[plot_id]
            predicted_agb = (
                float(allometry.power_model(float(stats["chm_mean_m"]), a, b))
                if accepted
                else math.nan
            )
            records.append(
                {
                    "plot_id": plot_id,
                    "cell_id": int(row["cell_id"]),
                    "x_center": float(row["x_center"]),
                    "y_center": float(row["y_center"]),
                    "agb_observed_mg_ha": float(row["agb_observed_mg_ha"]),
                    "tcct_oof_chm_mean_50m_m": float(
                        stats.get("chm_mean_m", math.nan)
                    ),
                    "tcct_oof_chm_valid_fraction": float(
                        stats.get("valid_fraction", 0.0)
                    ),
                    "allometry_a_without_plot": a,
                    "allometry_b_without_plot": b,
                    "agb_predicted_end_to_end_mg_ha": predicted_agb,
                    "accepted_end_to_end": bool(accepted),
                    "extraction_error": error_message,
                }
            )

    table = pd.DataFrame(records)
    accepted = table.loc[table["accepted_end_to_end"]].copy()
    if len(accepted) < 20 or accepted["plot_id"].nunique() < 5:
        raise OOFMappingError(
            "Too few field cells/plots have sufficient OOF CHM coverage for an "
            f"end-to-end AGB evaluation: cells={len(accepted)}, "
            f"plots={accepted['plot_id'].nunique()}"
        )

    metrics = regression_metrics(
        accepted["agb_observed_mg_ha"].to_numpy(float),
        accepted["agb_predicted_end_to_end_mg_ha"].to_numpy(float),
    )
    metrics.update(
        {
            "metric_scope": (
                "field AGB validation using spatial OOF TCCT CHM and "
                "plot-LOPO allometry"
            ),
            "n_inventory_cells_available": int(len(table)),
            "n_inventory_cells_evaluated": int(len(accepted)),
            "n_inventory_plots_evaluated": int(accepted["plot_id"].nunique()),
            "coverage_fraction_cells": float(len(accepted) / len(table)),
            "independence_note": (
                "Each CHM location is predicted by a model that excluded its "
                "spatial fold; each AGB cell uses allometric coefficients fitted "
                "without its entire inventory plot."
            ),
        }
    )
    table.to_csv(output / "end_to_end_agb_predictions_plot_lopo.csv", index=False)
    write_json(output / "end_to_end_agb_metrics_plot_lopo.json", metrics)

    observed = accepted["agb_observed_mg_ha"].to_numpy(float)
    predicted = accepted["agb_predicted_end_to_end_mg_ha"].to_numpy(float)
    plots = accepted["plot_id"].to_numpy(int)
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    margin = 0.04 * (high - low if high > low else 1.0)
    limits = (low - margin, high + margin)

    fig, ax = plt.subplots(figsize=(8.4, 7.5))
    scatter = ax.scatter(
        observed,
        predicted,
        c=plots,
        cmap="turbo",
        s=34,
        alpha=0.70,
        edgecolors="black",
        linewidths=0.25,
        rasterized=True,
    )
    ax.plot(limits, limits, "k--", linewidth=1.8, label="Identity line (1:1)")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("End-to-End AGB Validation — Spatial OOF + Plot-LOPO")
    ax.set_xlabel("Observed field AGB (Mg/ha)")
    ax.set_ylabel("Predicted AGB from TCCT OOF CHM (Mg/ha)")
    ax.grid(True, linestyle="--", alpha=0.32)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    cbar.set_label("Inventory plot ID")
    annotation = (
        f"RMSE = {metrics['rmse']:.2f} Mg/ha\n"
        f"rRMSE = {metrics['rrmse_pct']:.2f}%\n"
        f"$R^2$ = {metrics['r2']:.3f}\n"
        f"Bias = {metrics['bias']:.2f} Mg/ha\n"
        f"n = {metrics['n']} cells"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(
        output / "end_to_end_agb_observed_vs_predicted_plot_lopo.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return table, metrics


def run(args: argparse.Namespace) -> dict:
    root = find_project_root(args.project_root)
    os.environ["TCCT_SIBGRAPI_REV_DIR"] = str(root)
    pipeline_path = root / "Codigo" / PIPELINE_FILENAME
    allometry_path = root / "Codigo" / ALLOMETRY_FILENAME
    pipeline = import_module_from_path("tcct_pipeline_oof_mapping", pipeline_path)
    allometry = import_module_from_path("tcct_allometry_support_matched", allometry_path)
    device = choose_device(args.device, pipeline)

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (root / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache_pixel_predictions_phaseA"
    cache_dir.mkdir(parents=True, exist_ok=True)

    allometry_output = (root / ALLOMETRY_OUTPUT_RELATIVE).resolve()
    coefficient_path = allometry_output / "allometry_coefficients_and_metrics.json"
    if not coefficient_path.exists():
        raise FileNotFoundError(
            f"The support-matched allometric calibration must be run first: "
            f"{coefficient_path}"
        )
    coefficient_summary = json.loads(coefficient_path.read_text(encoding="utf-8"))
    a = float(coefficient_summary["a"])
    b = float(coefficient_summary["b"])

    print("=" * 96)
    print("TCCT — STRICT PHASE-A OOF CHM AND SUPPORT-MATCHED AGB")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Project root: {root}")
    print(f"Pipeline: {pipeline_path}")
    print(f"Pipeline version: {pipeline.VERSAO_PIPELINE}")
    print(f"Device: {device}")
    print(f"Allometry: AGB = {a:.10g} * mean_CHM_50m ^ {b:.10g}")
    print(f"Output: {output}")
    print("=" * 96, flush=True)

    dataset = pipeline.TropiSARDatasetV2(pipeline.CFG.dados.dir_dataset_v2)
    folds = pipeline.criar_folds_espaciais(dataset)
    audit_folds(dataset, folds)
    if len(folds) != N_FOLDS_EXPECTED:
        raise OOFMappingError(
            f"Expected {N_FOLDS_EXPECTED} folds, found {len(folds)}"
        )
    seeds = tuple(map(int, pipeline.CFG.treino.seeds_experimento))
    if seeds != SEEDS_EXPECTED:
        raise OOFMappingError(
            f"Expected seeds {SEEDS_EXPECTED}, found {seeds}"
        )

    reference_chm_path = pipeline.encontrar_chm(pipeline.CFG.dados)
    with rasterio.open(reference_chm_path) as reference:
        reference_chm = (
            reference.read(1, masked=True).filled(np.nan).astype(np.float32)
        )
        reference_profile = reference.profile.copy()
        height, width = reference.height, reference.width
        if reference.crs is None:
            raise OOFMappingError("The canonical CHM has no CRS.")
        if reference.transform.is_identity:
            raise OOFMappingError("The canonical CHM has an identity transform.")
    reference_valid = (
        np.isfinite(reference_chm)
        & (reference_chm >= PHYSICAL_CHM_MIN_M)
        & (reference_chm <= PHYSICAL_CHM_MAX_M)
    )

    execution_results: list[dict] = []
    execution_manifest: list[dict] = []
    for fold, (_, validation_indices) in enumerate(folds):
        for seed in seeds:
            cache_path = cache_dir / f"tcct_complete_phaseA_seed_{seed}_fold_{fold}.npz"
            result = infer_seed_fold_pixels(
                pipeline,
                dataset,
                validation_indices,
                seed,
                fold,
                device,
                cache_path,
                overwrite_cache=bool(args.overwrite_cache),
            )
            execution_results.append(result)
            execution_manifest.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "n_validation": int(len(validation_indices)),
                    "checkpoint": str(result["checkpoint"]),
                    "checkpoint_sha256": sha256_file(result["checkpoint"]),
                    "cache": str(cache_path),
                    "reused_cache": bool(result["reused_cache"]),
                }
            )

    raw_chm, physical_chm, coverage_count = reconstruct_oof_mosaic(
        execution_results,
        height,
        width,
        reference_valid,
    )
    finite_raw = np.isfinite(raw_chm)
    if not finite_raw.any():
        raise OOFMappingError("The reconstructed OOF CHM mosaic is empty.")

    valid_reference_count = int(reference_valid.sum())
    covered_reference_count = int((finite_raw & reference_valid).sum())
    coverage_fraction = float(covered_reference_count / valid_reference_count)
    out_of_range_low = int((finite_raw & (raw_chm < PHYSICAL_CHM_MIN_M)).sum())
    out_of_range_high = int((finite_raw & (raw_chm > PHYSICAL_CHM_MAX_M)).sum())

    raw_path = save_float_raster(
        output / "CHM_TCCT_OOF_PhaseA_seedmean_raw_1m.tif",
        raw_chm,
        reference_profile,
        tags={
            "product": "strict Phase-A spatial OOF TCCT CHM",
            "aggregation": "mean of three seeds within five validation folds",
            "units": "m",
            "clipped": "false",
            "script_version": SCRIPT_VERSION,
        },
    )
    physical_path = save_float_raster(
        output / "CHM_TCCT_OOF_PhaseA_seedmean_physical_1m.tif",
        physical_chm,
        reference_profile,
        tags={
            "product": "strict Phase-A spatial OOF TCCT CHM",
            "aggregation": "mean of three seeds within five validation folds",
            "units": "m",
            "physical_range_m": f"{PHYSICAL_CHM_MIN_M},{PHYSICAL_CHM_MAX_M}",
            "script_version": SCRIPT_VERSION,
        },
    )
    coverage_path = save_uint16_raster(
        output / "CHM_TCCT_OOF_prediction_count_1m.tif",
        coverage_count,
        reference_profile,
        tags={
            "description": "number of seed-fold patch predictions contributing to each pixel",
            "script_version": SCRIPT_VERSION,
        },
    )

    plot_raster(
        physical_path,
        "Strict Phase-A Out-of-Fold TCCT Canopy Height Mosaic",
        "Canopy height (m)",
        output / "CHM_TCCT_OOF_PhaseA_seedmean_physical_1m.png",
        cmap="viridis",
    )

    pixel_metrics = regression_metrics(
        reference_chm[reference_valid & np.isfinite(physical_chm)],
        physical_chm[reference_valid & np.isfinite(physical_chm)],
    )
    pixel_metrics.update(
        {
            "scope": "covered valid pixels on the canonical LiDAR grid",
            "reference": str(reference_chm_path),
            "prediction": str(physical_path),
        }
    )
    write_json(output / "chm_oof_pixel_metrics.json", pixel_metrics)

    reference_mean_50m, grid = allometry.aggregate_chm_to_regular_support(
        reference_chm_path
    )
    predicted_mean_50m, predicted_grid = allometry.aggregate_chm_to_regular_support(
        physical_path,
        target_grid=grid,
    )
    if predicted_grid != grid:
        raise OOFMappingError("Predicted and reference 50 m grids differ.")

    predicted_chm_50m_path = save_regular_grid_raster(
        output / "CHM_TCCT_OOF_PhaseA_seedmean_50m.tif",
        predicted_mean_50m,
        grid,
        tags={
            "support": "50 m mean CHM",
            "source": str(physical_path),
            "units": "m",
            "script_version": SCRIPT_VERSION,
        },
    )
    plot_raster(
        predicted_chm_50m_path,
        "Out-of-Fold TCCT Mean Canopy Height — 50 m Support",
        "Mean canopy height (m)",
        output / "CHM_TCCT_OOF_PhaseA_seedmean_50m.png",
        cmap="viridis",
    )

    fit = allometry.FitResult(
        a=a,
        b=b,
        covariance=np.full((2, 2), np.nan, dtype=float),
    )
    predicted_agb_path = allometry.save_agb_raster(
        predicted_mean_50m,
        grid,
        fit,
        output / "AGB_from_TCCT_OOF_predicted_CHM_50m.tif",
    )
    allometry.plot_agb_map(
        predicted_agb_path,
        "AGB Derived from Strict Phase-A OOF TCCT CHM — 50 m Support",
        output / "AGB_from_TCCT_OOF_predicted_CHM_50m.png",
    )

    reference_agb = np.full(reference_mean_50m.shape, np.nan, dtype=np.float32)
    valid_reference_50m = (
        np.isfinite(reference_mean_50m)
        & (reference_mean_50m >= float(allometry.MIN_MEAN_CHM_M))
    )
    reference_agb[valid_reference_50m] = allometry.power_model(
        reference_mean_50m[valid_reference_50m], a, b
    ).astype(np.float32)
    predicted_agb = np.full(predicted_mean_50m.shape, np.nan, dtype=np.float32)
    valid_predicted_50m = (
        np.isfinite(predicted_mean_50m)
        & (predicted_mean_50m >= float(allometry.MIN_MEAN_CHM_M))
    )
    predicted_agb[valid_predicted_50m] = allometry.power_model(
        predicted_mean_50m[valid_predicted_50m], a, b
    ).astype(np.float32)

    difference_agb = predicted_agb - reference_agb
    difference_agb_path = save_regular_grid_raster(
        output / "AGB_TCCT_OOF_minus_LiDAR_derived_50m.tif",
        difference_agb,
        grid,
        tags={
            "difference": "TCCT OOF-derived AGB minus LiDAR-derived AGB",
            "units": "Mg/ha",
            "support": "50 m",
            "script_version": SCRIPT_VERSION,
        },
    )
    plot_raster(
        difference_agb_path,
        "AGB Difference: TCCT OOF-Derived Minus LiDAR-Derived — 50 m",
        "AGB difference (Mg/ha)",
        output / "AGB_TCCT_OOF_minus_LiDAR_derived_50m.png",
        cmap="coolwarm",
        symmetric=True,
    )

    valid_50m = np.isfinite(reference_mean_50m) & np.isfinite(predicted_mean_50m)
    chm_50m_metrics = regression_metrics(
        reference_mean_50m[valid_50m],
        predicted_mean_50m[valid_50m],
    )
    chm_50m_metrics["scope"] = "LiDAR-derived versus TCCT OOF mean CHM at 50 m"
    write_json(output / "chm_oof_50m_consistency_metrics.json", chm_50m_metrics)

    valid_agb = np.isfinite(reference_agb) & np.isfinite(predicted_agb)
    agb_consistency_metrics = regression_metrics(
        reference_agb[valid_agb],
        predicted_agb[valid_agb],
    )
    agb_consistency_metrics.update(
        {
            "scope": (
                "consistency between AGB derived from LiDAR CHM and from TCCT "
                "OOF CHM using the same final allometric equation"
            ),
            "warning": (
                "This is not independent field-AGB validation because both maps "
                "use the same allometric equation. Use the plot-LOPO end-to-end "
                "metrics for field validation."
            ),
        }
    )
    write_json(
        output / "agb_50m_derived_map_consistency_metrics.json",
        agb_consistency_metrics,
    )

    _, end_to_end_metrics = evaluate_end_to_end_field_agb(
        physical_path,
        allometry_output,
        allometry,
        output,
    )

    pd.DataFrame(execution_manifest).to_csv(
        output / "checkpoint_manifest_phaseA_oof.csv",
        index=False,
    )
    run_summary = {
        "script_version": SCRIPT_VERSION,
        "created_utc": utc_now_iso(),
        "project_root": str(root),
        "pipeline_path": str(pipeline_path),
        "pipeline_version": str(pipeline.VERSAO_PIPELINE),
        "pipeline_sha256": sha256_file(pipeline_path),
        "allometry_script_path": str(allometry_path),
        "allometry_script_sha256": sha256_file(allometry_path),
        "allometry_coefficients_path": str(coefficient_path),
        "allometry_coefficients_sha256": sha256_file(coefficient_path),
        "allometry_a": a,
        "allometry_b": b,
        "device": str(device),
        "seeds": list(seeds),
        "folds": len(folds),
        "dataset_size": len(dataset),
        "reference_chm": str(reference_chm_path),
        "reference_chm_sha256": sha256_file(reference_chm_path),
        "oof_coverage_fraction_of_valid_reference": coverage_fraction,
        "covered_valid_reference_pixels": covered_reference_count,
        "valid_reference_pixels": valid_reference_count,
        "raw_predictions_below_zero_pixels": out_of_range_low,
        "raw_predictions_above_60m_pixels": out_of_range_high,
        "outputs": {
            "raw_oof_chm_1m": str(raw_path),
            "physical_oof_chm_1m": str(physical_path),
            "coverage_count_1m": str(coverage_path),
            "mean_oof_chm_50m": str(predicted_chm_50m_path),
            "agb_from_oof_chm_50m": str(predicted_agb_path),
            "agb_difference_50m": str(difference_agb_path),
        },
        "metrics": {
            "chm_pixel": pixel_metrics,
            "chm_50m": chm_50m_metrics,
            "agb_derived_map_consistency_50m": agb_consistency_metrics,
            "agb_field_end_to_end_plot_lopo": end_to_end_metrics,
        },
        "interpretation_constraints": [
            "The CHM product is a strict out-of-fold ensemble mosaic, not a single deployment model.",
            "The 50 m AGB map uses the final support-matched allometric equation.",
            "Map-to-map AGB consistency is not independent field validation.",
            "Field AGB performance is reported only from spatial OOF CHM plus plot-LOPO allometry.",
        ],
    }
    write_json(output / "run_summary.json", run_summary)

    print("\n" + "=" * 96)
    print("COMPLETED — STRICT PHASE-A OOF CHM AND SUPPORT-MATCHED AGB")
    print(f"OOF coverage of valid reference CHM: {coverage_fraction:.2%}")
    print(
        "CHM 50 m: "
        f"RMSE={chm_50m_metrics['rmse']:.3f} m, "
        f"R²={chm_50m_metrics['r2']:.3f}, "
        f"bias={chm_50m_metrics['bias']:.3f} m"
    )
    print(
        "End-to-end field AGB (spatial OOF + plot-LOPO): "
        f"RMSE={end_to_end_metrics['rmse']:.3f} Mg/ha, "
        f"rRMSE={end_to_end_metrics['rrmse_pct']:.2f}%, "
        f"R²={end_to_end_metrics['r2']:.3f}, "
        f"bias={end_to_end_metrics['bias']:.3f} Mg/ha"
    )
    print(f"Results: {output}")
    print("=" * 96, flush=True)
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a strict Phase-A spatial OOF TCCT CHM mosaic and a "
            "support-matched 50 m AGB map without retraining."
        )
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute per-seed/fold pixel predictions even if valid caches exist.",
    )
    return parser


if __name__ == "__main__":
    try:
        run(build_parser().parse_args())
    except Exception as exc:
        print("\n" + "!" * 96, file=sys.stderr)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("!" * 96, file=sys.stderr)
        raise
