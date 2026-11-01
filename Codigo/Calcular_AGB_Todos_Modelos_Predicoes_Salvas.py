#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Calcula métricas de AGB para todos os modelos/ablações usando somente as
predições OOF salvas durante os treinamentos e a calibração alométrica
50 x 50 m com validação leave-one-plot-out (LOPO).

PROTOCOLO
---------
1. Lê os vetores y_true/y_pred gravados no momento de cada treinamento.
2. Associa cada vetor aos índices exatos dos manifestos espaciais.
3. Reconstrói um mosaico OOF de médias de CHM em células de 30 m, sem carregar
   checkpoints e sem refazer inferência.
4. Agrega as células de 30 m ao suporte rotacionado de cada unidade de
   inventário de 50 x 50 m por área de interseção.
5. Aplica, para cada parcela, os coeficientes alométricos ajustados sem aquela
   parcela (plot-LOPO).
6. Compara a AGB estimada com a AGB observada em campo.
7. Resume resultados por seed, fold espacial e modelo, além do resultado OOF
   agrupado com média das três seeds.

IMPORTANTE
----------
- As predições salvas correspondem ao checkpoint selecionado em cada execução.
  Em algumas execuções TCCT, esse checkpoint pertence à Fase B/RVoG. Portanto,
  o produto é denominado "selected-checkpoint spatial OOF", e não "strict
  Phase-A OOF".
- O resultado é uma validação end-to-end dentro de Paracou, com separação
  espacial e alometria LOPO. Não constitui validação em um sítio independente.
- Não há treinamento nem inferência nova.

Uso no Prompt de Comando do Windows
-----------------------------------
<LOCAL_PATH> ^
"<PROJECT_ROOT>\Codigo\Calcular_AGB_Todos_Modelos_Predicoes_Salvas.py" ^
--project-root "<PROJECT_ROOT>"

Uso no Jupyter
--------------
import subprocess
subprocess.run(
    [
        r"<LOCAL_PATH>",
        r"<PROJECT_ROOT>\Codigo\Calcular_AGB_Todos_Modelos_Predicoes_Salvas.py",
        "--project-root",
        r"<PROJECT_ROOT>",
    ],
    check=True,
)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (42, 314, 2026)
FOLDS = tuple(range(5))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display: str
    path_parts: tuple[str, ...]
    summary_file: str
    kind: str = "neural"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="complete_tcct",
        display="Complete TCCT",
        path_parts=("tcct", "completo"),
        summary_file="resumo_final_tcct_completo.csv",
    ),
    ModelSpec(
        key="amplitude_only",
        display="Amplitude-only TCCT",
        path_parts=("tcct", "amplitude_only"),
        summary_file="resumo_final_tcct_amplitude_only.csv",
    ),
    ModelSpec(
        key="no_coattention",
        display="TCCT without co-attention",
        path_parts=("tcct_sem_coatencao", "completo"),
        summary_file="resumo_final_tcct_sem_coatencao_completo.csv",
    ),
    ModelSpec(
        key="cnn",
        display="CNN early fusion",
        path_parts=("cnn", "completo"),
        summary_file="resumo_final_cnn_completo.csv",
    ),
    ModelSpec(
        key="vit",
        display="Vision Transformer",
        path_parts=("vit", "completo"),
        summary_file="resumo_final_vit_completo.csv",
    ),
    ModelSpec(
        key="random_forest",
        display="Random Forest",
        path_parts=("random_forest",),
        summary_file="resumo_final_random_forest_completo.csv",
        kind="random_forest",
    ),
)


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            block = file.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def robust_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "sim"})
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "rrmse_pct": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
            "std_true": float("nan"),
            "std_pred": float("nan"),
            "std_ratio": float("nan"),
            "observed_mean": float("nan"),
        }

    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    observed_mean = float(np.mean(y_true))
    rrmse = float(100.0 * rmse / observed_mean) if observed_mean != 0 else float("nan")
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_true - observed_mean) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))

    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "rrmse_pct": rrmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "std_true": std_true,
        "std_pred": std_pred,
        "std_ratio": std_pred / std_true if std_true > 0 else float("nan"),
        "observed_mean": observed_mean,
    }


# -----------------------------------------------------------------------------
# Geometria: interseção de célula rotacionada com quadrado de patch
# -----------------------------------------------------------------------------


def polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    x = np.asarray([p[0] for p in vertices], dtype=np.float64)
    y = np.asarray([p[1] for p in vertices], dtype=np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon_axis(
    polygon: list[tuple[float, float]],
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []

    def inside(point: tuple[float, float]) -> bool:
        value = point[axis]
        return value >= boundary - 1e-10 if keep_greater else value <= boundary + 1e-10

    def intersection(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> tuple[float, float]:
        d = p2[axis] - p1[axis]
        if abs(d) < 1e-15:
            return p1
        t = (boundary - p1[axis]) / d
        t = min(1.0, max(0.0, t))
        return (
            p1[0] + t * (p2[0] - p1[0]),
            p1[1] + t * (p2[1] - p1[1]),
        )

    output: list[tuple[float, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)

    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside

    return output


def polygon_rectangle_intersection_area(
    polygon: list[tuple[float, float]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> float:
    clipped = polygon
    clipped = clip_polygon_axis(clipped, axis=0, boundary=xmin, keep_greater=True)
    clipped = clip_polygon_axis(clipped, axis=0, boundary=xmax, keep_greater=False)
    clipped = clip_polygon_axis(clipped, axis=1, boundary=ymin, keep_greater=True)
    clipped = clip_polygon_axis(clipped, axis=1, boundary=ymax, keep_greater=False)
    return polygon_area(clipped)


def inventory_cell_polygon(row: pd.Series) -> list[tuple[float, float]]:
    center = np.array([float(row.x_center), float(row.y_center)], dtype=np.float64)
    u = np.array([float(row.grid_u_x), float(row.grid_u_y)], dtype=np.float64)
    v = np.array([float(row.grid_v_x), float(row.grid_v_y)], dtype=np.float64)

    corners = (
        center - 0.5 * u - 0.5 * v,
        center + 0.5 * u - 0.5 * v,
        center + 0.5 * u + 0.5 * v,
        center - 0.5 * u + 0.5 * v,
    )
    return [(float(p[0]), float(p[1])) for p in corners]


# -----------------------------------------------------------------------------
# Carregamento dos manifestos e das predições salvas
# -----------------------------------------------------------------------------


def load_manifest_validation_rows(manifest_dir: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    per_fold: dict[int, pd.DataFrame] = {}
    all_rows: list[pd.DataFrame] = []

    for fold in FOLDS:
        path = manifest_dir / f"manifesto_fold_{fold}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        table = pd.read_csv(path)
        required = {
            "dataset_index", "fold", "particao", "x", "y", "chm_h", "chm_w"
        }
        missing = required - set(table.columns)
        if missing:
            raise KeyError(f"Missing columns in {path}: {sorted(missing)}")

        validation = table.loc[
            table["particao"].astype(str).str.lower().eq("validacao")
        ].copy()
        validation = validation.sort_values("dataset_index", kind="mergesort").reset_index(drop=True)

        if not np.all(validation["fold"].to_numpy(int) == fold):
            raise AssertionError(f"Fold column inconsistent in {path}")
        if validation["dataset_index"].duplicated().any():
            raise AssertionError(f"Duplicated validation indices in {path}")

        per_fold[fold] = validation
        all_rows.append(validation)

    patches = pd.concat(all_rows, ignore_index=True)
    if patches["dataset_index"].duplicated().any():
        duplicates = patches.loc[
            patches["dataset_index"].duplicated(keep=False), "dataset_index"
        ].tolist()[:20]
        raise AssertionError(f"Dataset indices occur in more than one validation fold: {duplicates}")

    patches = patches.sort_values("dataset_index", kind="mergesort").reset_index(drop=True)
    patches.insert(0, "patch_position", np.arange(len(patches), dtype=int))
    return patches, per_fold


def load_neural_prediction(
    multiseed_dir: Path,
    spec: ModelSpec,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, str, Path]:
    directory = multiseed_dir / f"seed_{seed}" / f"fold_{fold}"
    for part in spec.path_parts:
        directory = directory / part

    npz_path = directory / "predicoes_patch_final.npz"
    json_path = directory / "resumo_treinamento.json"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with np.load(npz_path) as data:
        if not {"y_true", "y_pred"}.issubset(data.files):
            raise KeyError(f"Unexpected NPZ keys in {npz_path}: {data.files}")
        y_true = np.asarray(data["y_true"], dtype=np.float64)
        y_pred = np.asarray(data["y_pred"], dtype=np.float64)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    phase = str(summary.get("fase_selecionada", "unknown"))
    return y_true, y_pred, phase, npz_path


def load_rf_prediction(
    multiseed_dir: Path,
    spec: ModelSpec,
    seed: int,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, str, Path]:
    directory = multiseed_dir / f"seed_{seed}" / f"fold_{fold}" / "random_forest"
    csv_path = directory / "predicoes_random_forest.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    table = pd.read_csv(csv_path)
    required = {"y_true", "y_pred"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"Missing RF prediction columns in {csv_path}: {sorted(missing)}")
    return (
        table["y_true"].to_numpy(dtype=np.float64),
        table["y_pred"].to_numpy(dtype=np.float64),
        "random_forest",
        csv_path,
    )


def load_all_predictions(
    multiseed_dir: Path,
    patches: pd.DataFrame,
    manifest_by_fold: dict[int, pd.DataFrame],
) -> tuple[
    np.ndarray,
    dict[str, dict[int, np.ndarray]],
    pd.DataFrame,
]:
    n_patches = len(patches)
    dataset_to_position = pd.Series(
        patches["patch_position"].to_numpy(int),
        index=patches["dataset_index"].to_numpy(int),
    )

    y_true_global = np.full(n_patches, np.nan, dtype=np.float64)
    predictions: dict[str, dict[int, np.ndarray]] = {
        spec.key: {
            seed: np.full(n_patches, np.nan, dtype=np.float64)
            for seed in SEEDS
        }
        for spec in MODEL_SPECS
    }
    phase_rows: list[dict[str, Any]] = []

    reference_by_fold: dict[int, np.ndarray] = {}

    for fold in FOLDS:
        manifest = manifest_by_fold[fold]
        dataset_indices = manifest["dataset_index"].to_numpy(int)
        positions = dataset_to_position.loc[dataset_indices].to_numpy(int)

        for spec in MODEL_SPECS:
            for seed in SEEDS:
                if spec.kind == "random_forest":
                    y_true, y_pred, phase, source = load_rf_prediction(
                        multiseed_dir, spec, seed, fold
                    )
                else:
                    y_true, y_pred, phase, source = load_neural_prediction(
                        multiseed_dir, spec, seed, fold
                    )

                if len(y_true) != len(manifest) or len(y_pred) != len(manifest):
                    raise AssertionError(
                        f"Length mismatch: model={spec.key}, seed={seed}, fold={fold}, "
                        f"predictions={len(y_true)}, manifest={len(manifest)}"
                    )

                if fold not in reference_by_fold:
                    reference_by_fold[fold] = y_true.copy()
                else:
                    tolerance = 1e-5 if spec.kind == "random_forest" else 0.0
                    if not np.allclose(
                        reference_by_fold[fold], y_true, rtol=0.0, atol=tolerance
                    ):
                        max_diff = float(np.max(np.abs(reference_by_fold[fold] - y_true)))
                        raise AssertionError(
                            f"y_true order mismatch: model={spec.key}, seed={seed}, "
                            f"fold={fold}, max_diff={max_diff}"
                        )

                if np.any(np.isfinite(predictions[spec.key][seed][positions])):
                    raise AssertionError(
                        f"Predictions already assigned: model={spec.key}, seed={seed}, fold={fold}"
                    )
                predictions[spec.key][seed][positions] = y_pred

                if np.any(np.isfinite(y_true_global[positions])):
                    if not np.allclose(y_true_global[positions], y_true, rtol=0.0, atol=1e-5):
                        raise AssertionError(f"Global y_true conflict in fold {fold}")
                else:
                    y_true_global[positions] = y_true

                phase_rows.append({
                    "model_key": spec.key,
                    "model": spec.display,
                    "seed": seed,
                    "fold": fold,
                    "selected_phase": phase,
                    "source_file": str(source),
                    "source_sha256": sha256_file(source),
                })

    if not np.all(np.isfinite(y_true_global)):
        raise AssertionError("Not all OOF y_true values were assigned")

    for spec in MODEL_SPECS:
        for seed in SEEDS:
            if not np.all(np.isfinite(predictions[spec.key][seed])):
                missing = int(np.sum(~np.isfinite(predictions[spec.key][seed])))
                raise AssertionError(
                    f"Missing OOF predictions: model={spec.key}, seed={seed}, n={missing}"
                )

    return y_true_global, predictions, pd.DataFrame(phase_rows)


# -----------------------------------------------------------------------------
# Relação espacial entre patches de 30 m e inventário de 50 m
# -----------------------------------------------------------------------------


def build_overlap_links(
    patches: pd.DataFrame,
    inventory: pd.DataFrame,
    patch_size_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    half = patch_size_m / 2.0
    patch_x = patches["x"].to_numpy(dtype=np.float64)
    patch_y = patches["y"].to_numpy(dtype=np.float64)
    patch_positions = patches["patch_position"].to_numpy(dtype=int)
    patch_folds = patches["fold"].to_numpy(dtype=int)

    link_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []

    for cell_position, row in inventory.reset_index(drop=True).iterrows():
        polygon = inventory_cell_polygon(row)
        polygon_np = np.asarray(polygon, dtype=np.float64)
        min_x, min_y = np.min(polygon_np, axis=0)
        max_x, max_y = np.max(polygon_np, axis=0)

        candidates = np.flatnonzero(
            (patch_x + half >= min_x)
            & (patch_x - half <= max_x)
            & (patch_y + half >= min_y)
            & (patch_y - half <= max_y)
        )

        fold_area = {fold: 0.0 for fold in FOLDS}
        total_overlap = 0.0
        n_intersecting = 0

        for candidate in candidates:
            area = polygon_rectangle_intersection_area(
                polygon,
                xmin=float(patch_x[candidate] - half),
                xmax=float(patch_x[candidate] + half),
                ymin=float(patch_y[candidate] - half),
                ymax=float(patch_y[candidate] + half),
            )
            if area <= 1e-8:
                continue

            position = int(patch_positions[candidate])
            fold = int(patch_folds[candidate])
            link_rows.append({
                "cell_position": int(cell_position),
                "plot_id": int(row.plot_id),
                "cell_id": int(row.cell_id),
                "patch_position": position,
                "dataset_index": int(patches.iloc[position]["dataset_index"]),
                "fold": fold,
                "overlap_area_m2": area,
            })
            fold_area[fold] += area
            total_overlap += area
            n_intersecting += 1

        cell_area = float(row.cell_area_inferred_m2)
        if not math.isfinite(cell_area) or cell_area <= 0:
            cell_area = polygon_area(polygon)

        dominant_fold = max(fold_area, key=fold_area.get) if total_overlap > 0 else -1
        dominant_area = fold_area.get(dominant_fold, 0.0) if dominant_fold >= 0 else 0.0

        cell_rows.append({
            "cell_position": int(cell_position),
            "plot_id": int(row.plot_id),
            "cell_id": int(row.cell_id),
            "inventory_cell_area_m2": cell_area,
            "patch_overlap_area_m2": total_overlap,
            "patch_coverage_fraction": total_overlap / cell_area if cell_area > 0 else np.nan,
            "n_intersecting_patches": n_intersecting,
            "dominant_fold": dominant_fold,
            "dominant_fold_fraction": dominant_area / total_overlap if total_overlap > 0 else np.nan,
            **{f"fold_{fold}_area_m2": fold_area[fold] for fold in FOLDS},
        })

    links = pd.DataFrame(link_rows)
    cells = pd.DataFrame(cell_rows)
    if links.empty:
        raise RuntimeError("No intersections between inventory cells and OOF patch footprints")
    return links, cells


def weighted_cell_values(
    values: np.ndarray,
    links: pd.DataFrame,
    n_cells: int,
) -> np.ndarray:
    patch_positions = links["patch_position"].to_numpy(dtype=int)
    cell_positions = links["cell_position"].to_numpy(dtype=int)
    weights = links["overlap_area_m2"].to_numpy(dtype=np.float64)
    patch_values = np.asarray(values, dtype=np.float64)[patch_positions]

    valid = np.isfinite(patch_values) & np.isfinite(weights) & (weights > 0)
    numerator = np.bincount(
        cell_positions[valid],
        weights=weights[valid] * patch_values[valid],
        minlength=n_cells,
    )
    denominator = np.bincount(
        cell_positions[valid],
        weights=weights[valid],
        minlength=n_cells,
    )

    output = np.full(n_cells, np.nan, dtype=np.float64)
    positive = denominator > 0
    output[positive] = numerator[positive] / denominator[positive]
    return output


# -----------------------------------------------------------------------------
# Resumos por modelo
# -----------------------------------------------------------------------------


def load_official_chm_summary(summary_dir: Path, spec: ModelSpec) -> dict[str, float]:
    path = summary_dir / spec.summary_file
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(table)}")
    row = table.iloc[0]
    return {
        "chm_rmse_mean": float(row.patch_rmse_mean),
        "chm_rmse_sd": float(row.patch_rmse_std_espacial),
        "chm_r2_mean": float(row.patch_r2_mean),
        "chm_r2_sd": float(row.patch_r2_std_espacial),
        "chm_bias_mean": float(row.patch_bias_mean),
        "chm_bias_sd": float(row.patch_bias_std_espacial),
        "chm_std_pred_mean": float(row.patch_std_pred_mean),
        "chm_std_true_mean": float(row.patch_std_true_mean),
    }


def summarize_spatial_folds(per_seed_fold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "rmse", "rrmse_pct", "mae", "bias", "r2", "std_true", "std_pred",
        "std_ratio", "observed_mean", "n",
    ]

    per_fold = (
        per_seed_fold
        .groupby(["model_key", "model", "fold"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
    )

    rows: list[dict[str, Any]] = []
    for (model_key, model), group in per_fold.groupby(["model_key", "model"], sort=False):
        row: dict[str, Any] = {
            "model_key": model_key,
            "model": model,
            "n_folds": int(group["fold"].nunique()),
            "n_cells_fold_mean": float(group["n"].mean()),
            "n_cells_fold_min": int(group["n"].min()),
            "n_cells_fold_max": int(group["n"].max()),
        }
        for metric in [
            "rmse", "rrmse_pct", "mae", "bias", "r2", "std_true", "std_pred",
            "std_ratio", "observed_mean",
        ]:
            row[f"agb_{metric}_mean"] = float(group[metric].mean())
            row[f"agb_{metric}_sd_spatial"] = float(group[metric].std(ddof=1))
        rows.append(row)

    return per_fold, pd.DataFrame(rows)


def format_mean_sd(mean: float, sd: float, decimals: int) -> str:
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def dataframe_to_latex_full(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Selected-checkpoint spatial OOF performance for canopy height and field-referenced AGB estimation. AGB was obtained after area-weighted aggregation of OOF patch-mean CHM predictions to the 50$\times$50~m inventory support and plot-LOPO allometric conversion. Values are mean $\pm$ spatial standard deviation over five folds after averaging the three seeds within each fold.}",
        r"\label{tab:main_results_agb}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Model & \multicolumn{3}{c}{CHM} & \multicolumn{4}{c}{AGB} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r" & RMSE (m)$\downarrow$ & $R^2\uparrow$ & Bias (m) & RMSE (Mg/ha)$\downarrow$ & rRMSE (\%)$\downarrow$ & $R^2\uparrow$ & Bias (Mg/ha) \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        model = str(row["model"]).replace("_", r"\_")
        values = [
            model,
            format_mean_sd(row.chm_rmse_mean, row.chm_rmse_sd, 3),
            format_mean_sd(row.chm_r2_mean, row.chm_r2_sd, 3),
            format_mean_sd(row.chm_bias_mean, row.chm_bias_sd, 3),
            format_mean_sd(row.agb_rmse_mean, row.agb_rmse_sd_spatial, 1),
            format_mean_sd(row.agb_rrmse_pct_mean, row.agb_rrmse_pct_sd_spatial, 2),
            format_mean_sd(row.agb_r2_mean, row.agb_r2_sd_spatial, 3),
            format_mean_sd(row.agb_bias_mean, row.agb_bias_sd_spatial, 1),
        ]
        lines.append(" & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def dataframe_to_latex_compact(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Selected-checkpoint spatial OOF comparison. AGB metrics are field-referenced and use plot-LOPO allometry at the 50$\times$50~m inventory support. Values are mean $\pm$ spatial standard deviation across five folds.}",
        r"\label{tab:main_results_agb_compact}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & CHM RMSE (m)$\downarrow$ & CHM $R^2\uparrow$ & AGB RMSE (Mg/ha)$\downarrow$ & AGB rRMSE (\%)$\downarrow$ & AGB $R^2\uparrow$ \\",
        r"\midrule",
    ]

    for _, row in table.iterrows():
        model = str(row["model"]).replace("_", r"\_")
        values = [
            model,
            format_mean_sd(row.chm_rmse_mean, row.chm_rmse_sd, 3),
            format_mean_sd(row.chm_r2_mean, row.chm_r2_sd, 3),
            format_mean_sd(row.agb_rmse_mean, row.agb_rmse_sd_spatial, 1),
            format_mean_sd(row.agb_rrmse_pct_mean, row.agb_rrmse_pct_sd_spatial, 2),
            format_mean_sd(row.agb_r2_mean, row.agb_r2_sd_spatial, 3),
        ]
        lines.append(" & ".join(values) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def save_model_scatter(
    path: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict[str, float],
) -> None:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(y_true, y_pred, s=18, alpha=0.38, rasterized=True)
    lower = float(min(np.min(y_true), np.min(y_pred)))
    upper = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.7, label="Identity line (1:1)")
    ax.set_title(f"Field-referenced AGB — {model_name}")
    ax.set_xlabel("Observed AGB (Mg/ha)")
    ax.set_ylabel("Predicted AGB (Mg/ha)")
    ax.grid(True, alpha=0.25)
    ax.legend(
        title=(
            f"RMSE = {metrics['rmse']:.1f} Mg/ha\n"
            f"rRMSE = {metrics['rrmse_pct']:.2f}%\n"
            f"$R^2$ = {metrics['r2']:.3f}"
        )
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Execução principal
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate field-referenced AGB metrics for all saved OOF model predictions."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"<PROJECT_ROOT>"),
        help="Root of SIBGRAPI_REV.",
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--allometry-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--patch-size-m",
        type=float,
        default=30.0,
        help="Ground footprint represented by each saved patch-mean prediction.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Minimum inventory-cell area covered by OOF patch footprints.",
    )
    parser.add_argument(
        "--min-fold-purity",
        type=float,
        default=0.95,
        help="Minimum fraction of a cell's covered area assigned to one fold for fold-wise summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else (project_root / "Resultados" / "TCCT_v3_2_SIBGRAPI").resolve()
    )
    allometry_dir = (
        args.allometry_dir.expanduser().resolve()
        if args.allometry_dir is not None
        else (results_dir / "AGB_Allometry_SupportMatched").resolve()
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (results_dir / "AGB_All_Methods_Saved_OOF").resolve()
    )

    manifest_dir = results_dir / "manifestos_folds"
    multiseed_dir = results_dir / "multiseed"
    summary_dir = results_dir / "resumos_multiseed"

    inventory_path = allometry_dir / "inventory_50m_support_matched.csv"
    lopo_coefficients_path = allometry_dir / "lopo_coefficients.csv"
    lopo_predictions_path = allometry_dir / "lopo_agb_predictions.csv"
    allometry_metrics_path = allometry_dir / "allometry_coefficients_and_metrics.json"

    print("=" * 96)
    print("FIELD-REFERENCED AGB METRICS FOR ALL SAVED OOF MODEL PREDICTIONS")
    print("=" * 96)
    print(f"Project root:   {project_root}")
    print(f"Results:        {results_dir}")
    print(f"Allometry:      {allometry_dir}")
    print(f"Output:         {output_dir}")
    print(f"Patch size:     {args.patch_size_m:.2f} m")
    print(f"Min coverage:   {args.min_coverage:.2f}")
    print(f"Min fold purity:{args.min_fold_purity:.2f}")

    required_paths = [
        manifest_dir, multiseed_dir, summary_dir, inventory_path,
        lopo_coefficients_path, lopo_predictions_path, allometry_metrics_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    patches, manifest_by_fold = load_manifest_validation_rows(manifest_dir)
    print(f"✓ OOF patch manifests loaded: {len(patches)} unique validation patches")

    y_true_patch, predictions, phase_manifest = load_all_predictions(
        multiseed_dir, patches, manifest_by_fold
    )
    phase_manifest.to_csv(output_dir / "selected_phase_manifest.csv", index=False)
    print("✓ Saved predictions loaded and aligned for all models, seeds, and folds")

    inventory = pd.read_csv(inventory_path)
    required_inventory = {
        "plot_id", "cell_id", "x_center", "y_center", "grid_u_x", "grid_u_y",
        "grid_v_x", "grid_v_y", "cell_area_inferred_m2", "agb_observed_mg_ha",
        "chm_mean_m", "accepted_for_calibration",
    }
    missing_inventory = required_inventory - set(inventory.columns)
    if missing_inventory:
        raise KeyError(f"Missing inventory columns: {sorted(missing_inventory)}")

    inventory = inventory.loc[robust_bool(inventory["accepted_for_calibration"])].copy()
    inventory = inventory.sort_values(["plot_id", "cell_id"], kind="mergesort").reset_index(drop=True)

    lopo_coefficients = pd.read_csv(lopo_coefficients_path)
    lopo_coefficients = lopo_coefficients.rename(columns={"held_out_plot": "plot_id"})
    inventory = inventory.merge(
        lopo_coefficients[["plot_id", "a", "b"]],
        on="plot_id",
        how="left",
        validate="many_to_one",
    )
    if inventory[["a", "b"]].isna().any().any():
        missing_plots = inventory.loc[inventory["a"].isna() | inventory["b"].isna(), "plot_id"].unique()
        raise AssertionError(f"Missing LOPO coefficients for plots: {missing_plots.tolist()}")

    links, cell_support = build_overlap_links(
        patches=patches,
        inventory=inventory,
        patch_size_m=float(args.patch_size_m),
    )
    links.to_csv(output_dir / "inventory_patch_overlap_links.csv", index=False)

    cells = inventory.copy()
    cells.insert(0, "cell_position", np.arange(len(cells), dtype=int))
    cells = cells.merge(cell_support, on=["cell_position", "plot_id", "cell_id"], how="left", validate="one_to_one")

    cells["eligible_pooled"] = (
        np.isfinite(cells["patch_coverage_fraction"])
        & (cells["patch_coverage_fraction"] >= float(args.min_coverage))
    )
    cells["eligible_fold_summary"] = (
        cells["eligible_pooled"]
        & np.isfinite(cells["dominant_fold_fraction"])
        & (cells["dominant_fold_fraction"] >= float(args.min_fold_purity))
    )

    y_true_50m_from_patches = weighted_cell_values(
        y_true_patch, links, n_cells=len(cells)
    )
    cells["chm_true_from_saved_patch_means_m"] = y_true_50m_from_patches

    support_audit_mask = cells["eligible_pooled"].to_numpy(bool)
    support_audit = calculate_metrics(
        cells.loc[support_audit_mask, "chm_mean_m"].to_numpy(float),
        cells.loc[support_audit_mask, "chm_true_from_saved_patch_means_m"].to_numpy(float),
    )

    print(
        "✓ Support audit (canonical 50 m LiDAR CHM vs area-weighted saved patch means): "
        f"n={support_audit['n']}, RMSE={support_audit['rmse']:.3f} m, "
        f"bias={support_audit['bias']:.3f} m, R²={support_audit['r2']:.3f}"
    )

    cell_prediction_rows: list[pd.DataFrame] = []
    per_seed_fold_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []

    pooled_predictions_by_model: dict[str, tuple[np.ndarray, np.ndarray, dict[str, float]]] = {}

    for spec in MODEL_SPECS:
        seed_chm_predictions: dict[int, np.ndarray] = {}

        for seed in SEEDS:
            chm_pred_50m = weighted_cell_values(
                predictions[spec.key][seed], links, n_cells=len(cells)
            )
            seed_chm_predictions[seed] = chm_pred_50m
            agb_pred = (
                cells["a"].to_numpy(float)
                * np.maximum(chm_pred_50m, 0.0) ** cells["b"].to_numpy(float)
            )

            seed_table = pd.DataFrame({
                "model_key": spec.key,
                "model": spec.display,
                "seed": seed,
                "plot_id": cells["plot_id"].to_numpy(int),
                "cell_id": cells["cell_id"].to_numpy(int),
                "dominant_fold": cells["dominant_fold"].to_numpy(int),
                "dominant_fold_fraction": cells["dominant_fold_fraction"].to_numpy(float),
                "patch_coverage_fraction": cells["patch_coverage_fraction"].to_numpy(float),
                "chm_predicted_50m_m": chm_pred_50m,
                "agb_observed_mg_ha": cells["agb_observed_mg_ha"].to_numpy(float),
                "agb_predicted_mg_ha": agb_pred,
                "lopo_a": cells["a"].to_numpy(float),
                "lopo_b": cells["b"].to_numpy(float),
                "eligible_pooled": cells["eligible_pooled"].to_numpy(bool),
                "eligible_fold_summary": cells["eligible_fold_summary"].to_numpy(bool),
            })
            cell_prediction_rows.append(seed_table)

            for fold in FOLDS:
                mask = (
                    cells["eligible_fold_summary"].to_numpy(bool)
                    & (cells["dominant_fold"].to_numpy(int) == fold)
                )
                metrics = calculate_metrics(
                    cells.loc[mask, "agb_observed_mg_ha"].to_numpy(float),
                    agb_pred[mask],
                )
                per_seed_fold_rows.append({
                    "model_key": spec.key,
                    "model": spec.display,
                    "seed": seed,
                    "fold": fold,
                    **metrics,
                })

        # Ensemble OOF: average CHM across seeds before the nonlinear allometry.
        chm_seed_mean = np.mean(
            np.stack([seed_chm_predictions[seed] for seed in SEEDS], axis=0),
            axis=0,
        )
        agb_seed_mean = (
            cells["a"].to_numpy(float)
            * np.maximum(chm_seed_mean, 0.0) ** cells["b"].to_numpy(float)
        )
        pooled_mask = cells["eligible_pooled"].to_numpy(bool)
        pooled_metrics = calculate_metrics(
            cells.loc[pooled_mask, "agb_observed_mg_ha"].to_numpy(float),
            agb_seed_mean[pooled_mask],
        )
        pooled_rows.append({
            "model_key": spec.key,
            "model": spec.display,
            **pooled_metrics,
        })
        pooled_predictions_by_model[spec.key] = (
            cells.loc[pooled_mask, "agb_observed_mg_ha"].to_numpy(float),
            agb_seed_mean[pooled_mask],
            pooled_metrics,
        )

        save_model_scatter(
            figure_dir / f"agb_observed_vs_predicted_{spec.key}.png",
            model_name=spec.display,
            y_true=pooled_predictions_by_model[spec.key][0],
            y_pred=pooled_predictions_by_model[spec.key][1],
            metrics=pooled_metrics,
        )

    cell_predictions = pd.concat(cell_prediction_rows, ignore_index=True)
    per_seed_fold = pd.DataFrame(per_seed_fold_rows)
    pooled_metrics_table = pd.DataFrame(pooled_rows)

    per_fold, agb_spatial_summary = summarize_spatial_folds(per_seed_fold)

    cell_predictions.to_csv(output_dir / "agb_predictions_all_models_all_seeds.csv", index=False)
    per_seed_fold.to_csv(output_dir / "agb_metrics_per_seed_fold.csv", index=False)
    per_fold.to_csv(output_dir / "agb_metrics_seed_mean_by_fold.csv", index=False)
    agb_spatial_summary.to_csv(output_dir / "agb_metrics_spatial_summary.csv", index=False)
    pooled_metrics_table.to_csv(output_dir / "agb_metrics_pooled_oof_seed_mean.csv", index=False)

    cells.to_csv(output_dir / "inventory_support_and_fold_assignment.csv", index=False)

    # LOPO LiDAR baseline already calculated by the support-matched allometry script.
    lopo_predictions = pd.read_csv(lopo_predictions_path)
    lopo_baseline_metrics = calculate_metrics(
        lopo_predictions["agb_observed_mg_ha"].to_numpy(float),
        lopo_predictions["agb_predicted_lopo_mg_ha"].to_numpy(float),
    )
    baseline_row = pd.DataFrame([{
        "model_key": "canonical_lidar_lopo_allometry",
        "model": "Canonical LiDAR CHM + plot-LOPO allometry",
        **lopo_baseline_metrics,
    }])
    pd.concat([baseline_row, pooled_metrics_table], ignore_index=True).to_csv(
        output_dir / "agb_field_reference_baseline_and_models.csv", index=False
    )

    # Main table: official selected-checkpoint CHM summary + AGB spatial summary.
    chm_rows = []
    for spec in MODEL_SPECS:
        chm_rows.append({
            "model_key": spec.key,
            "model": spec.display,
            **load_official_chm_summary(summary_dir, spec),
        })
    chm_summary = pd.DataFrame(chm_rows)
    main_table = chm_summary.merge(
        agb_spatial_summary,
        on=["model_key", "model"],
        how="left",
        validate="one_to_one",
    )

    ordered_columns = [
        "model_key", "model",
        "chm_rmse_mean", "chm_rmse_sd", "chm_r2_mean", "chm_r2_sd",
        "chm_bias_mean", "chm_bias_sd", "chm_std_pred_mean", "chm_std_true_mean",
        "agb_rmse_mean", "agb_rmse_sd_spatial",
        "agb_rrmse_pct_mean", "agb_rrmse_pct_sd_spatial",
        "agb_r2_mean", "agb_r2_sd_spatial",
        "agb_bias_mean", "agb_bias_sd_spatial",
        "agb_std_pred_mean", "agb_std_true_mean", "agb_std_ratio_mean",
        "n_folds", "n_cells_fold_mean", "n_cells_fold_min", "n_cells_fold_max",
    ]
    main_table = main_table[ordered_columns]
    main_table.to_csv(output_dir / "Table_I_CHM_AGB_all_models_numeric.csv", index=False)

    formatted = pd.DataFrame({
        "Model": main_table["model"],
        "CHM RMSE (m)": [
            format_mean_sd(m, s, 3)
            for m, s in zip(main_table.chm_rmse_mean, main_table.chm_rmse_sd)
        ],
        "CHM R2": [
            format_mean_sd(m, s, 3)
            for m, s in zip(main_table.chm_r2_mean, main_table.chm_r2_sd)
        ],
        "CHM Bias (m)": [
            format_mean_sd(m, s, 3)
            for m, s in zip(main_table.chm_bias_mean, main_table.chm_bias_sd)
        ],
        "AGB RMSE (Mg/ha)": [
            format_mean_sd(m, s, 1)
            for m, s in zip(main_table.agb_rmse_mean, main_table.agb_rmse_sd_spatial)
        ],
        "AGB rRMSE (%)": [
            format_mean_sd(m, s, 2)
            for m, s in zip(main_table.agb_rrmse_pct_mean, main_table.agb_rrmse_pct_sd_spatial)
        ],
        "AGB R2": [
            format_mean_sd(m, s, 3)
            for m, s in zip(main_table.agb_r2_mean, main_table.agb_r2_sd_spatial)
        ],
        "AGB Bias (Mg/ha)": [
            format_mean_sd(m, s, 1)
            for m, s in zip(main_table.agb_bias_mean, main_table.agb_bias_sd_spatial)
        ],
    })
    formatted.to_csv(output_dir / "Table_I_CHM_AGB_all_models_formatted.csv", index=False)

    try:
        with pd.ExcelWriter(output_dir / "Table_I_CHM_AGB_all_models.xlsx", engine="openpyxl") as writer:
            main_table.to_excel(writer, sheet_name="Numeric", index=False)
            formatted.to_excel(writer, sheet_name="Formatted", index=False)
            per_seed_fold.to_excel(writer, sheet_name="AGB seed-fold", index=False)
            per_fold.to_excel(writer, sheet_name="AGB fold summary", index=False)
            pooled_metrics_table.to_excel(writer, sheet_name="AGB pooled OOF", index=False)
            cells.to_excel(writer, sheet_name="Support audit", index=False)
    except ModuleNotFoundError:
        print("⚠ openpyxl is not installed; XLSX output was skipped. CSV files were saved.")

    (output_dir / "Table_I_CHM_AGB_all_models_full.tex").write_text(
        dataframe_to_latex_full(main_table), encoding="utf-8"
    )
    (output_dir / "Table_I_CHM_AGB_all_models_compact.tex").write_text(
        dataframe_to_latex_compact(main_table), encoding="utf-8"
    )

    # Comparison plot for AGB rRMSE based on the fold summary.
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(main_table))
    ax.bar(
        x,
        main_table["agb_rrmse_pct_mean"].to_numpy(float),
        yerr=main_table["agb_rrmse_pct_sd_spatial"].to_numpy(float),
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(main_table["model"], rotation=25, ha="right")
    ax.set_ylabel("Field-referenced AGB rRMSE (%)")
    ax.set_title("AGB estimation across models — spatial OOF + plot-LOPO allometry")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "agb_rrmse_all_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    allometry_metrics = json.loads(allometry_metrics_path.read_text(encoding="utf-8"))
    summary = {
        "generated_at": datetime.now().isoformat(),
        "protocol": (
            "saved selected-checkpoint spatial OOF patch-mean CHM predictions; "
            "area-weighted 30 m to rotated 50x50 m inventory support; "
            "plot-LOPO allometric coefficients; field-referenced AGB evaluation"
        ),
        "project_root": str(project_root),
        "results_dir": str(results_dir),
        "allometry_dir": str(allometry_dir),
        "output_dir": str(output_dir),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "models": [spec.display for spec in MODEL_SPECS],
        "patch_size_m": float(args.patch_size_m),
        "minimum_patch_coverage": float(args.min_coverage),
        "minimum_fold_purity": float(args.min_fold_purity),
        "n_inventory_cells_accepted_allometry": int(len(inventory)),
        "n_inventory_cells_eligible_pooled": int(cells["eligible_pooled"].sum()),
        "n_inventory_cells_eligible_fold_summary": int(cells["eligible_fold_summary"].sum()),
        "support_audit_canonical_vs_saved_patch_true": support_audit,
        "lopo_lidar_allometry_baseline": lopo_baseline_metrics,
        "allometry_original_summary": allometry_metrics,
        "pooled_oof_seed_mean_metrics": pooled_metrics_table.to_dict(orient="records"),
        "spatial_fold_summary": agb_spatial_summary.to_dict(orient="records"),
        "limitations": [
            "The saved predictions represent the selected checkpoint of each run, not strict Phase A in the four runs where Phase B was selected.",
            "The AGB evaluation is within-site at Paracou and is not independent-site validation.",
            "Patch-mean CHM predictions are aggregated by geometric overlap; the support audit quantifies the approximation relative to canonical 50 m LiDAR CHM.",
            "The allometry is excluded plot-wise (LOPO), while CHM predictions are spatially OOF by grouped folds.",
        ],
        "file_hashes": {
            "inventory_50m_support_matched.csv": sha256_file(inventory_path),
            "lopo_coefficients.csv": sha256_file(lopo_coefficients_path),
            "lopo_agb_predictions.csv": sha256_file(lopo_predictions_path),
            "allometry_coefficients_and_metrics.json": sha256_file(allometry_metrics_path),
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nTABLE I — CHM AND FIELD-REFERENCED AGB")
    print(formatted.to_string(index=False))
    print("\nPooled OOF seed-mean metrics (describes the pooled cell cloud):")
    print(pooled_metrics_table[["model", "n", "rmse", "rrmse_pct", "r2", "bias"]].to_string(index=False))
    print(f"\n✓ Outputs saved to: {output_dir}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("\n" + "!" * 96, file=sys.stderr)
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        print("!" * 96, file=sys.stderr)
        raise
