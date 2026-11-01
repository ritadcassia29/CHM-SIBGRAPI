#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Generate pairwise comparison figures directly from predictions saved during training.

This script DOES NOT:
- recreate spatial folds;
- reload neural checkpoints;
- rerun inference;
- retrain any model;
- recalculate Random Forest features.

It reads only the prediction files written by the original training runs:
- neural models: predicoes_patch_final.npz
- Random Forest: predicoes_random_forest.csv

The resulting figures therefore reproduce the selected-checkpoint predictions
actually evaluated during each seed/fold run.

Important methodological note
-----------------------------
Four of the 75 neural runs selected Phase B (RVoG) as their final checkpoint:
- Complete TCCT: seed 2026, fold 3
- TCCT without co-attention: seed 42/fold 0, seed 314/fold 1,
  and seed 314/fold 3

Consequently, these point-cloud figures represent the checkpoint selected by
the original training protocol, not a strictly Phase-A-only comparison.
The strict Phase-A table should remain a separate numerical analysis.

Usage
-----
Windows command prompt:

<LOCAL_PATH> ^
"<PROJECT_ROOT>\Codigo\Gerar_Figuras_Pareadas_Predicoes_Salvas.py" ^
--project-root "<PROJECT_ROOT>"

Jupyter:

import subprocess

subprocess.run(
    [
        r"<LOCAL_PATH>",
        r"<PROJECT_ROOT>\Codigo\Gerar_Figuras_Pareadas_Predicoes_Salvas.py",
        "--project-root",
        r"<PROJECT_ROOT>",
    ],
    check=True,
)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (42, 314, 2026)
FOLDS = tuple(range(5))

MODEL_SPECS = {
    "complete_tcct": {
        "display": "Complete TCCT",
        "path": ("tcct", "completo"),
        "summary": "resumo_final_tcct_completo.csv",
        "color": "#009E73",
        "marker": "o",
    },
    "amplitude_only": {
        "display": "Amplitude-only TCCT",
        "path": ("tcct", "amplitude_only"),
        "summary": "resumo_final_tcct_amplitude_only.csv",
        "color": "#E69F00",
        "marker": "x",
    },
    "no_coattention": {
        "display": "TCCT without co-attention",
        "path": ("tcct_sem_coatencao", "completo"),
        "summary": "resumo_final_tcct_sem_coatencao_completo.csv",
        "color": "#D55E00",
        "marker": "x",
    },
    "cnn": {
        "display": "CNN early fusion",
        "path": ("cnn", "completo"),
        "summary": "resumo_final_cnn_completo.csv",
        "color": "#0072B2",
        "marker": "x",
    },
    "vit": {
        "display": "Vision Transformer",
        "path": ("vit", "completo"),
        "summary": "resumo_final_vit_completo.csv",
        "color": "#CC79A7",
        "marker": "x",
    },
    "random_forest": {
        "display": "Random Forest",
        "path": ("random_forest",),
        "summary": "resumo_final_random_forest_completo.csv",
        "color": "#F05A71",
        "marker": "x",
    },
}

COMPARISONS = (
    ("complete_tcct", "amplitude_only"),
    ("complete_tcct", "no_coattention"),
    ("complete_tcct", "cnn"),
    ("complete_tcct", "vit"),
    ("complete_tcct", "random_forest"),
)


@dataclass
class PredictionRun:
    model_key: str
    seed: int
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    selected_phase: str
    source_file: Path


def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes incompatíveis: y_true={y_true.shape}, y_pred={y_pred.shape}"
        )

    erro = y_pred - y_true
    rmse = float(np.sqrt(np.mean(erro ** 2)))
    bias = float(np.mean(erro))

    ss_res = float(np.sum(erro ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    std_pred = float(np.std(y_pred))
    std_true = float(np.std(y_true))

    return {
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "std_pred": std_pred,
        "std_true": std_true,
        "std_ratio": std_pred / std_true if std_true > 0 else float("nan"),
    }


def carregar_neural(
    multiseed_dir: Path,
    model_key: str,
    seed: int,
    fold: int,
) -> PredictionRun:
    spec = MODEL_SPECS[model_key]
    pasta = multiseed_dir / f"seed_{seed}" / f"fold_{fold}"

    for parte in spec["path"]:
        pasta = pasta / parte

    caminho_npz = pasta / "predicoes_patch_final.npz"
    caminho_json = pasta / "resumo_treinamento.json"

    if not caminho_npz.exists():
        raise FileNotFoundError(caminho_npz)
    if not caminho_json.exists():
        raise FileNotFoundError(caminho_json)

    with np.load(caminho_npz) as dados:
        if set(dados.files) != {"y_true", "y_pred"}:
            raise KeyError(
                f"NPZ inesperado em {caminho_npz}: {dados.files}"
            )
        y_true = np.asarray(dados["y_true"], dtype=np.float64)
        y_pred = np.asarray(dados["y_pred"], dtype=np.float64)

    resumo = json.loads(caminho_json.read_text(encoding="utf-8"))
    fase = str(resumo.get("fase_selecionada", "unknown"))

    calculadas = metricas(y_true, y_pred)
    registradas = resumo["metricas_patch_final"]

    for chave in ("rmse", "r2", "bias", "std_pred", "std_true"):
        if not math.isclose(
            calculadas[chave],
            float(registradas[chave]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise AssertionError(
                f"Métrica divergente em {caminho_npz}: "
                f"{chave} calculada={calculadas[chave]} "
                f"registrada={registradas[chave]}"
            )

    return PredictionRun(
        model_key=model_key,
        seed=seed,
        fold=fold,
        y_true=y_true,
        y_pred=y_pred,
        selected_phase=fase,
        source_file=caminho_npz,
    )


def carregar_rf(
    multiseed_dir: Path,
    seed: int,
    fold: int,
) -> PredictionRun:
    pasta = (
        multiseed_dir
        / f"seed_{seed}"
        / f"fold_{fold}"
        / "random_forest"
    )

    caminho_csv = pasta / "predicoes_random_forest.csv"
    caminho_json = pasta / "metricas_random_forest.json"

    if not caminho_csv.exists():
        raise FileNotFoundError(caminho_csv)
    if not caminho_json.exists():
        raise FileNotFoundError(caminho_json)

    tabela = pd.read_csv(caminho_csv)
    esperadas = {"y_true", "y_pred"}
    if not esperadas.issubset(tabela.columns):
        raise KeyError(
            f"Colunas ausentes em {caminho_csv}: "
            f"{sorted(esperadas - set(tabela.columns))}"
        )

    y_true = tabela["y_true"].to_numpy(dtype=np.float64)
    y_pred = tabela["y_pred"].to_numpy(dtype=np.float64)

    calculadas = metricas(y_true, y_pred)
    registradas = json.loads(caminho_json.read_text(encoding="utf-8"))

    for chave in ("rmse", "r2", "bias", "std_pred", "std_true"):
        if not math.isclose(
            calculadas[chave],
            float(registradas[chave]),
            rel_tol=0.0,
            abs_tol=5e-8,
        ):
            raise AssertionError(
                f"Métrica RF divergente em {caminho_csv}: "
                f"{chave} calculada={calculadas[chave]} "
                f"registrada={registradas[chave]}"
            )

    return PredictionRun(
        model_key="random_forest",
        seed=seed,
        fold=fold,
        y_true=y_true,
        y_pred=y_pred,
        selected_phase="random_forest",
        source_file=caminho_csv,
    )


def carregar_execucoes(multiseed_dir: Path) -> dict[str, list[PredictionRun]]:
    execucoes: dict[str, list[PredictionRun]] = {
        chave: [] for chave in MODEL_SPECS
    }

    for seed in SEEDS:
        for fold in FOLDS:
            for model_key in MODEL_SPECS:
                if model_key == "random_forest":
                    run = carregar_rf(multiseed_dir, seed, fold)
                else:
                    run = carregar_neural(
                        multiseed_dir,
                        model_key,
                        seed,
                        fold,
                    )
                execucoes[model_key].append(run)

    return execucoes


def validar_alinhamento(
    execucoes: dict[str, list[PredictionRun]],
) -> pd.DataFrame:
    linhas = []

    for fold in FOLDS:
        referencias = [
            run
            for run in execucoes["complete_tcct"]
            if run.fold == fold
        ]
        referencias = sorted(referencias, key=lambda r: r.seed)
        y_ref = referencias[0].y_true

        for model_key, runs in execucoes.items():
            for run in sorted(
                [r for r in runs if r.fold == fold],
                key=lambda r: r.seed,
            ):
                tolerancia = 1e-5 if model_key == "random_forest" else 0.0
                alinhado = np.allclose(
                    y_ref,
                    run.y_true,
                    rtol=0.0,
                    atol=tolerancia,
                )
                max_diff = float(
                    np.max(np.abs(y_ref - run.y_true))
                )

                linhas.append({
                    "model_key": model_key,
                    "model": MODEL_SPECS[model_key]["display"],
                    "seed": run.seed,
                    "fold": run.fold,
                    "selected_phase": run.selected_phase,
                    "n_samples": int(len(run.y_true)),
                    "y_true_aligned": bool(alinhado),
                    "max_abs_y_true_difference": max_diff,
                    "source_file": str(run.source_file),
                })

                if not alinhado:
                    raise AssertionError(
                        f"y_true não alinhado: modelo={model_key}, "
                        f"seed={run.seed}, fold={run.fold}, "
                        f"max_diff={max_diff}"
                    )

    return pd.DataFrame(linhas)


def construir_oof_seed_mean(
    runs: list[PredictionRun],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y_folds: list[np.ndarray] = []
    pred_folds: list[np.ndarray] = []
    linhas_metricas = []

    for fold in FOLDS:
        runs_fold = sorted(
            [run for run in runs if run.fold == fold],
            key=lambda run: run.seed,
        )

        if len(runs_fold) != len(SEEDS):
            raise AssertionError(
                f"Fold {fold}: esperado {len(SEEDS)} seeds, "
                f"encontrado {len(runs_fold)}"
            )

        y_true = runs_fold[0].y_true
        for run in runs_fold[1:]:
            np.testing.assert_allclose(
                y_true,
                run.y_true,
                rtol=0.0,
                atol=1e-5,
            )

        y_pred = np.mean(
            np.stack([run.y_pred for run in runs_fold], axis=0),
            axis=0,
        )

        met = metricas(y_true, y_pred)
        linhas_metricas.append({
            "fold": fold,
            "n_samples": int(len(y_true)),
            **met,
        })

        y_folds.append(y_true)
        pred_folds.append(y_pred)

    return (
        np.concatenate(y_folds),
        np.concatenate(pred_folds),
        pd.DataFrame(linhas_metricas),
    )


def carregar_resumo_oficial(
    resumos_dir: Path,
    model_key: str,
) -> dict[str, float]:
    caminho = resumos_dir / MODEL_SPECS[model_key]["summary"]

    if not caminho.exists():
        raise FileNotFoundError(caminho)

    tabela = pd.read_csv(caminho)
    if len(tabela) != 1:
        raise ValueError(
            f"Resumo deveria ter uma linha: {caminho}"
        )

    linha = tabela.iloc[0]

    return {
        "rmse_mean": float(linha["patch_rmse_mean"]),
        "rmse_sd": float(linha["patch_rmse_std_espacial"]),
        "r2_mean": float(linha["patch_r2_mean"]),
        "r2_sd": float(linha["patch_r2_std_espacial"]),
        "bias_mean": float(linha["patch_bias_mean"]),
        "bias_sd": float(linha["patch_bias_std_espacial"]),
        "std_pred_mean": float(linha["patch_std_pred_mean"]),
        "std_true_mean": float(linha["patch_std_true_mean"]),
    }


def label_oficial(model_key: str, resumo: dict[str, float]) -> str:
    nome = MODEL_SPECS[model_key]["display"]
    return (
        f"{nome}\n"
        f"RMSE = {resumo['rmse_mean']:.2f} ± {resumo['rmse_sd']:.2f} m; "
        f"$R^2$ = {resumo['r2_mean']:.3f} ± {resumo['r2_sd']:.3f}"
    )


def limites_grafico(*arrays: np.ndarray) -> tuple[float, float]:
    concatenado = np.concatenate(
        [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
    )
    concatenado = concatenado[np.isfinite(concatenado)]

    minimo = float(np.percentile(concatenado, 0.1))
    maximo = float(np.percentile(concatenado, 99.9))
    margem = max(1.0, 0.04 * (maximo - minimo))

    return max(0.0, minimo - margem), maximo + margem


def gerar_figura(
    output_dir: Path,
    base_key: str,
    comparison_key: str,
    dados_oof: dict[str, tuple[np.ndarray, np.ndarray]],
    resumos: dict[str, dict[str, float]],
) -> Path:
    y_base, pred_base = dados_oof[base_key]
    y_comp, pred_comp = dados_oof[comparison_key]

    np.testing.assert_allclose(
        y_base,
        y_comp,
        rtol=0.0,
        atol=1e-5,
    )

    spec_base = MODEL_SPECS[base_key]
    spec_comp = MODEL_SPECS[comparison_key]

    fig, ax = plt.subplots(figsize=(8.2, 8.2))

    ax.scatter(
        y_comp,
        pred_comp,
        s=14,
        alpha=0.24,
        marker=spec_comp["marker"],
        linewidths=0.9,
        color=spec_comp["color"],
        rasterized=True,
        label=label_oficial(comparison_key, resumos[comparison_key]),
        zorder=2,
    )

    ax.scatter(
        y_base,
        pred_base,
        s=16,
        alpha=0.38,
        marker=spec_base["marker"],
        linewidths=0.35,
        edgecolors="white",
        color=spec_base["color"],
        rasterized=True,
        label=label_oficial(base_key, resumos[base_key]),
        zorder=3,
    )

    minimo, maximo = limites_grafico(
        y_base,
        pred_base,
        pred_comp,
    )

    ax.plot(
        [minimo, maximo],
        [minimo, maximo],
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="Identity line (1:1)",
        zorder=4,
    )

    ax.set_xlim(minimo, maximo)
    ax.set_ylim(minimo, maximo)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(
        f"{spec_base['display']} vs {spec_comp['display']}",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel(
        "LiDAR-derived reference canopy height (m)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Predicted canopy height (m)",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(True, linestyle="--", alpha=0.30)
    ax.legend(
        loc="upper left",
        fontsize=9,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
    )

    fig.text(
        0.5,
        0.015,
        (
            "Points are spatial out-of-fold predictions averaged across "
            "three seeds within each fold. Legend metrics are mean ± SD "
            "across the five spatial folds for the checkpoints selected "
            "during the original training runs."
        ),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1))

    nome = (
        f"{base_key}_vs_{comparison_key}_"
        "saved_selected_checkpoint_oof.png"
    )
    caminho = output_dir / nome
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return caminho


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera comparações pareadas usando somente as predições "
            "salvas durante os treinamentos."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"<PROJECT_ROOT>"),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Pasta TCCT_v3_2_SIBGRAPI. Quando omitida, usa "
            "<project-root>/Resultados/TCCT_v3_2_SIBGRAPI."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = args.project_root.expanduser().resolve()
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else (
            project_root
            / "Resultados"
            / "TCCT_v3_2_SIBGRAPI"
        ).resolve()
    )

    multiseed_dir = results_dir / "multiseed"
    resumos_dir = results_dir / "resumos_multiseed"

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            results_dir
            / "figuras_pareadas_predicoes_salvas"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("PAIRWISE FIGURES FROM SAVED TRAINING PREDICTIONS")
    print("=" * 88)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print()

    execucoes = carregar_execucoes(multiseed_dir)
    auditoria = validar_alinhamento(execucoes)

    auditoria.to_csv(
        output_dir / "audit_saved_prediction_alignment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dados_oof: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    resumos: dict[str, dict[str, float]] = {}
    linhas_metricas = []

    for model_key, runs in execucoes.items():
        y_true, y_pred, metricas_folds = construir_oof_seed_mean(runs)
        dados_oof[model_key] = (y_true, y_pred)
        resumos[model_key] = carregar_resumo_oficial(
            resumos_dir,
            model_key,
        )

        metricas_folds.insert(
            0,
            "model",
            MODEL_SPECS[model_key]["display"],
        )
        metricas_folds.insert(0, "model_key", model_key)
        linhas_metricas.append(metricas_folds)

        pooled = metricas(y_true, y_pred)
        print(
            f"{MODEL_SPECS[model_key]['display']}: "
            f"pooled seed-mean RMSE={pooled['rmse']:.6f} m, "
            f"R²={pooled['r2']:.6f}"
        )

    pd.concat(linhas_metricas, ignore_index=True).to_csv(
        output_dir / "metrics_seed_mean_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    linhas_resumo = []
    for model_key, (y_true, y_pred) in dados_oof.items():
        pooled = metricas(y_true, y_pred)
        official = resumos[model_key]

        linhas_resumo.append({
            "model_key": model_key,
            "model": MODEL_SPECS[model_key]["display"],
            "pooled_seed_mean_rmse": pooled["rmse"],
            "pooled_seed_mean_r2": pooled["r2"],
            "pooled_seed_mean_bias": pooled["bias"],
            "pooled_seed_mean_std_pred": pooled["std_pred"],
            "pooled_seed_mean_std_true": pooled["std_true"],
            "pooled_seed_mean_std_ratio": pooled["std_ratio"],
            "official_fold_mean_rmse": official["rmse_mean"],
            "official_fold_sd_rmse": official["rmse_sd"],
            "official_fold_mean_r2": official["r2_mean"],
            "official_fold_sd_r2": official["r2_sd"],
            "official_fold_mean_bias": official["bias_mean"],
            "official_fold_sd_bias": official["bias_sd"],
        })

    pd.DataFrame(linhas_resumo).to_csv(
        output_dir / "pooled_vs_official_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    phase_rows = []
    for model_key, runs in execucoes.items():
        for run in runs:
            phase_rows.append({
                "model_key": model_key,
                "model": MODEL_SPECS[model_key]["display"],
                "seed": run.seed,
                "fold": run.fold,
                "selected_phase": run.selected_phase,
                "source_file": str(run.source_file),
            })

    pd.DataFrame(phase_rows).to_csv(
        output_dir / "selected_phase_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure_paths = []
    for base_key, comparison_key in COMPARISONS:
        caminho = gerar_figura(
            output_dir=output_dir,
            base_key=base_key,
            comparison_key=comparison_key,
            dados_oof=dados_oof,
            resumos=resumos,
        )
        figure_paths.append(caminho)
        print(f"✓ Figure: {caminho.name}")

    run_summary = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "models": list(MODEL_SPECS),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "figures": [str(path) for path in figure_paths],
        "methodological_scope": (
            "Saved selected-checkpoint predictions from the original "
            "training runs. Four neural runs selected Phase B; therefore "
            "the figures are not a strictly Phase-A-only comparison."
        ),
        "no_retraining": True,
        "no_inference": True,
        "no_fold_reconstruction": True,
    }

    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("✓ No retraining, inference, or fold reconstruction was performed.")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("\n" + "!" * 88, file=sys.stderr)
        print(
            f"ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        print("!" * 88, file=sys.stderr)
        raise
