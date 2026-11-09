#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preparar_Pacote_Reprodutibilidade_TCCT.py
=========================================

Localiza, valida, inventaria e opcionalmente copia os artefatos necessários
para a auditoria/reprodutibilidade do TCCT (SIBGRAPI 2026, Paper 142).

O script NÃO refaz treinamentos e NÃO recalcula métricas. Ele trabalha apenas
com os artefatos já gravados pelo pipeline:

- 5 folds espaciais x 3 seeds;
- TCCT completo;
- TCCT amplitude-only;
- TCCT sem co-attention;
- CNN early fusion;
- Vision Transformer;
- Random Forest;
- manifestos dos folds e hashes;
- históricos, resumos, predições OOF e checkpoints;
- produtos da análise AGB support-matched;
- tabelas e figuras derivadas das predições salvas;
- código, configuração e informações do ambiente Python.

Modos de cópia
--------------
inventory:
    Apenas localiza, verifica e calcula hashes. Não duplica os artefatos.
evidence:
    Copia código, manifests, CSV/JSON/NPZ, logs, tabelas e figuras leves.
    Checkpoints .pth/.pt e modelos .joblib ficam apenas no inventário.
full:
    Copia também checkpoints e modelos, respeitando --max-file-mb.

Exemplos
--------
Windows:
    <LOCAL_PATH> Preparar_Pacote_Reprodutibilidade_TCCT.py ^
      --project-root "<LOCAL_PATH>" ^
      --copy-mode evidence ^
      --zip

Pacote completo, incluindo checkpoints:
    <LOCAL_PATH> Preparar_Pacote_Reprodutibilidade_TCCT.py ^
      --project-root "<LOCAL_PATH>" ^
      --copy-mode full ^
      --max-file-mb 4096 ^
      --zip

macOS:
    python3 Preparar_Pacote_Reprodutibilidade_TCCT.py \
      --project-root "<PROJECT_ROOT>" \
      --copy-mode evidence \
      --zip

Saídas
------
<project-root>/Auditoria/Reprodutibilidade_TCCT_<data_hora>/

- inventario_geral.csv
- verificacao_execucoes_5fold_3seed.csv
- verificacao_manifestos.csv
- verificacao_resumos_multiseed.csv
- verificacao_produtos_agb.csv
- ausencias_criticas.csv
- checksums.sha256
- resumo_auditoria.json
- ambiente_python.txt
- pip_freeze.txt
- README_PACOTE.txt
- artefatos/                 (nos modos evidence/full)
- <nome_da_pasta>.zip       (com --zip)

Requisitos
----------
Somente a biblioteca padrão do Python 3.10+.\nCompatível com volumes externos do macOS: a cópia não exige preservar permissões ou atributos estendidos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence


SCRIPT_VERSION = "1.0.1-macos"
DEFAULT_RESULTS_RELATIVE = Path("Resultados") / "TCCT_v3_2_SIBGRAPI"
EXPECTED_SEEDS = (42, 314, 2026)
EXPECTED_FOLDS = tuple(range(5))

# Estrutura gravada pelo pipeline v3.2.1.
NEURAL_CONFIGS = (
    ("tcct", "completo", "tcct_completo"),
    ("tcct", "amplitude_only", "tcct_amplitude_only"),
    ("tcct_sem_coatencao", "completo", "tcct_sem_coatencao_completo"),
    ("cnn", "completo", "cnn_completo"),
    ("vit", "completo", "vit_completo"),
)
RF_CONFIG = ("random_forest", "completo", "random_forest_completo")

NEURAL_REQUIRED = (
    "predicoes_patch_final.npz",
    "resumo_treinamento.json",
    "historico_completo.csv",
    "melhor_modelo_final.pth",
)
NEURAL_EXPECTED_IF_APPLICABLE = (
    "comparacao_fases.csv",
    "config_execucao.json",
    "xai_final.csv",
    "pesos_gating_modalidade_final.csv",
)
RF_REQUIRED = (
    "predicoes_random_forest.csv",
    "metricas_random_forest.json",
    "random_forest.joblib",
)

MANIFEST_REQUIRED = tuple(
    item
    for fold in EXPECTED_FOLDS
    for item in (
        f"manifesto_fold_{fold}.csv",
        f"manifesto_fold_{fold}.sha256.txt",
    )
) + (
    "resumo_manifestos_folds.csv",
    "index_dataset.sha256.txt",
)

SUMMARY_PREFIXES = tuple(cfg[2] for cfg in NEURAL_CONFIGS) + (RF_CONFIG[2],)
SUMMARY_PATTERNS = tuple(
    f"{prefix_name}_{base}.csv"
    for base in SUMMARY_PREFIXES
    for prefix_name in (
        "resultados_detalhados",
        "resumo_por_fold",
        "resumo_final",
    )
)

AGB_ALLOMETRY_REQUIRED = (
    "inventory_50m_support_matched.csv",
    "lopo_coefficients.csv",
    "lopo_agb_predictions.csv",
    "allometry_coefficients_and_metrics.json",
)

AGB_ALL_MODELS_REQUIRED = (
    "selected_phase_manifest.csv",
    "inventory_patch_overlap_links.csv",
    "agb_predictions_all_models_all_seeds.csv",
    "agb_metrics_per_seed_fold.csv",
    "agb_metrics_seed_mean_by_fold.csv",
    "agb_metrics_spatial_summary.csv",
    "agb_metrics_pooled_oof_seed_mean.csv",
    "inventory_support_and_fold_assignment.csv",
    "agb_field_reference_baseline_and_models.csv",
    "Table_I_CHM_AGB_all_models_numeric.csv",
    "Table_I_CHM_AGB_all_models_formatted.csv",
    "Table_I_CHM_AGB_all_models_full.tex",
    "Table_I_CHM_AGB_all_models_compact.tex",
    "run_summary.json",
)

PAIRWISE_REQUIRED = (
    "audit_saved_prediction_alignment.csv",
    "metrics_seed_mean_by_fold.csv",
    "pooled_vs_official_metrics.csv",
    "selected_phase_manifest.csv",
    "run_summary.json",
)

OOF_SUPPORT_EXPECTED = (
    "run_summary.json",
)

TEXT_LIKE_SUFFIXES = {
    ".csv", ".json", ".txt", ".md", ".tex", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".log", ".sha256", ".ipynb", ".py", ".r", ".m",
}
PREDICTION_SUFFIXES = {".npz", ".npy", ".parquet", ".feather", ".pkl"}
FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps", ".tif", ".tiff"}
MODEL_SUFFIXES = {".pth", ".pt", ".joblib", ".onnx", ".ckpt", ".safetensors"}
ENVIRONMENT_NAMES = {
    "requirements.txt", "requirements-dev.txt", "environment.yml",
    "environment.yaml", "pyproject.toml", "poetry.lock", "pdm.lock",
    "uv.lock", "conda-lock.yml", "conda-lock.yaml", "pipfile",
    "pipfile.lock",
}
IGNORED_PARTS = {
    "__pycache__", ".git", ".svn", ".hg", ".idea", ".vscode",
    ".ipynb_checkpoints", "__MACOSX", "node_modules",
}
RAW_DATA_HINTS = {
    "dados_brutos", "raw_data", "raw", "tropisar", "landsat",
    "lidar_chm", "dados_processados",
}


@dataclass
class InventoryRow:
    category: str
    path_relative_project: str
    path_absolute: str
    filename: str
    extension: str
    size_bytes: int
    size_mb: float
    modified_utc: str
    sha256: str
    seed: int | None
    fold: int | None
    model: str | None
    ablation: str | None
    copied: bool
    copied_relative_package: str
    note: str


@dataclass
class CheckRow:
    group: str
    seed: int | None
    fold: int | None
    model: str | None
    ablation: str | None
    expected_path: str
    required: bool
    present: bool
    missing_files: str
    note: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalized_parts(path: Path) -> set[str]:
    return {part.lower() for part in path.parts}


def ignored(path: Path) -> bool:
    return any(
        part in IGNORED_PARTS or part.startswith("._")
        for part in path.parts
    )


def infer_run_metadata(path: Path) -> tuple[int | None, int | None, str | None, str | None]:
    text = path.as_posix()
    seed_match = re.search(r"(?:^|/)seed_(\d+)(?:/|$)", text, flags=re.I)
    fold_match = re.search(r"(?:^|/)fold_(\d+)(?:/|$)", text, flags=re.I)
    seed = int(seed_match.group(1)) if seed_match else None
    fold = int(fold_match.group(1)) if fold_match else None

    model = None
    ablation = None
    parts = path.parts
    for idx, part in enumerate(parts):
        if re.fullmatch(r"fold_\d+", part, flags=re.I):
            after = parts[idx + 1:]
            if after:
                model = after[0]
            if model and model.lower() == "random_forest":
                ablation = "completo"
            elif len(after) >= 2:
                ablation = after[1]
            break
    return seed, fold, model, ablation


def classify_file(path: Path, results_dir: Path, code_dirs: Sequence[Path]) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()

    try:
        rel_results = path.relative_to(results_dir)
        parts = normalized_parts(rel_results)
        if "manifestos_folds" in parts:
            return "fold_manifest"
        if "resumos_multiseed" in parts:
            return "multiseed_summary"
        if "agb_allometry_supportmatched" in parts:
            return "agb_allometry"
        if "agb_all_methods_saved_oof" in parts:
            return "agb_all_models"
        if "oof_chm_agb_supportmatched" in parts:
            return "oof_map"
        if "figuras_pareadas_predicoes_salvas" in parts:
            return "pairwise_figures"
        if "multiseed" in parts:
            if suffix in MODEL_SUFFIXES:
                return "checkpoint_or_model"
            if "predic" in name or suffix in PREDICTION_SUFFIXES:
                return "saved_prediction"
            if "historico" in name or suffix == ".log":
                return "training_log"
            return "multiseed_run"
        if suffix in FIGURE_SUFFIXES:
            return "result_figure"
        return "other_result"
    except ValueError:
        pass

    for code_dir in code_dirs:
        try:
            path.relative_to(code_dir)
            if suffix in {".py", ".ipynb"}:
                return "source_code"
            if name in ENVIRONMENT_NAMES:
                return "environment"
            return "project_documentation"
        except ValueError:
            continue

    if name in ENVIRONMENT_NAMES:
        return "environment"
    if suffix in {".py", ".ipynb"}:
        return "source_code"
    return "project_documentation"


def choose_copy(
    path: Path,
    category: str,
    copy_mode: str,
    max_bytes: int,
) -> tuple[bool, str]:
    if copy_mode == "inventory":
        return False, "inventory mode"
    if path.stat().st_size > max_bytes:
        return False, f"larger than max-file-mb ({max_bytes / 1024**2:.1f} MB)"
    if category == "checkpoint_or_model" and copy_mode != "full":
        return False, "checkpoint/model only inventoried in evidence mode"

    suffix = path.suffix.lower()
    if copy_mode == "evidence":
        allowed = (
            suffix in TEXT_LIKE_SUFFIXES
            or suffix in PREDICTION_SUFFIXES
            or suffix in FIGURE_SUFFIXES
            or path.name.lower() in ENVIRONMENT_NAMES
        )
        if not allowed:
            return False, "file type excluded from evidence mode"
    return True, ""


def safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        drive = re.sub(r"[^A-Za-z0-9_-]+", "_", path.drive or "external")
        return Path("_external") / drive / Path(*path.parts[1:])


def copy_file_preserving_tree(source: Path, package_artifacts: Path, project_root: Path) -> Path:
    """Copia o conteúdo sem depender de metadados incompatíveis do volume.

    Em volumes externos do macOS (especialmente exFAT/NTFS), shutil.copy2 pode
    copiar os bytes corretamente e falhar depois ao tentar preservar permissões,
    datas ou atributos estendidos. Para a auditoria, o conteúdo e o SHA-256 são
    o que importam. Por isso usamos copyfile e tratamos metadados como opcionais.
    """
    relative = safe_relative(source, project_root)
    destination = package_artifacts / relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(source, destination)
    except PermissionError:
        # Remove um destino parcial, caso tenha sido criado antes da falha.
        try:
            if destination.exists():
                destination.unlink()
        except OSError:
            pass
        raise

    # Preservar timestamp é útil, mas não deve interromper a coleta.
    try:
        stat = source.stat()
        os.utime(destination, (stat.st_atime, stat.st_mtime))
    except (OSError, PermissionError):
        pass

    return destination.relative_to(package_artifacts.parent)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def discover_code_dirs(project_root: Path, explicit: Sequence[Path]) -> list[Path]:
    result: list[Path] = []
    for path in explicit:
        resolved = path.expanduser().resolve()
        if resolved.exists() and resolved.is_dir():
            result.append(resolved)

    for name in ("Codigo", "Código", "code", "scripts"):
        candidate = project_root / name
        if candidate.exists() and candidate.is_dir():
            result.append(candidate.resolve())

    # Os arquivos de código podem estar diretamente na raiz.
    result.append(project_root.resolve())

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in result:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def iter_relevant_project_files(
    project_root: Path,
    results_dir: Path,
    code_dirs: Sequence[Path],
    output_dir: Path,
) -> Iterator[Path]:
    roots: list[Path] = [results_dir]
    roots.extend(code_dirs)

    yielded: set[Path] = set()
    output_resolved = output_dir.resolve()

    for root in roots:
        if not root.exists():
            continue

        # Para a raiz do projeto, evita varrer os dados brutos inteiros:
        # coleta arquivos diretamente na raiz e as árvores explicitamente relevantes.
        if root.resolve() == project_root.resolve():
            candidates = (
                p for p in root.iterdir()
                if p.is_file()
            )
        else:
            candidates = root.rglob("*")

        for path in candidates:
            if not path.is_file() or ignored(path):
                continue
            try:
                path.resolve().relative_to(output_resolved)
                continue
            except ValueError:
                pass

            resolved = path.resolve()
            if resolved in yielded:
                continue
            yielded.add(resolved)
            yield resolved


def verify_file_set(
    group: str,
    base_dir: Path,
    filenames: Sequence[str],
    required: bool = True,
    seed: int | None = None,
    fold: int | None = None,
    model: str | None = None,
    ablation: str | None = None,
    note: str = "",
) -> CheckRow:
    missing = [name for name in filenames if not (base_dir / name).exists()]
    return CheckRow(
        group=group,
        seed=seed,
        fold=fold,
        model=model,
        ablation=ablation,
        expected_path=base_dir.as_posix(),
        required=required,
        present=not missing,
        missing_files="; ".join(missing),
        note=note,
    )


def verify_runs(results_dir: Path) -> list[CheckRow]:
    multiseed = results_dir / "multiseed"
    rows: list[CheckRow] = []

    for seed in EXPECTED_SEEDS:
        for fold in EXPECTED_FOLDS:
            for model, ablation, _ in NEURAL_CONFIGS:
                base = multiseed / f"seed_{seed}" / f"fold_{fold}" / model / ablation
                rows.append(
                    verify_file_set(
                        "neural_core",
                        base,
                        NEURAL_REQUIRED,
                        required=True,
                        seed=seed,
                        fold=fold,
                        model=model,
                        ablation=ablation,
                        note="Core needed to verify metrics and selected checkpoint.",
                    )
                )
                rows.append(
                    verify_file_set(
                        "neural_auxiliary",
                        base,
                        NEURAL_EXPECTED_IF_APPLICABLE,
                        required=False,
                        seed=seed,
                        fold=fold,
                        model=model,
                        ablation=ablation,
                        note="Auxiliary diagnostics; absence is reported but is not always critical.",
                    )
                )

            model, ablation, _ = RF_CONFIG
            base = multiseed / f"seed_{seed}" / f"fold_{fold}" / model
            rows.append(
                verify_file_set(
                    "random_forest_core",
                    base,
                    RF_REQUIRED,
                    required=True,
                    seed=seed,
                    fold=fold,
                    model=model,
                    ablation=ablation,
                )
            )
    return rows


def verify_global_products(results_dir: Path) -> tuple[list[CheckRow], list[CheckRow], list[CheckRow]]:
    manifest_rows = [
        verify_file_set(
            "fold_manifests",
            results_dir / "manifestos_folds",
            MANIFEST_REQUIRED,
            required=True,
            note="Five spatial fold manifests and their hashes.",
        )
    ]

    summary_rows = [
        verify_file_set(
            "multiseed_summaries",
            results_dir / "resumos_multiseed",
            SUMMARY_PATTERNS,
            required=True,
            note="Detailed, per-fold, and final hierarchical summaries for all models.",
        )
    ]

    agb_rows = [
        verify_file_set(
            "agb_allometry",
            results_dir / "AGB_Allometry_SupportMatched",
            AGB_ALLOMETRY_REQUIRED,
            required=True,
        ),
        verify_file_set(
            "agb_all_models",
            results_dir / "AGB_All_Methods_Saved_OOF",
            AGB_ALL_MODELS_REQUIRED,
            required=True,
        ),
        verify_file_set(
            "pairwise_saved_predictions",
            results_dir / "figuras_pareadas_predicoes_salvas",
            PAIRWISE_REQUIRED,
            required=False,
        ),
        verify_file_set(
            "oof_chm_agb_map",
            results_dir / "OOF_CHM_AGB_SupportMatched",
            OOF_SUPPORT_EXPECTED,
            required=False,
        ),
    ]
    return manifest_rows, summary_rows, agb_rows


def capture_environment(output_dir: Path) -> None:
    info = [
        f"generated_at_utc={utc_now_iso()}",
        f"script_version={SCRIPT_VERSION}",
        f"python_executable={sys.executable}",
        f"python_version={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"system={platform.system()}",
        f"machine={platform.machine()}",
        f"processor={platform.processor()}",
        f"cwd={Path.cwd()}",
    ]
    (output_dir / "ambiente_python.txt").write_text(
        "\n".join(info) + "\n",
        encoding="utf-8",
    )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        content = result.stdout
        if result.stderr:
            content += "\n# STDERR\n" + result.stderr
        (output_dir / "pip_freeze.txt").write_text(content, encoding="utf-8")
    except Exception as exc:
        (output_dir / "pip_freeze.txt").write_text(
            f"Could not execute pip freeze: {exc}\n",
            encoding="utf-8",
        )


def create_zip(folder: Path) -> Path:
    zip_path = folder.with_suffix(".zip")
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            archive.write(path, arcname=path.relative_to(folder.parent).as_posix())
    return zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate and package TCCT reproducibility artifacts."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root, e.g. <LOCAL_PATH>",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override results directory.",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional code directory. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override audit package output directory.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("inventory", "evidence", "full"),
        default="evidence",
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=2048.0,
        help="Maximum size of a single copied file.",
    )
    parser.add_argument("--zip", action="store_true", help="Create final ZIP.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when a required artifact is missing.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 calculation. Not recommended for the final package.",
    )
    return parser.parse_args()


def resolve_project_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("TCCT_PROJECT_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path(r"<PROJECT_ROOT>"),
            Path("<PROJECT_ROOT>"),
            Path.cwd(),
        ]
    )

    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    raise FileNotFoundError(
        "Project root not found. Use --project-root with the correct path."
    )


def main() -> int:
    args = parse_args()
    project_root = resolve_project_root(args.project_root)
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else (project_root / DEFAULT_RESULTS_RELATIVE).resolve()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (
            project_root
            / "Auditoria"
            / f"Reprodutibilidade_TCCT_{timestamp}"
        ).resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = output_dir / "artefatos"

    code_dirs = discover_code_dirs(project_root, args.code_dir)
    max_bytes = int(args.max_file_mb * 1024 * 1024)

    print("=" * 96)
    print("TCCT REPRODUCIBILITY ARTIFACT AUDIT")
    print("=" * 96)
    print(f"Project root: {project_root}")
    print(f"Results dir:  {results_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Copy mode:    {args.copy_mode}")
    print()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    run_checks = verify_runs(results_dir)
    manifest_checks, summary_checks, agb_checks = verify_global_products(results_dir)
    all_checks = run_checks + manifest_checks + summary_checks + agb_checks

    write_csv(
        output_dir / "verificacao_execucoes_5fold_3seed.csv",
        [asdict(row) for row in run_checks],
    )
    write_csv(
        output_dir / "verificacao_manifestos.csv",
        [asdict(row) for row in manifest_checks],
    )
    write_csv(
        output_dir / "verificacao_resumos_multiseed.csv",
        [asdict(row) for row in summary_checks],
    )
    write_csv(
        output_dir / "verificacao_produtos_agb.csv",
        [asdict(row) for row in agb_checks],
    )

    required_missing = [
        row for row in all_checks
        if row.required and not row.present
    ]
    write_csv(
        output_dir / "ausencias_criticas.csv",
        [asdict(row) for row in required_missing],
    )

    inventory: list[InventoryRow] = []
    checksums: list[tuple[str, str]] = []
    count = 0

    for path in iter_relevant_project_files(
        project_root=project_root,
        results_dir=results_dir,
        code_dirs=code_dirs,
        output_dir=output_dir,
    ):
        count += 1
        category = classify_file(path, results_dir, code_dirs)
        seed, fold, model, ablation = infer_run_metadata(path)
        should_copy, note = choose_copy(
            path=path,
            category=category,
            copy_mode=args.copy_mode,
            max_bytes=max_bytes,
        )

        digest = "" if args.no_hash else sha256_file(path)
        copied_relative = ""
        if should_copy:
            copied_path = copy_file_preserving_tree(
                source=path,
                package_artifacts=artifacts_dir,
                project_root=project_root,
            )
            copied_relative = copied_path.as_posix()

        stat = path.stat()
        inventory.append(
            InventoryRow(
                category=category,
                path_relative_project=safe_relative(path, project_root).as_posix(),
                path_absolute=str(path),
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=int(stat.st_size),
                size_mb=round(stat.st_size / 1024**2, 6),
                modified_utc=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                sha256=digest,
                seed=seed,
                fold=fold,
                model=model,
                ablation=ablation,
                copied=should_copy,
                copied_relative_package=copied_relative,
                note=note,
            )
        )
        if digest:
            checksums.append((digest, safe_relative(path, project_root).as_posix()))

        if count % 100 == 0:
            print(f"  inventoried: {count} files")

    write_csv(
        output_dir / "inventario_geral.csv",
        [asdict(row) for row in inventory],
    )
    (output_dir / "checksums.sha256").write_text(
        "\n".join(f"{digest}  {relative}" for digest, relative in checksums) + "\n",
        encoding="utf-8",
    )

    capture_environment(output_dir)

    required_total = sum(1 for row in all_checks if row.required)
    required_complete = sum(
        1 for row in all_checks if row.required and row.present
    )
    optional_missing = sum(
        1 for row in all_checks if not row.required and not row.present
    )

    category_counts: dict[str, int] = {}
    copied_count = 0
    copied_bytes = 0
    for row in inventory:
        category_counts[row.category] = category_counts.get(row.category, 0) + 1
        if row.copied:
            copied_count += 1
            copied_bytes += row.size_bytes

    summary = {
        "generated_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "project_root": str(project_root),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "copy_mode": args.copy_mode,
        "max_file_mb": args.max_file_mb,
        "expected_design": {
            "seeds": list(EXPECTED_SEEDS),
            "folds": list(EXPECTED_FOLDS),
            "neural_configs": [
                {"model": m, "ablation": a, "summary_base": b}
                for m, a, b in NEURAL_CONFIGS
            ],
            "random_forest": {
                "model": RF_CONFIG[0],
                "ablation": RF_CONFIG[1],
                "summary_base": RF_CONFIG[2],
            },
            "expected_neural_runs": len(EXPECTED_SEEDS) * len(EXPECTED_FOLDS) * len(NEURAL_CONFIGS),
            "expected_rf_runs": len(EXPECTED_SEEDS) * len(EXPECTED_FOLDS),
        },
        "verification": {
            "required_groups_total": required_total,
            "required_groups_complete": required_complete,
            "required_groups_missing": len(required_missing),
            "optional_groups_missing": optional_missing,
        },
        "inventory": {
            "files_found": len(inventory),
            "files_copied": copied_count,
            "copied_bytes": copied_bytes,
            "copied_gb": copied_bytes / 1024**3,
            "category_counts": category_counts,
        },
    }
    (output_dir / "resumo_auditoria.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme_lines = [
        "PACOTE DE REPRODUTIBILIDADE — TCCT / SIBGRAPI 2026 / PAPER 142",
        "",
        f"Generated at UTC: {summary['generated_at_utc']}",
        f"Project root: {project_root}",
        f"Results directory: {results_dir}",
        f"Copy mode: {args.copy_mode}",
        "",
        "DESIGN CHECKED",
        "- 5 spatial folds;",
        "- seeds 42, 314, and 2026;",
        "- complete TCCT;",
        "- amplitude-only TCCT;",
        "- TCCT without co-attention;",
        "- early-fusion CNN;",
        "- Vision Transformer;",
        "- Random Forest.",
        "",
        "CORE EVIDENCE",
        "- exact fold manifests and hashes;",
        "- per-seed/per-fold training summaries and histories;",
        "- saved selected-checkpoint patch predictions;",
        "- phase-comparison files;",
        "- checkpoints/model files (copied only in full mode);",
        "- hierarchical multiseed summaries;",
        "- support-matched allometry and end-to-end AGB tables;",
        "- paired figures generated from saved predictions;",
        "- source code and Python environment capture.",
        "",
        "IMPORTANT",
        "- Raw TropiSAR, Landsat, LiDAR, and inventory data are not automatically",
        "  duplicated by this package. Their availability remains subject to the",
        "  original distribution conditions.",
        "- Absolute paths are retained only in inventario_geral.csv for local audit.",
        "  Remove or anonymize that column before public release if necessary.",
        "- Check ausencias_criticas.csv. It must be empty before claiming that the",
        "  evidence package is complete.",
        "",
        f"Required groups complete: {required_complete}/{required_total}",
        f"Critical missing groups: {len(required_missing)}",
        f"Files inventoried: {len(inventory)}",
        f"Files copied: {copied_count}",
    ]
    (output_dir / "README_PACOTE.txt").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    zip_path = None
    if args.zip:
        print("Creating ZIP...")
        zip_path = create_zip(output_dir)

    print()
    print("=" * 96)
    print("AUDIT COMPLETE")
    print("=" * 96)
    print(f"Files inventoried:       {len(inventory)}")
    print(f"Required groups:         {required_complete}/{required_total} complete")
    print(f"Critical missing groups: {len(required_missing)}")
    print(f"Files copied:            {copied_count}")
    print(f"Package:                 {output_dir}")
    if zip_path:
        print(f"ZIP:                     {zip_path}")

    if required_missing:
        print("\nCRITICAL MISSING ITEMS:")
        for row in required_missing:
            print(
                f"- {row.group} | seed={row.seed} fold={row.fold} "
                f"model={row.model} ablation={row.ablation} | "
                f"{row.expected_path} | missing: {row.missing_files}"
            )

    return 2 if args.strict and required_missing else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
