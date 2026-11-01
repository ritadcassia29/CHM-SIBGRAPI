#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Coleta artefatos leves das execuções multiseed do TCCT.

Objetivo
--------
Criar um pacote único para auditoria, sem copiar checkpoints .pth/.pt nem
modelos Random Forest .joblib.

O pacote reúne:
  1. manifestos_folds;
  2. resumos_multiseed;
  3. todos os arquivos .npz e .json existentes em multiseed;
  4. arquivos pequenos necessários para interpretar cada execução:
       - predicoes_random_forest.csv
       - comparacao_fases.csv
       - historico_completo.csv
  5. inventário com SHA-256;
  6. relatório de combinações esperadas e ausentes;
  7. um arquivo ZIP pronto para envio.

Uso no Prompt de Comando
------------------------
<LOCAL_PATH> ^
"<PROJECT_ROOT>\Codigo\Coletar_Artefatos_Multiseed_TCCT.py" ^
--project-root "<PROJECT_ROOT>"

Uso no Jupyter
--------------
import subprocess

subprocess.run(
    [
        r"<LOCAL_PATH>",
        r"<PROJECT_ROOT>\Codigo\Coletar_Artefatos_Multiseed_TCCT.py",
        "--project-root",
        r"<PROJECT_ROOT>",
    ],
    check=True,
)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_RESULT_RELATIVE = Path("Resultados") / "TCCT_v3_2_SIBGRAPI"

SEEDS_ESPERADAS = (42, 314, 2026)
FOLDS_ESPERADOS = tuple(range(5))

CONFIGURACOES_NEURAIS = (
    ("tcct", "completo"),
    ("tcct", "amplitude_only"),
    ("tcct_sem_coatencao", "completo"),
    ("cnn", "completo"),
    ("vit", "completo"),
)

ARQUIVOS_CSV_ADICIONAIS = {
    "predicoes_random_forest.csv",
    "comparacao_fases.csv",
    "historico_completo.csv",
}

PARTES_IGNORADAS = {
    "__pycache__",
    ".ipynb_checkpoints",
}

ARQUIVOS_CORE_NEURAIS = (
    "predicoes_patch_final.npz",
    "resumo_treinamento.json",
)

ARQUIVOS_CORE_RF = (
    "predicoes_random_forest.csv",
    "metricas_random_forest.json",
)


def sha256_arquivo(caminho: Path, bloco: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with caminho.open("rb") as arquivo:
        while True:
            dados = arquivo.read(bloco)
            if not dados:
                break
            digest.update(dados)
    return digest.hexdigest()


def caminho_tem_parte_ignorada(caminho: Path) -> bool:
    return any(parte in PARTES_IGNORADAS or parte.startswith("._")
               for parte in caminho.parts)


def deve_coletar_multiseed(caminho: Path) -> bool:
    if not caminho.is_file() or caminho_tem_parte_ignorada(caminho):
        return False

    nome = caminho.name
    sufixo = caminho.suffix.lower()

    if nome.startswith("._"):
        return False

    if sufixo in {".npz", ".json"}:
        return True

    return nome in ARQUIVOS_CSV_ADICIONAIS


def extrair_metadados_relativos(relativo: Path) -> dict[str, Any]:
    texto = relativo.as_posix()

    seed_match = re.search(r"(?:^|/)seed_(\d+)(?:/|$)", texto)
    fold_match = re.search(r"(?:^|/)fold_(\d+)(?:/|$)", texto)

    seed = int(seed_match.group(1)) if seed_match else None
    fold = int(fold_match.group(1)) if fold_match else None

    modelo = None
    ablation = None

    partes = relativo.parts
    try:
        indice_fold = next(
            i for i, parte in enumerate(partes)
            if re.fullmatch(r"fold_\d+", parte)
        )
        posteriores = partes[indice_fold + 1:]

        if posteriores:
            modelo = posteriores[0]

        if modelo != "random_forest" and len(posteriores) >= 2:
            ablation = posteriores[1]
        elif modelo == "random_forest":
            ablation = "completo"
    except StopIteration:
        pass

    return {
        "seed": seed,
        "fold": fold,
        "modelo": modelo,
        "ablation": ablation,
    }


def copiar_arquivo(
    origem: Path,
    destino: Path,
    raiz_resultados: Path,
    categoria: str,
) -> dict[str, Any]:
    relativo = origem.relative_to(raiz_resultados)
    destino_arquivo = destino / relativo
    destino_arquivo.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino_arquivo)

    metadados = extrair_metadados_relativos(relativo)

    return {
        "categoria": categoria,
        "origem_relativa_resultados": relativo.as_posix(),
        "destino_relativo_pacote": relativo.as_posix(),
        "nome": origem.name,
        "extensao": origem.suffix.lower(),
        "tamanho_bytes": int(origem.stat().st_size),
        "sha256": sha256_arquivo(destino_arquivo),
        **metadados,
    }


def coletar_arvore_completa(
    origem_dir: Path,
    destino: Path,
    raiz_resultados: Path,
    categoria: str,
    inventario: list[dict[str, Any]],
    limite_bytes: int,
) -> None:
    if not origem_dir.exists():
        print(f"⚠️ Pasta não encontrada: {origem_dir}")
        return

    for origem in sorted(origem_dir.rglob("*")):
        if not origem.is_file() or caminho_tem_parte_ignorada(origem):
            continue
        if origem.stat().st_size > limite_bytes:
            print(
                f"⚠️ Ignorado por tamanho ({origem.stat().st_size / 1024**2:.1f} MB): "
                f"{origem}"
            )
            continue
        inventario.append(
            copiar_arquivo(
                origem=origem,
                destino=destino,
                raiz_resultados=raiz_resultados,
                categoria=categoria,
            )
        )


def coletar_multiseed(
    multiseed_dir: Path,
    destino: Path,
    raiz_resultados: Path,
    inventario: list[dict[str, Any]],
    limite_bytes: int,
) -> None:
    if not multiseed_dir.exists():
        raise FileNotFoundError(
            f"Pasta multiseed não encontrada: {multiseed_dir}"
        )

    encontrados = 0

    for origem in sorted(multiseed_dir.rglob("*")):
        if not deve_coletar_multiseed(origem):
            continue

        tamanho = origem.stat().st_size
        if tamanho > limite_bytes:
            print(
                f"⚠️ Ignorado por tamanho ({tamanho / 1024**2:.1f} MB): {origem}"
            )
            continue

        inventario.append(
            copiar_arquivo(
                origem=origem,
                destino=destino,
                raiz_resultados=raiz_resultados,
                categoria="multiseed_execucao",
            )
        )
        encontrados += 1

    print(f"✓ Artefatos encontrados em multiseed: {encontrados}")


def verificar_combinacoes_core(multiseed_dir: Path) -> list[dict[str, Any]]:
    linhas: list[dict[str, Any]] = []

    for seed in SEEDS_ESPERADAS:
        for fold in FOLDS_ESPERADOS:
            for modelo, ablation in CONFIGURACOES_NEURAIS:
                pasta = (
                    multiseed_dir
                    / f"seed_{seed}"
                    / f"fold_{fold}"
                    / modelo
                    / ablation
                )

                presentes = {
                    nome: (pasta / nome).exists()
                    for nome in ARQUIVOS_CORE_NEURAIS
                }

                ausentes = [
                    nome for nome, existe in presentes.items()
                    if not existe
                ]

                linhas.append({
                    "tipo": "neural",
                    "seed": seed,
                    "fold": fold,
                    "modelo": modelo,
                    "ablation": ablation,
                    "pasta_relativa": pasta.relative_to(
                        multiseed_dir.parent
                    ).as_posix(),
                    "completo": len(ausentes) == 0,
                    "arquivos_ausentes": "; ".join(ausentes),
                })

            pasta_rf = (
                multiseed_dir
                / f"seed_{seed}"
                / f"fold_{fold}"
                / "random_forest"
            )

            presentes_rf = {
                nome: (pasta_rf / nome).exists()
                for nome in ARQUIVOS_CORE_RF
            }

            ausentes_rf = [
                nome for nome, existe in presentes_rf.items()
                if not existe
            ]

            linhas.append({
                "tipo": "random_forest",
                "seed": seed,
                "fold": fold,
                "modelo": "random_forest",
                "ablation": "completo",
                "pasta_relativa": pasta_rf.relative_to(
                    multiseed_dir.parent
                ).as_posix(),
                "completo": len(ausentes_rf) == 0,
                "arquivos_ausentes": "; ".join(ausentes_rf),
            })

    return linhas


def escrever_csv(
    caminho: Path,
    linhas: Iterable[dict[str, Any]],
    colunas: list[str] | None = None,
) -> None:
    linhas = list(linhas)

    if not linhas:
        caminho.write_text("", encoding="utf-8")
        return

    if colunas is None:
        colunas = list(linhas[0].keys())

    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=colunas,
            extrasaction="ignore",
        )
        escritor.writeheader()
        escritor.writerows(linhas)


def criar_zip(pasta: Path, caminho_zip: Path) -> None:
    with zipfile.ZipFile(
        caminho_zip,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as arquivo_zip:
        for arquivo in sorted(pasta.rglob("*")):
            if not arquivo.is_file():
                continue
            if arquivo.resolve() == caminho_zip.resolve():
                continue
            arquivo_zip.write(
                arquivo,
                arcname=arquivo.relative_to(pasta.parent).as_posix(),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coleta manifestos, resumos e artefatos leves das execuções "
            "multiseed do TCCT."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"<PROJECT_ROOT>"),
        help="Raiz do projeto SIBGRAPI_REV.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Pasta de resultados. Quando omitida, usa "
            "<project-root>/Resultados/TCCT_v3_2_SIBGRAPI."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Pasta de saída. Quando omitida, cria uma pasta com data/hora "
            "em <project-root>/Auditoria."
        ),
    )

    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=512.0,
        help="Tamanho máximo de cada arquivo a copiar. Padrão: 512 MB.",
    )

    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Não cria o ZIP final.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    projeto = args.project_root.expanduser().resolve()
    resultados = (
        args.results_dir.expanduser().resolve()
        if args.results_dir is not None
        else (projeto / DEFAULT_RESULT_RELATIVE).resolve()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saida = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            projeto
            / "Auditoria"
            / f"Pacote_Artefatos_Multiseed_{timestamp}"
        ).resolve()
    )

    multiseed_dir = resultados / "multiseed"
    manifestos_dir = resultados / "manifestos_folds"
    resumos_dir = resultados / "resumos_multiseed"

    print("=" * 88)
    print("COLETA DE ARTEFATOS MULTISEED — TCCT")
    print("=" * 88)
    print(f"Projeto:    {projeto}")
    print(f"Resultados: {resultados}")
    print(f"Saída:      {saida}")
    print()

    if not resultados.exists():
        raise FileNotFoundError(
            f"Pasta de resultados não encontrada: {resultados}"
        )

    if saida.exists():
        raise FileExistsError(
            f"A pasta de saída já existe: {saida}\n"
            "Escolha outra com --output-dir ou remova a pasta anterior."
        )

    saida.mkdir(parents=True, exist_ok=False)

    inventario: list[dict[str, Any]] = []
    limite_bytes = int(args.max_file_mb * 1024 * 1024)

    coletar_arvore_completa(
        origem_dir=manifestos_dir,
        destino=saida,
        raiz_resultados=resultados,
        categoria="manifestos_folds",
        inventario=inventario,
        limite_bytes=limite_bytes,
    )

    coletar_arvore_completa(
        origem_dir=resumos_dir,
        destino=saida,
        raiz_resultados=resultados,
        categoria="resumos_multiseed",
        inventario=inventario,
        limite_bytes=limite_bytes,
    )

    coletar_multiseed(
        multiseed_dir=multiseed_dir,
        destino=saida,
        raiz_resultados=resultados,
        inventario=inventario,
        limite_bytes=limite_bytes,
    )

    verificacao = verificar_combinacoes_core(multiseed_dir)

    caminho_inventario = saida / "inventario_artefatos.csv"
    escrever_csv(
        caminho_inventario,
        inventario,
        colunas=[
            "categoria",
            "origem_relativa_resultados",
            "destino_relativo_pacote",
            "nome",
            "extensao",
            "tamanho_bytes",
            "sha256",
            "seed",
            "fold",
            "modelo",
            "ablation",
        ],
    )

    caminho_verificacao = saida / "verificacao_combinacoes_core.csv"
    escrever_csv(caminho_verificacao, verificacao)

    completos = sum(bool(linha["completo"]) for linha in verificacao)
    incompletos = len(verificacao) - completos

    contagem_por_categoria: dict[str, int] = {}
    tamanho_total = 0

    for item in inventario:
        categoria = str(item["categoria"])
        contagem_por_categoria[categoria] = (
            contagem_por_categoria.get(categoria, 0) + 1
        )
        tamanho_total += int(item["tamanho_bytes"])

    resumo = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(projeto),
        "results_dir": str(resultados),
        "output_dir": str(saida),
        "collection_rules": {
            "multiseed_extensions": [".npz", ".json"],
            "additional_csv_names": sorted(ARQUIVOS_CSV_ADICIONAIS),
            "excluded_model_files": [".pth", ".pt", ".joblib"],
            "max_file_mb": float(args.max_file_mb),
        },
        "expected_design": {
            "seeds": list(SEEDS_ESPERADAS),
            "folds": list(FOLDS_ESPERADOS),
            "neural_configurations": [
                {"modelo": modelo, "ablation": ablation}
                for modelo, ablation in CONFIGURACOES_NEURAIS
            ],
            "random_forest": True,
            "expected_core_combinations": len(verificacao),
        },
        "collection_summary": {
            "files_copied": len(inventario),
            "bytes_copied": tamanho_total,
            "megabytes_copied": tamanho_total / 1024**2,
            "count_by_category": contagem_por_categoria,
            "core_combinations_complete": completos,
            "core_combinations_incomplete": incompletos,
        },
    }

    (saida / "resumo_coleta.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (saida / "LEIA-ME.txt").write_text(
        "\n".join([
            "PACOTE DE ARTEFATOS MULTISEED — TCCT",
            "",
            "Este pacote contém apenas artefatos leves usados na auditoria:",
            "- manifestos_folds;",
            "- resumos_multiseed;",
            "- arquivos .npz e .json das execuções multiseed;",
            "- predicoes_random_forest.csv;",
            "- comparacao_fases.csv;",
            "- historico_completo.csv.",
            "",
            "Não contém checkpoints .pth/.pt nem modelos .joblib.",
            "",
            "Consulte:",
            "- inventario_artefatos.csv;",
            "- verificacao_combinacoes_core.csv;",
            "- resumo_coleta.json.",
            "",
        ]),
        encoding="utf-8",
    )

    print()
    print(f"✓ Arquivos copiados: {len(inventario)}")
    print(f"✓ Volume: {tamanho_total / 1024**2:.2f} MB")
    print(f"✓ Combinações core completas: {completos}/{len(verificacao)}")

    if incompletos:
        print(
            f"⚠️ Combinações incompletas: {incompletos}. "
            f"Consulte {caminho_verificacao}"
        )

    if not args.no_zip:
        caminho_zip = saida.with_suffix(".zip")
        criar_zip(saida, caminho_zip)
        print(f"✓ ZIP criado: {caminho_zip}")
    else:
        caminho_zip = None

    print(f"✓ Pasta criada: {saida}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as erro:
        print("\n" + "!" * 88, file=sys.stderr)
        print(
            f"ERRO: {type(erro).__name__}: {erro}",
            file=sys.stderr,
        )
        print("!" * 88, file=sys.stderr)
        raise
