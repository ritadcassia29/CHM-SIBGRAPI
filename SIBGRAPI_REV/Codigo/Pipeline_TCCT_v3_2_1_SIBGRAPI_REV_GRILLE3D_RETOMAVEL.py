
# ================================================================
# 0. IMPORTAÇÕES, CONFIGURAÇÃO E REPRODUTIBILIDADE
# ================================================================
from __future__ import annotations

import os
import re
import gc
import json
import math
import time
import glob
import random
import hashlib
import platform
import warnings
import traceback
import sys
import copy
import shutil
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Iterable, Sequence
from collections import Counter

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import xy as raster_xy
from rasterio.crs import CRS
from pyproj import Transformer, CRS as PyprojCRS, datadir
from pyproj.transformer import TransformerGroup
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

VERSAO_PIPELINE = "3.2.1-SIBGRAPI-REV-GRILLE3D-RETOMAVEL"
VERSAO_PATCH_APRESENTACAO = "FIGURES-EN-OOF-INDEX-FIX-2026-08-01"
VERSAO_DATASET = 6
POLARIZACOES = ("Hh", "Hv", "Vh", "Vv")

# Layouts binários determinados por tamanho do arquivo, cabeçalhos e .grille.
LAYOUTS_SLC_APROVADOS = {
    ("SAR_L", "tropi0506"): {"shape": (18243, 2600), "offset_bytes": 4},
    ("SAR_L", "tropi0603"): {"shape": (16361, 2600), "offset_bytes": 4},
    ("SAR_P", "tropi0506"): {"shape": (10734, 4000), "offset_bytes": 4},
    ("SAR_P", "tropi0603"): {"shape": (9601, 4000), "offset_bytes": 4},
}
LAYOUT_HA_P_APROVADO = {"shape": (7478, 4000), "dtype": "<f4", "offset_bytes": 0}
NOME_GRADE_GEOIDAL_GUYANA = "fr_ign_ggguy15.tif"
URL_GRADE_GEOIDAL_GUYANA = "https://cdn.proj.org/fr_ign_ggguy15.tif"


PASTA_PROJETO_MACOS = Path("<PROJECT_ROOT>")
VARIAVEL_AMBIENTE_RAIZ = "TCCT_SIBGRAPI_REV_DIR"



def _candidatos_windows_sibgrapi_rev() -> list[Path]:
    """Busca controlada da raiz isolada em unidades Windows."""
    candidatos: list[Path] = []
    usuario = os.environ.get("USERNAME", "")
    for letra in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raiz = Path(f"{letra}:/")
        try:
            if not raiz.exists():
                continue
        except OSError:
            continue
        candidatos.extend([
            raiz / "SIBGRAPI_REV",
            raiz / "PROJECT_VOLUME" / "SIBGRAPI_REV",
            raiz / "TCCT_Geral" / "SIBGRAPI_REV",
            raiz / "TCCT_Geral" / "Perplexity" / "SIBGRAPI_REV",
        ])
        if usuario:
            candidatos.extend([
                raiz / "Users" / usuario / "SIBGRAPI_REV",
                raiz / "Users" / usuario / "Desktop" / "SIBGRAPI_REV",
                raiz / "Users" / usuario / "Documents" / "SIBGRAPI_REV",
            ])
        for padrao in ("*/SIBGRAPI_REV", "*/*/SIBGRAPI_REV"):
            try:
                candidatos.extend(raiz.glob(padrao))
            except (OSError, PermissionError):
                pass
    return candidatos


def _candidatos_cwd_sibgrapi_rev() -> list[Path]:
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        cwd = Path.cwd()
    candidatos = [cwd, cwd / "SIBGRAPI_REV"]
    candidatos.extend(cwd.parents)
    candidatos.extend(p / "SIBGRAPI_REV" for p in cwd.parents)
    return candidatos


def _raiz_sibgrapi_valida(caminho: Path) -> bool:
    try:
        caminho = caminho.expanduser().resolve(strict=False)
        return (
            caminho.is_dir()
            and caminho.name == "SIBGRAPI_REV"
            and (caminho / "Codigo").is_dir()
            and (caminho / "Dados_Brutos").is_dir()
        )
    except OSError:
        return False


def localizar_base_dir(explicito: Optional[str] = None) -> Path:
    """Resolve a raiz SIBGRAPI_REV em macOS, Windows ou Linux.

    Caminho explícito e variável de ambiente têm prioridade. Na detecção
    automática, múltiplas cópias válidas provocam interrupção para impedir que
    resultados sejam gravados no projeto errado.
    """
    prioritarios: list[Path] = []
    if explicito:
        prioritarios.append(Path(explicito).expanduser())
    if os.getenv(VARIAVEL_AMBIENTE_RAIZ):
        prioritarios.append(
            Path(os.environ[VARIAVEL_AMBIENTE_RAIZ]).expanduser()
        )
    for caminho in prioritarios:
        if _raiz_sibgrapi_valida(caminho):
            return caminho.resolve()

    candidatos: list[Path] = []
    candidatos.extend(_candidatos_cwd_sibgrapi_rev())
    sistema = platform.system()
    if sistema == "Darwin":
        candidatos.extend([
            PASTA_PROJETO_MACOS,
            Path.home() / "SIBGRAPI_REV",
        ])
        try:
            candidatos.extend(Path("/Volumes").glob("*/SIBGRAPI_REV"))
        except (OSError, PermissionError):
            pass
    elif sistema == "Windows":
        candidatos.extend(_candidatos_windows_sibgrapi_rev())
    else:
        usuario = os.getenv("USER", "")
        candidatos.extend([
            Path("/media") / usuario / "PROJECT_VOLUME" / "SIBGRAPI_REV",
            Path("<LOCAL_PATH>"),
            Path.home() / "SIBGRAPI_REV",
        ])

    encontrados: list[Path] = []
    vistos = set()
    for caminho in candidatos:
        try:
            resolvido = caminho.expanduser().resolve(strict=False)
        except OSError:
            continue
        chave = str(resolvido)
        if chave in vistos:
            continue
        vistos.add(chave)
        if _raiz_sibgrapi_valida(resolvido):
            encontrados.append(resolvido)

    if len(encontrados) == 1:
        return encontrados[0]
    if len(encontrados) > 1:
        raise RuntimeError(
            "Múltiplas cópias de SIBGRAPI_REV foram encontradas:\n"
            + "\n".join(f"  - {p}" for p in encontrados)
            + "\nDefina TCCT_SIBGRAPI_REV_DIR explicitamente."
        )

    if prioritarios:
        return prioritarios[0]
    if sistema == "Darwin":
        return PASTA_PROJETO_MACOS
    return Path.cwd() / "SIBGRAPI_REV"


def detectar_device() -> tuple[torch.device, bool]:
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


@dataclass
class ConfigDados:
    base_dir: Optional[str] = None
    aquisicao_mestre: str = "tropi0506"
    aquisicao_escrava: str = "tropi0603"
    aquisicoes_ortho: tuple[str, ...] = ("tropi0506", "tropi0603")
    bandas: tuple[str, ...] = ("SAR_L", "SAR_P")
    tamanho_patch: int = 30
    janela_coerencia: int = 5
    tamanho_bloco_landsat: int = 8
    # Protocolo principal vigente: buffer=0. O buffer=1 permanece como análise
    # de sensibilidade espacial pré-declarada e não como ajuste pós-hoc.
    buffer_blocos: int = 0
    buffers_a_auditar: tuple[int, ...] = (0, 1)
    fracao_dominio_local: float = 0.20
    fracao_ajuste_local: float = 0.50
    fracao_validacao_local: float = 0.25
    buffer_finetuning_blocos: int = 1
    # Preparação reproduzível do CHM LiDAR.
    lidar_crs_epsg: int = 32622
    # Atribuição documental do CRS LiDAR sem modificar dados brutos.
    lidar_permitir_atribuicao_crs_documental: bool = True
    lidar_exigir_readme_para_crs: bool = True
    dtm_ncols_esperado: int = 2953
    dtm_nrows_esperado: int = 4369
    dtm_xllcorner_esperado: float = 284620.0
    dtm_yllcorner_esperado: float = 580502.0
    dtm_cellsize_esperado: float = 1.0
    dtm_nodata_esperado: float = -9999.0
    validar_cabecalho_dtm_estritamente: bool = True
    chm_tolerancia_negativa_m: float = 0.50
    chm_alerta_altura_m: float = 60.0
    salvar_dtm_alinhado: bool = True
    # Inventário integral e rastreabilidade do acervo original.
    inventario_hash_completo_todos: bool = False
    inventario_hash_completo_limite_mb: float = 64.0
    inventario_chunk_mb: int = 4

    # Composição Landsat-5 multitemporal por célula de 30 m.
    # A seleção usa somente qualidade externa ao alvo: QA, NoData, saturação,
    # fração local de pixels claros, nuvens globais da cena e proximidade temporal.
    landsat_modo: str = "mosaico_qualidade"
    landsat_janela_qualidade: int = 5
    landsat_exigir_qa_pixel: bool = True
    landsat_usar_qa_radsat: bool = True
    landsat_data_referencia_tropisar: Optional[str] = None
    landsat_fracao_clara_local_minima: float = 0.0
    landsat_max_cenas: int = 8
    landsat_salvar_stack_cenas_no_patch: bool = True
    fracao_minima_chm_valido: float = 0.80
    chm_minimo_medio: float = 2.0
    chm_maximo_valido: float = 60.0
    sobrescrever: bool = False
    # Back-geocoding científico com as grades tridimensionais ONERA (.grille).
    backgeo_chunk_linhas: int = 128
    backgeo_interpolacao_slc: str = "bilinear_complexa"
    backgeo_fracao_geometrica_minima_patch: float = 1.0
    backgeo_schema_version: int = 2
    # A decisão vertical é feita com o DTM e os pontos ROI, antes de qualquer
    # treino. Valores: "auto_roi", "direto" ou "ngg1977_para_elipsoidal".
    backgeo_modo_vertical_dtm: str = "auto_roi"
    backgeo_min_pontos_roi_dtm: int = 100
    backgeo_margem_decisao_vertical_fracao_geoide: float = 0.25
    backgeo_max_residuo_vertical_fracao_geoide: float = 0.50
    backgeo_salvar_mapas_linha_coluna: bool = True
    salvar_matrizes_polinsar: bool = True
    salvar_auditoria_detalhada: bool = True
    total_legacy_referencia: int = 13247
    # Nunca usar constantes arbitrárias. Preencha somente com metadados reais.
    kz_l_constante: Optional[float] = None
    kz_p_constante: Optional[float] = None
    incidencia_l_deg: Optional[float] = None
    incidencia_p_deg: Optional[float] = None
    # Padrões de busca para rasters/metadados, caso existam no volume externo.
    padroes_kz: tuple[str, ...] = ("*kz*.tif", "*k_z*.tif", "*vertical*wavenumber*.tif", "*kz*.npy")
    padroes_incidencia: tuple[str, ...] = ("*incidence*.tif", "*incidencia*.tif", "*look_angle*.tif", "*theta*.tif", "*incidence*.npy")

    # Geometria física do par definitivo 0506–0603.
    # A v2.7 deriva kz_P por diferença de dois produtos de altura de ambiguidade
    # gerados com a mesma referência geométrica 0007:
    # kz(0506,0603) = 2π/Ha(0007,0603) - 2π/Ha(0007,0506).
    # O voo 0007 cancela-se algebricamente e não entra como dado do TCCT.
    modo_geometria_rvog: str = "ha_p_real"  # "ha_p_real", "auto", "metadados", "calibracao_treino", "off"
    permitir_calibracao_rvog_treino: bool = False
    permitir_calibracao_atenuacao_com_kz_real: bool = True

    ha_p_0506_relativo: str = r"Dados_Brutos/SAR_P/tropi0506/tropi0007_0506_CONS_Ha.1"
    ha_p_0603_relativo: str = r"Dados_Brutos/SAR_P/tropi0603/tropi0007_0603_CONS_Ha.1"
    # Layout real inferido e revalidado por tamanho do arquivo + extensão LUT.
    ha_p_largura: int = 4000
    ha_p_altura: int = 7478
    ha_p_dtype: str = "<f4"
    ha_nodata_abs: float = 1.0e30
    ha_abs_minimo: float = 1.0e-6
    kz_p_abs_minimo_produto: float = 0.005
    kz_p_abs_maximo_produto: float = 0.40
    ha_p_referencia_geometrica: str = "tropi0007"
    max_amostras_calibracao_rvog: int = 4096
    gamma_min_calibracao: float = 0.03
    gamma_max_calibracao: float = 0.995
    kz_abs_min_calibracao: float = 0.005
    kz_abs_max_calibracao: float = 0.40
    atenuacao_min_calibracao: float = 0.005
    atenuacao_max_calibracao: float = 0.50
    rmse_complexo_max_calibracao: float = 0.35
    rmse_magnitude_max_calibracao: float = 0.25

    @property
    def raiz(self) -> Path:
        return localizar_base_dir(self.base_dir)

    @property
    def dir_codigo(self) -> Path:
        return self.raiz / "Codigo"

    @property
    def dir_brutos(self) -> Path:
        return self.raiz / "Dados_Brutos"

    @property
    def dir_processados(self) -> Path:
        return self.raiz / "Dados_Processados"

    @property
    def dir_lidar_original(self) -> Path:
        return self.dir_brutos / "LiDAR_Original"

    @property
    def dir_lidar_processado(self) -> Path:
        return self.dir_processados / "LiDAR_CHM"

    @property
    def dir_tropisar_original(self) -> Path:
        return self.dir_brutos / "TropiSAR_Original"

    @property
    def dir_inventario(self) -> Path:
        return self.dir_auditoria / "Inventario_TropiSAR"

    @property
    def dir_landsat_processado(self) -> Path:
        return self.dir_processados / "Landsat_Mosaico"

    @property
    def dir_backgeo_grille3d(self) -> Path:
        return self.dir_processados / "BackGeocoding_Grille3D"

    @property
    def dir_geometria_sar(self) -> Path:
        return self.dir_backgeo_grille3d / "Geometria"

    @property
    def caminho_dtm_alinhado(self) -> Path:
        return self.dir_lidar_processado / "DTM_alinhado_a_grade_DSM.tif"

    @property
    def caminho_dtm_para_grille(self) -> Path:
        return self.dir_backgeo_grille3d / "DTM_para_grille_GRS80.tif"

    @property
    def caminho_chm_canonico(self) -> Path:
        return self.dir_lidar_processado / "CHM_Paracou_2009_DSM_menos_DTM_validado.tif"

    @property
    def caminho_chm_bruto(self) -> Path:
        return self.dir_lidar_processado / "CHM_Paracou_2009_DSM_menos_DTM_bruto.tif"

    @property
    def caminho_mascara_chm(self) -> Path:
        return self.dir_lidar_processado / "CHM_Paracou_2009_valid_mask.tif"

    @property
    def dir_dataset_v2(self) -> Path:
        # Dataset separado dos resultados para evitar que tensores, checkpoints
        # e tabelas sejam confundidos.
        return self.raiz / "Dataset" / "TCCT_dataset_v6_grille3d"

    @property
    def dir_dataset_legacy(self) -> Path:
        # Opcional e vazio por padrão. Não copie o dataset antigo para cá,
        # exceto quando uma auditoria histórica específica for necessária.
        return self.raiz / "Dataset_Legado" / "Treino_TCCT_PolInSAR_30m"

    @property
    def dir_resultados_v2(self) -> Path:
        return self.raiz / "Resultados" / "TCCT_v3_2_SIBGRAPI"

    @property
    def dir_auditoria(self) -> Path:
        return self.raiz / "Auditoria"

    @property
    def dir_logs(self) -> Path:
        return self.raiz / "Logs"



def preparar_estrutura_projeto(
    cfg: ConfigDados,
    criar: bool = True,
) -> dict[str, Path]:
    """Cria e valida a árvore isolada SIBGRAPI_REV.

    Nenhum arquivo é copiado e nenhum processamento científico é iniciado.
    A função apenas cria diretórios vazios e impede que a execução recaia
    silenciosamente em TCCT_Geral ou em resultados históricos.
    """
    raiz = cfg.raiz

    if platform.system() == "Darwin":
        volume = Path("<LOCAL_PATH>")
        if not volume.exists():
            raise FileNotFoundError(
                "O HD externo 'PROJECT_VOLUME' não está montado em <LOCAL_PATH> "
                "Conecte o volume antes de executar o pipeline."
            )

    if raiz.name != "SIBGRAPI_REV":
        raise RuntimeError(
            f"Raiz recusada: {raiz}. A pasta final deve se chamar exatamente "
            "'SIBGRAPI_REV' para manter o experimento isolado."
        )

    diretorios = {
        "raiz": raiz,
        "codigo": cfg.dir_codigo,
        "dados_brutos": cfg.dir_brutos,
        "lidar_original": cfg.dir_lidar_original,
        "dados_processados": cfg.dir_processados,
        "lidar_chm_processado": cfg.dir_lidar_processado,
        "landsat_mosaico": cfg.dir_landsat_processado,
        "inventario_tropisar": cfg.dir_inventario,
        "dataset": cfg.dir_dataset_v2,
        "resultados": cfg.dir_resultados_v2,
        "auditoria": cfg.dir_auditoria,
        "logs": cfg.dir_logs,
    }
    if criar:
        for caminho in diretorios.values():
            caminho.mkdir(parents=True, exist_ok=True)

    marcador = raiz / "PROJETO_SIBGRAPI_REV.json"
    if criar:
        marcador.write_text(
            json.dumps(
                {
                    "projeto": "SIBGRAPI_REV",
                    "pipeline": VERSAO_PIPELINE,
                    "raiz_absoluta": str(raiz.resolve()),
                    "criado_ou_validado_em": pd.Timestamp.now().isoformat(),
                    "politica": (
                        "Dados brutos entram por Dados_Brutos; todo produto "
                        "derivado é gravado exclusivamente nesta raiz."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return diretorios


def imprimir_estrutura_projeto(cfg: ConfigDados) -> None:
    dirs = preparar_estrutura_projeto(cfg, criar=False)
    print("\nEstrutura isolada da revisão:")
    for nome, caminho in dirs.items():
        print(f"  {nome:<18} -> {caminho}")


@dataclass
class ConfigModelo:
    dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.10
    local_scale_inicial: float = 0.10


@dataclass
class ConfigTreino:
    seed: int = 42
    batch_size: int = 24
    num_workers: int = 0
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    k_folds: int = 5
    fold_validacao: int = 0
    # Três inicializações independentes para separar a variabilidade de seed
    # da variabilidade espacial entre folds.
    seeds_experimento: tuple[int, ...] = (42, 314, 2026)

    # Teste de sanidade: confirma que o modelo e os gradientes conseguem
    # reduzir a loss em um subconjunto pequeno antes do treino completo.
    executar_teste_sobreajuste: bool = True
    amostras_sobreajuste: int = 64
    passos_sobreajuste: int = 120
    reducao_minima_sobreajuste: float = 0.20

    # Loss supervisionada. O CCC substitui a penalização isolada da variância:
    # ele penaliza simultaneamente correlação, deslocamento de média e escala.
    w_pixel: float = 0.30
    w_patch: float = 0.50
    w_gradiente: float = 0.08
    w_ccc: float = 0.12
    limiar_estrato_alto_m: float = 28.0
    peso_estrato_alto: float = 1.30

    # Fase A — pré-treino supervisionado.
    fase_a_epocas: int = 80
    fase_a_min_epocas: int = 6
    fase_a_paciencia: int = 8
    fase_a_lr: float = 1.0e-4
    fase_a_lr_min: float = 1.0e-6
    fase_a_scheduler_paciencia: int = 3

    # Baselines CNN/ViT usam treinamento supervisionado de uma fase.
    baseline_epocas: int = 80
    baseline_min_epocas: int = 6
    baseline_paciencia: int = 8
    baseline_lr: float = 1.0e-4
    baseline_lr_min: float = 1.0e-6
    baseline_scheduler_paciencia: int = 3

    # Fase B — refinamento físico iniciado do melhor checkpoint da Fase A.
    executar_fase_b: bool = True
    fase_b_epocas: int = 30
    fase_b_min_epocas: int = 5
    fase_b_paciencia: int = 8
    fase_b_lr: float = 1.0e-5
    fase_b_lr_min: float = 1.0e-7
    fase_b_scheduler_paciencia: int = 3
    lambda_rvog_inicial: float = 0.002
    lambda_rvog_final: float = 0.020
    fase_b_epocas_rampa: int = 10
    min_amostras_fisica_treino: int = 64
    min_amostras_fisica_validacao: int = 16

    # Parâmetros físicos. Quando a geometria real está disponível, sigma é
    # aprendível. Sem metadados, a v2.7 usa kz real P e calibra apenas a atenuação efetiva
    # a=2*sigma/cos(theta) exclusivamente no conjunto de treino do fold.
    sigma_l_inicial: float = 0.15
    sigma_p_inicial: float = 0.10
    modo_fisica: str = "magnitude"  # "off", "magnitude" ou "complexa_derotada"

    # Aumento geométrico fica desligado por padrão: SAR é direcional.
    augmentacao_geometrica: bool = False


@dataclass
class ConfigPipeline:
    dados: ConfigDados = field(default_factory=ConfigDados)
    modelo: ConfigModelo = field(default_factory=ConfigModelo)
    treino: ConfigTreino = field(default_factory=ConfigTreino)


def executar_etapa(nome: str, funcao, *args, **kwargs):
    """Executa uma etapa com saída imediata, cronômetro e traceback completo."""
    inicio = time.perf_counter()
    print("\n" + "="*88, flush=True)
    print(f"▶ INÍCIO — {nome}", flush=True)
    print(f"  Python: {sys.executable}", flush=True)
    print(f"  Horário: {pd.Timestamp.now().isoformat()}", flush=True)
    print("="*88, flush=True)
    try:
        resultado = funcao(*args, **kwargs)
    except Exception as exc:
        duracao = time.perf_counter() - inicio
        print("\n" + "!"*88, flush=True)
        print(f"✗ ERRO EM {nome} após {duracao:.1f} s: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        print("!"*88, flush=True)
        raise
    duracao = time.perf_counter() - inicio
    print("\n" + "="*88, flush=True)
    print(f"✓ FIM — {nome} | duração: {duracao:.1f} s", flush=True)
    print("="*88, flush=True)
    return resultado


CFG = ConfigPipeline()
DEVICE, PIN_MEMORY = detectar_device()


def fixar_sementes(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


fixar_sementes(CFG.treino.seed)
print(f"TCCT pipeline {VERSAO_PIPELINE}")
print(f"Sistema={platform.system()} | Python={platform.python_version()} | PyTorch={torch.__version__}")
print(f"Dispositivo={DEVICE} | Base={CFG.dados.raiz}")






# ================================================================
# 1. AUDITORIA PRÉVIA DOS ARQUIVOS E METADADOS
# ================================================================
def hash_curto(caminho: Path, bytes_iniciais: int = 1_048_576) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        h.update(f.read(bytes_iniciais))
    return h.hexdigest()[:16]


def _arquivo_auxiliar_macos(path: Path) -> bool:
    """Identifica AppleDouble e artefatos de extração que não são dados científicos."""
    partes = set(path.parts)
    return (
        path.name.startswith("._")
        or path.name == ".DS_Store"
        or "__MACOSX" in partes
    )


def _arquivo_cientifico_real(path: Path) -> bool:
    return path.is_file() and not _arquivo_auxiliar_macos(path)


def _listar_reais(pasta: Path, padroes: Sequence[str]) -> list[Path]:
    achados: set[Path] = set()
    for padrao in padroes:
        for p in pasta.glob(padrao):
            if _arquivo_cientifico_real(p):
                achados.add(p.resolve())
    return sorted(achados)


def _contem_componente(path: Path, componente: str) -> bool:
    alvo = componente.lower()
    return alvo in [p.lower() for p in path.parts]


def _esta_na_pasta_temporal_da_aquisicao(path: Path, aquisicao: str) -> bool:
    partes = [p.lower() for p in path.parts]
    aq = aquisicao.lower()
    for i, parte in enumerate(partes[:-1]):
        if parte == "temporal_data" and i + 1 < len(partes):
            if partes[i + 1] == aq:
                return True
    return False


def _esta_na_banda(path: Path, banda: str) -> bool:
    esperado = "l_band" if banda == "SAR_L" else "p_band"
    return _contem_componente(path, esperado)


def localizar_dsm_dtm_originais(cfg: ConfigDados) -> tuple[Path, Path]:
    """Localiza DSM e DTM reais, ignorando arquivos AppleDouble `._*`."""
    raiz = cfg.dir_lidar_original
    if not raiz.exists():
        raise FileNotFoundError(
            f"Pasta LiDAR original ausente: {raiz}. "
            "Crie-a e copie DSM, DTM e arquivos README originais."
        )

    candidatos_dsm = sorted({
        p.resolve()
        for padrao in ("*DSM*.tif", "*DSM*.TIF", "*dsm*.tif", "*dsm*.TIF")
        for p in raiz.rglob(padrao)
        if _arquivo_cientifico_real(p)
    })
    candidatos_dtm = sorted({
        p.resolve()
        for padrao in ("DTM.asc", "*DTM*.asc", "*dtm*.asc")
        for p in raiz.rglob(padrao)
        if _arquivo_cientifico_real(p)
    })

    if len(candidatos_dsm) != 1:
        raise RuntimeError(
            "Era esperado exatamente um DSM GeoTIFF real em LiDAR_Original; "
            f"foram encontrados {len(candidatos_dsm)}:\n"
            + "\n".join(map(str, candidatos_dsm))
        )
    if len(candidatos_dtm) != 1:
        raise RuntimeError(
            "Era esperado exatamente um DTM ASCII real em LiDAR_Original; "
            f"foram encontrados {len(candidatos_dtm)}:\n"
            + "\n".join(map(str, candidatos_dtm))
        )
    return candidatos_dsm[0], candidatos_dtm[0]


def encontrar_chm(cfg: ConfigDados) -> Path:
    """Retorna somente o CHM canônico reproduzível gerado pela v2.8.3."""
    caminho = cfg.caminho_chm_canonico
    if not caminho.exists():
        raise FileNotFoundError(
            f"CHM canônico ainda não gerado: {caminho}\n"
            "Execute primeiro a etapa 'preparar-chm'. O pipeline não usa "
            "automaticamente um CHM legado ou produzido por código anterior."
        )
    return caminho


def achar_arquivo_unico(
    pasta: Path,
    padroes: Sequence[str],
) -> Optional[Path]:
    """Retorna um arquivo real; ignora `._*`, `.DS_Store` e `__MACOSX`."""
    achados = _listar_reais(pasta, padroes)
    if not achados:
        return None
    if len(achados) > 1:
        raise RuntimeError(
            f"Mais de um arquivo real corresponde aos padrões {tuple(padroes)} "
            f"em {pasta}:\n" + "\n".join(map(str, achados))
        )
    return achados[0]


def _ler_mtl_texto_rapido(cena: Path) -> str:
    """Leitura mínima de MTL usada já na auditoria inicial."""
    mtl = achar_arquivo_unico(cena, ("*_MTL.txt", "*MTL*.txt"))
    return (
        mtl.read_text(encoding="utf-8", errors="ignore")
        if mtl is not None else ""
    )


def extrair_data_cena_landsat_rapida(cena: Path) -> Optional[str]:
    """Extrai data sem depender das funções definidas no bloco Landsat posterior."""
    texto = _ler_mtl_texto_rapido(cena)
    m = re.search(
        r'^\s*DATE_ACQUIRED\s*=\s*"?(\d{4}-\d{2}-\d{2})"?\s*$',
        texto,
        re.M,
    )
    if m:
        return m.group(1)

    m = re.search(r"_(\d{8})_", cena.name)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return None


def descobrir_cenas_landsat(cfg: ConfigDados) -> list[Path]:
    """Descobre cenas completas sem depender de células executadas depois."""
    raiz = cfg.dir_brutos / "Landsat-5" / "Paracou"
    if not raiz.exists():
        raise FileNotFoundError(f"Pasta Landsat-5/Paracou ausente: {raiz}")

    candidatos = []
    pastas = [raiz] + [p for p in raiz.rglob("*") if p.is_dir()]
    print(
        f"  Landsat: examinando {len(pastas)} pastas candidatas...",
        flush=True,
    )
    for pasta in pastas:
        presentes = sum(
            achar_arquivo_unico(
                pasta,
                (f"*_SR_B{b}.TIF", f"*B{b}.TIF"),
            ) is not None
            for b in (1, 2, 3, 4, 5, 7)
        )
        if presentes == 6:
            candidatos.append(pasta.resolve())

    cenas = sorted(
        set(candidatos),
        key=lambda p: (
            extrair_data_cena_landsat_rapida(p) or "",
            p.name,
        ),
    )
    if not cenas:
        raise FileNotFoundError(
            f"Nenhuma cena Landsat-5 completa encontrada em {raiz}"
        )
    if len(cenas) > cfg.landsat_max_cenas:
        raise RuntimeError(
            f"Foram encontradas {len(cenas)} cenas, acima do limite auditado "
            f"{cfg.landsat_max_cenas}. Revise a pasta antes de continuar."
        )

    print(f"  Landsat: {len(cenas)} cena(s) completa(s) encontrada(s):", flush=True)
    for cena in cenas:
        print(
            f"    - {cena.name} | data="
            f"{extrair_data_cena_landsat_rapida(cena) or 'não identificada'}",
            flush=True,
        )
    return cenas


def encontrar_cena_landsat(cfg: ConfigDados) -> Path:
    """Compatibilidade com rotinas que ainda esperam uma cena determinística."""
    cenas = descobrir_cenas_landsat(cfg)
    preferidas = [
        p for p in cenas if "LT05_L2SP_227056_20090829" in p.name
    ]
    return preferidas[0] if preferidas else cenas[0]


def validar_crs_documentado_lidar(
    cfg: ConfigDados,
) -> dict:
    """Confirma em README original a referência WGS 84 / UTM zona 22N.

    Esta função não infere CRS por localização geográfica. Ela exige evidência
    textual dentro de Dados_Brutos/LiDAR_Original.
    """
    padroes = (
        "*ReadMe*.txt",
        "*README*.txt",
        "*readme*.txt",
        "*Readme*.txt",
    )
    arquivos = sorted({
        p.resolve()
        for padrao in padroes
        for p in cfg.dir_lidar_original.rglob(padrao)
        if _arquivo_cientifico_real(p)
    })

    evidencias = []
    regexes = (
        re.compile(r"WGS\s*84.{0,80}UTM.{0,40}(?:N\s*22|22\s*N)", re.I | re.S),
        re.compile(r"UTM\s*(?:zone|zona)?\s*22\s*N", re.I),
        re.compile(r"UTM\s*N\s*22", re.I),
    )

    for arquivo in arquivos:
        texto = arquivo.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        for regex in regexes:
            achado = regex.search(texto)
            if achado:
                trecho = " ".join(achado.group(0).split())
                evidencias.append({
                    "arquivo": str(arquivo),
                    "expressao_encontrada": trecho,
                })
                break

    confirmado = bool(evidencias)
    resultado = {
        "epsg_atribuido": int(cfg.lidar_crs_epsg),
        "crs_esperado": f"EPSG:{cfg.lidar_crs_epsg}",
        "readmes_examinados": [str(p) for p in arquivos],
        "evidencias": evidencias,
        "confirmado": confirmado,
    }

    if cfg.lidar_exigir_readme_para_crs and not confirmado:
        raise RuntimeError(
            "O raster não possui CRS embutido e nenhum README original confirmou "
            "WGS 84 / UTM zona 22N. A atribuição automática foi recusada."
        )
    return resultado


def ler_cabecalho_esri_ascii(path: Path) -> dict[str, float]:
    """Lê as seis primeiras linhas de um ESRI ASCII Grid."""
    campos: dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(6):
            linha = f.readline()
            if not linha:
                break
            partes = linha.strip().split()
            if len(partes) >= 2:
                chave = partes[0].lower()
                try:
                    campos[chave] = float(partes[1])
                except ValueError:
                    pass
    return campos


def validar_cabecalho_dtm(path: Path, cfg: ConfigDados) -> dict:
    cab = ler_cabecalho_esri_ascii(path)
    esperado = {
        "ncols": float(cfg.dtm_ncols_esperado),
        "nrows": float(cfg.dtm_nrows_esperado),
        "xllcorner": float(cfg.dtm_xllcorner_esperado),
        "yllcorner": float(cfg.dtm_yllcorner_esperado),
        "cellsize": float(cfg.dtm_cellsize_esperado),
        "nodata_value": float(cfg.dtm_nodata_esperado),
    }
    ausentes = [k for k in esperado if k not in cab]
    divergencias = {
        k: {"observado": cab.get(k), "esperado": v}
        for k, v in esperado.items()
        if k in cab and not np.isclose(cab[k], v, rtol=0.0, atol=1e-6)
    }
    if cfg.validar_cabecalho_dtm_estritamente and (ausentes or divergencias):
        raise RuntimeError(
            "O cabeçalho do DTM diverge do produto documentado. "
            f"Ausentes={ausentes}; divergências={divergencias}"
        )
    return {
        "observado": cab,
        "esperado": esperado,
        "ausentes": ausentes,
        "divergencias": divergencias,
        "valido": not ausentes and not divergencias,
    }


def _estatisticas_raster_valido(arr: np.ndarray) -> dict:
    x = np.asarray(arr, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])
    return {
        "n": int(x.size),
        "min": float(q[0]),
        "q0_1": float(q[1]),
        "q1": float(q[2]),
        "q5": float(q[3]),
        "q25": float(q[4]),
        "mediana": float(q[5]),
        "q75": float(q[6]),
        "q95": float(q[7]),
        "q99": float(q[8]),
        "q99_9": float(q[9]),
        "max": float(q[10]),
        "media": float(np.mean(x)),
        "desvio_padrao": float(np.std(x)),
    }


def _configurar_blocos_geotiff(profile: dict) -> dict:
    """Configura blocos TIFF válidos para rasters grandes e pequenos."""
    profile = profile.copy()
    for chave in ("blockxsize", "blockysize", "tiled"):
        profile.pop(chave, None)

    largura = int(profile["width"])
    altura = int(profile["height"])
    if largura >= 16 and altura >= 16:
        bloco_x = min(256, (largura // 16) * 16)
        bloco_y = min(256, (altura // 16) * 16)
        profile.update(
            tiled=True,
            blockxsize=max(16, bloco_x),
            blockysize=max(16, bloco_y),
        )
    else:
        profile.update(tiled=False)
    return profile


def _salvar_geotiff_float32(
    path: Path,
    arr: np.ndarray,
    profile_ref: dict,
    nodata: float = -9999.0,
) -> None:
    profile = profile_ref.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=nodata,
        compress="deflate",
        predictor=3,
        BIGTIFF="IF_SAFER",
    )
    profile = _configurar_blocos_geotiff(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    saida = np.where(
        np.isfinite(arr),
        arr,
        nodata,
    ).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(saida, 1)


def _salvar_geotiff_uint8(
    path: Path,
    arr: np.ndarray,
    profile_ref: dict,
) -> None:
    profile = profile_ref.copy()
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=0,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    profile = _configurar_blocos_geotiff(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


def _comparar_chm_legado_opcional(
    cfg: ConfigDados,
    chm_novo: np.ndarray,
    profile_ref: dict,
) -> dict:
    """Compara, sem usar no treino, um CHM legado colocado em Auditoria/CHM_Legado."""
    pasta = cfg.dir_auditoria / "CHM_Legado"
    candidatos = sorted(list(pasta.glob("*.tif")) + list(pasta.glob("*.TIF"))) if pasta.exists() else []
    if not candidatos:
        return {"executado": False, "motivo": "nenhum CHM legado fornecido"}
    if len(candidatos) > 1:
        return {
            "executado": False,
            "motivo": "mais de um CHM legado encontrado; comparação ambígua",
            "arquivos": list(map(str, candidatos)),
        }

    legado_alinhado = np.full(
        (profile_ref["height"], profile_ref["width"]),
        np.nan,
        dtype=np.float32,
    )
    with rasterio.open(candidatos[0]) as src:
        src_crs = src.crs or CRS.from_epsg(cfg.lidar_crs_epsg)
        reproject(
            source=rasterio.band(src, 1),
            destination=legado_alinhado,
            src_transform=src.transform,
            src_crs=src_crs,
            src_nodata=src.nodata,
            dst_transform=profile_ref["transform"],
            dst_crs=profile_ref["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid = np.isfinite(chm_novo) & np.isfinite(legado_alinhado)
    if not valid.any():
        return {"executado": False, "motivo": "sem interseção válida"}
    dif = chm_novo[valid] - legado_alinhado[valid]
    return {
        "executado": True,
        "arquivo_legado": str(candidatos[0]),
        "n_pixels": int(valid.sum()),
        "rmse_m": float(np.sqrt(np.mean(dif ** 2))),
        "bias_novo_menos_legado_m": float(np.mean(dif)),
        "mae_m": float(np.mean(np.abs(dif))),
        "correlacao": float(np.corrcoef(chm_novo[valid], legado_alinhado[valid])[0, 1]),
        "observacao": "comparação apenas diagnóstica; o CHM legado não entra no treino",
    }


def preparar_chm_lidar_reproduzivel(
    cfg: ConfigDados = CFG.dados,
    sobrescrever: Optional[bool] = None,
) -> Path:
    """Gera o CHM canônico por DSM − DTM com CRS documental auditado.

    O DSM original pode não conter a tag CRS. Nesse caso, EPSG:32622 só é
    atribuído quando:
    1. a configuração permite;
    2. um README original confirma WGS 84 / UTM zona 22N;
    3. o DSM possui transformação espacial válida e não identitária.

    A atribuição é registrada como proveniência, não como CRS embutido original.
    """
    sobrescrever = (
        cfg.sobrescrever if sobrescrever is None else bool(sobrescrever)
    )
    out = cfg.dir_lidar_processado
    out.mkdir(parents=True, exist_ok=True)

    if cfg.caminho_chm_canonico.exists() and not sobrescrever:
        print(
            f"CHM canônico já existe; mantendo: {cfg.caminho_chm_canonico}"
        )
        return cfg.caminho_chm_canonico

    dsm_path, dtm_path = localizar_dsm_dtm_originais(cfg)
    cabecalho_dtm = validar_cabecalho_dtm(dtm_path, cfg)
    crs_documentado = CRS.from_epsg(cfg.lidar_crs_epsg)
    evidencia_crs = None

    with rasterio.open(dsm_path) as src_dsm:
        if src_dsm.transform.is_identity:
            raise RuntimeError(
                "O DSM possui transformação espacial identitária. O README "
                "documenta o CRS, mas não fornece geotransformação suficiente "
                "para posicionar o raster; a geração do CHM foi recusada."
            )

        crs_dsm_embutido = src_dsm.crs
        if crs_dsm_embutido is None:
            if not cfg.lidar_permitir_atribuicao_crs_documental:
                raise RuntimeError(
                    "O DSM não possui CRS embutido e a atribuição documental "
                    "está desativada na configuração."
                )
            evidencia_crs = validar_crs_documentado_lidar(cfg)
            crs_dsm_usado = crs_documentado
            origem_crs_dsm = "atribuido_do_README_original"
            print(
                "DSM sem CRS embutido: atribuindo EPSG:"
                f"{cfg.lidar_crs_epsg} com base no README original.",
                flush=True,
            )
            for ev in evidencia_crs["evidencias"]:
                print(
                    f"  Evidência documental: {ev['arquivo']} | "
                    f"{ev['expressao_encontrada']}",
                    flush=True,
                )
        else:
            crs_dsm_usado = crs_dsm_embutido
            origem_crs_dsm = "embutido_no_DSM"
            if crs_dsm_usado != crs_documentado:
                raise RuntimeError(
                    f"CRS do DSM inesperado: {crs_dsm_usado}; "
                    f"esperado EPSG:{cfg.lidar_crs_epsg}."
                )

        dsm_profile = src_dsm.profile.copy()
        dsm_profile.update(
            height=src_dsm.height,
            width=src_dsm.width,
            transform=src_dsm.transform,
            crs=crs_dsm_usado,
        )
        dsm_bounds = {
            "left": float(src_dsm.bounds.left),
            "bottom": float(src_dsm.bounds.bottom),
            "right": float(src_dsm.bounds.right),
            "top": float(src_dsm.bounds.top),
        }
        dsm_resolucao = [
            float(abs(src_dsm.transform.a)),
            float(abs(src_dsm.transform.e)),
        ]
        dsm_ma = src_dsm.read(1, masked=True)
        dsm = dsm_ma.filled(np.nan).astype(np.float32)

    with rasterio.open(dtm_path) as src_dtm:
        if src_dtm.transform.is_identity:
            raise RuntimeError(
                "O DTM ASCII foi aberto sem transformação espacial válida. "
                "Revise o cabeçalho ESRI ASCII."
            )

        crs_dtm_embutido = src_dtm.crs
        if crs_dtm_embutido is None:
            if evidencia_crs is None:
                evidencia_crs = validar_crs_documentado_lidar(cfg)
            dtm_crs = crs_documentado
            origem_crs_dtm = "atribuido_do_README_original"
        else:
            dtm_crs = crs_dtm_embutido
            origem_crs_dtm = "embutido_no_DTM"
            if dtm_crs != crs_documentado:
                raise RuntimeError(
                    f"CRS do DTM inesperado: {dtm_crs}; "
                    f"esperado EPSG:{cfg.lidar_crs_epsg}."
                )

        dtm_bounds_original = {
            "left": float(src_dtm.bounds.left),
            "bottom": float(src_dtm.bounds.bottom),
            "right": float(src_dtm.bounds.right),
            "top": float(src_dtm.bounds.top),
        }
        dtm_resolucao_original = [
            float(abs(src_dtm.transform.a)),
            float(abs(src_dtm.transform.e)),
        ]

        # Verificação geométrica antes da reamostragem.
        inter_left = max(dsm_bounds["left"], dtm_bounds_original["left"])
        inter_bottom = max(dsm_bounds["bottom"], dtm_bounds_original["bottom"])
        inter_right = min(dsm_bounds["right"], dtm_bounds_original["right"])
        inter_top = min(dsm_bounds["top"], dtm_bounds_original["top"])
        largura_inter = max(0.0, inter_right - inter_left)
        altura_inter = max(0.0, inter_top - inter_bottom)
        area_inter = largura_inter * altura_inter
        area_dsm = max(
            0.0,
            (dsm_bounds["right"] - dsm_bounds["left"])
            * (dsm_bounds["top"] - dsm_bounds["bottom"]),
        )
        fracao_area_dsm_coberta = (
            float(area_inter / area_dsm) if area_dsm > 0 else 0.0
        )
        if area_inter <= 0:
            raise RuntimeError(
                "DSM e DTM não possuem interseção espacial segundo suas "
                "geotransformações. A subtração foi recusada."
            )

        dtm_alinhado = np.full(dsm.shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src_dtm, 1),
            destination=dtm_alinhado,
            src_transform=src_dtm.transform,
            src_crs=dtm_crs,
            src_nodata=(
                src_dtm.nodata
                if src_dtm.nodata is not None
                else cfg.dtm_nodata_esperado
            ),
            dst_transform=dsm_profile["transform"],
            dst_crs=dsm_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    valid_dsm = np.isfinite(dsm)
    valid_base = valid_dsm & np.isfinite(dtm_alinhado)
    if not valid_base.any():
        raise RuntimeError(
            "Após o alinhamento, DSM e DTM não possuem pixels válidos em comum."
        )

    fracao_dsm_valido_com_dtm = float(
        valid_base.sum() / max(1, valid_dsm.sum())
    )

    chm_bruto = np.full(dsm.shape, np.nan, dtype=np.float32)
    chm_bruto[valid_base] = dsm[valid_base] - dtm_alinhado[valid_base]

    pequenos_negativos = (
        valid_base
        & (chm_bruto < 0.0)
        & (chm_bruto >= -cfg.chm_tolerancia_negativa_m)
    )
    negativos_graves = (
        valid_base
        & (chm_bruto < -cfg.chm_tolerancia_negativa_m)
    )

    chm_validado = chm_bruto.copy()
    chm_validado[pequenos_negativos] = 0.0
    chm_validado[negativos_graves] = np.nan
    valid_final = np.isfinite(chm_validado)

    _salvar_geotiff_float32(
        cfg.caminho_chm_bruto,
        chm_bruto,
        dsm_profile,
    )
    _salvar_geotiff_float32(
        cfg.caminho_chm_canonico,
        chm_validado,
        dsm_profile,
    )
    _salvar_geotiff_uint8(
        cfg.caminho_mascara_chm,
        valid_final.astype(np.uint8),
        dsm_profile,
    )

    if cfg.salvar_dtm_alinhado:
        _salvar_geotiff_float32(
            out / "DTM_alinhado_a_grade_DSM.tif",
            dtm_alinhado,
            dsm_profile,
        )

    bounds_dtm_header = {
        "left": cfg.dtm_xllcorner_esperado,
        "bottom": cfg.dtm_yllcorner_esperado,
        "right": (
            cfg.dtm_xllcorner_esperado
            + cfg.dtm_ncols_esperado * cfg.dtm_cellsize_esperado
        ),
        "top": (
            cfg.dtm_yllcorner_esperado
            + cfg.dtm_nrows_esperado * cfg.dtm_cellsize_esperado
        ),
    }

    comparacao_legado = _comparar_chm_legado_opcional(
        cfg,
        chm_validado,
        dsm_profile,
    )

    relatorio = {
        "pipeline_version": VERSAO_PIPELINE,
        "metodo": "DSM_alinhado_menos_DTM_alinhado",
        "equivalencia_com_CHM_historico_LAStools": "nao_garantida",
        "justificativa": (
            "O README informa que o CHM histórico foi gerado pela rotina "
            "Canopy Height do LAStools, mas esse GeoTIFF não estava disponível. "
            "A v2.8.3.4 gera um produto reproduzível a partir dos rasters DSM "
            "e DTM originais."
        ),
        "dsm": str(dsm_path),
        "dtm": str(dtm_path),
        "crs_usado": str(dsm_profile["crs"]),
        "epsg_esperado": int(cfg.lidar_crs_epsg),
        "crs_dsm_embutido_originalmente": (
            str(crs_dsm_embutido) if crs_dsm_embutido is not None else None
        ),
        "origem_crs_dsm": origem_crs_dsm,
        "crs_dtm_embutido_originalmente": (
            str(crs_dtm_embutido) if crs_dtm_embutido is not None else None
        ),
        "origem_crs_dtm": origem_crs_dtm,
        "evidencia_documental_crs": evidencia_crs,
        "shape_grade_saida": [
            int(dsm_profile["height"]),
            int(dsm_profile["width"]),
        ],
        "transform_grade_saida": list(dsm_profile["transform"])[:6],
        "bounds_dsm": dsm_bounds,
        "resolucao_dsm_m": dsm_resolucao,
        "bounds_dtm_original": dtm_bounds_original,
        "resolucao_dtm_original_m": dtm_resolucao_original,
        "fracao_area_dsm_coberta_pelo_extent_dtm": fracao_area_dsm_coberta,
        "fracao_pixels_dsm_validos_com_dtm_valido": fracao_dsm_valido_com_dtm,
        "cabecalho_dtm": cabecalho_dtm,
        "bounds_dtm_derivados_do_cabecalho": bounds_dtm_header,
        "resampling_dtm_para_dsm": "bilinear",
        "politica_negativos": {
            "intervalo_clamp_zero_m": [
                -cfg.chm_tolerancia_negativa_m,
                0.0,
            ],
            "abaixo_disso": "NoData",
        },
        "contadores": {
            "pixels_total": int(chm_bruto.size),
            "pixels_dsm_validos": int(valid_dsm.sum()),
            "pixels_dsm_dtm_validos": int(valid_base.sum()),
            "pequenos_negativos_convertidos_zero": int(
                pequenos_negativos.sum()
            ),
            "negativos_graves_convertidos_nodata": int(
                negativos_graves.sum()
            ),
            "pixels_validos_finais": int(valid_final.sum()),
            "pixels_acima_alerta_60m": int(
                np.sum(
                    valid_final
                    & (chm_validado > cfg.chm_alerta_altura_m)
                )
            ),
        },
        "estatisticas_dsm": _estatisticas_raster_valido(dsm),
        "estatisticas_dtm_alinhado": _estatisticas_raster_valido(
            dtm_alinhado
        ),
        "estatisticas_chm_bruto": _estatisticas_raster_valido(
            chm_bruto
        ),
        "estatisticas_chm_validado": _estatisticas_raster_valido(
            chm_validado
        ),
        "comparacao_chm_legado": comparacao_legado,
        "arquivos_saida": {
            "chm_bruto": str(cfg.caminho_chm_bruto),
            "chm_canonico": str(cfg.caminho_chm_canonico),
            "mascara": str(cfg.caminho_mascara_chm),
        },
        "proveniencia_documental": {
            "DSM": (
                "DSM produzido por LAStools a partir da nuvem de pontos de "
                "abril de 2009 e filtrado por mediana condicional."
            ),
            "DTM": (
                "DTM de 1 m produzido a partir de pontos classificados como "
                "terreno."
            ),
            "CRS": (
                "WGS 84 / UTM zona 22N confirmado em README original; "
                "atribuído ao raster apenas quando a tag CRS está ausente."
            ),
            "CHM_historico": (
                "README informa geração pela rotina Canopy Height do LAStools; "
                "produto histórico não usado automaticamente."
            ),
        },
    }
    (out / "relatorio_geracao_chm.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Figuras diagnósticas leves.
    passo = max(1, int(max(chm_validado.shape) / 1200))
    mapa_dec = chm_validado[::passo, ::passo]
    fin = mapa_dec[np.isfinite(mapa_dec)]
    if fin.size:
        vmin, vmax = np.percentile(fin, [1, 99])
        plt.figure(figsize=(10, 7))
        plt.imshow(mapa_dec, vmin=vmin, vmax=vmax)
        plt.colorbar(label="CHM [m]")
        plt.title("Reconstructed CHM: DSM − DTM")
        plt.xlabel("Decimated column")
        plt.ylabel("Decimated row")
        plt.tight_layout()
        plt.savefig(
            out / "mapa_diagnostico_chm.png",
            dpi=180,
        )
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.hist(fin, bins=150)
        plt.xlabel("CHM [m]")
        plt.ylabel("Frequency")
        plt.title("Distribution of the reconstructed CHM")
        plt.tight_layout()
        plt.savefig(
            out / "histograma_chm.png",
            dpi=180,
        )
        plt.close()

    print("\nCHM LiDAR reproduzível gerado com sucesso.")
    print(f"DSM: {dsm_path}")
    print(f"DTM: {dtm_path}")
    print(f"CRS usado: {dsm_profile['crs']} ({origem_crs_dsm})")
    print(f"CHM canônico: {cfg.caminho_chm_canonico}")
    print(
        json.dumps(
            relatorio["contadores"],
            indent=2,
            ensure_ascii=False,
        )
    )
    return cfg.caminho_chm_canonico




def _sha256_completo(path: Path, chunk_mb: int = 4) -> str:
    h = hashlib.sha256()
    tamanho = max(1, int(chunk_mb)) * 1024 * 1024
    with path.open("rb") as f:
        while True:
            bloco = f.read(tamanho)
            if not bloco:
                break
            h.update(bloco)
    return h.hexdigest()


def _sha256_amostral(path: Path, chunk_bytes: int = 1_048_576) -> str:
    """Hash rápido de tamanho + início + meio + fim; não substitui SHA-256 completo."""
    tamanho = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(tamanho).encode("ascii"))
    with path.open("rb") as f:
        if tamanho <= 3 * chunk_bytes:
            h.update(f.read())
        else:
            h.update(f.read(chunk_bytes))
            f.seek(max(0, tamanho // 2 - chunk_bytes // 2))
            h.update(f.read(chunk_bytes))
            f.seek(max(0, tamanho - chunk_bytes))
            h.update(f.read(chunk_bytes))
    return h.hexdigest()


def _classificar_arquivo_acervo(path: Path) -> str:
    texto = str(path).lower()
    nome = path.name.lower()
    sufixo = path.suffix.lower()
    raster_ext = {".tif", ".tiff", ".asc", ".img", ".vrt"}

    if (
        any(t in nome for t in ("chm", "canopy", "mnh"))
        and sufixo in raster_ext
    ):
        return "candidato_chm"
    if "dsm" in nome and sufixo in {".tif", ".tiff", ".img", ".vrt"}:
        return "dsm"
    if "dtm" in nome and sufixo in raster_ext:
        return "dtm"
    if "biomass" in nome or "biomassa" in nome:
        return "biomassa_in_situ"
    if "summary_in-situ" in nome or "summary_in_situ" in nome:
        return "resumo_in_situ"
    if "tropisar_paracou" in nome and sufixo in (".xls", ".xlsx", ".csv"):
        return "roi_analysis"
    if "interfero_geom" in texto:
        return "geometria_interferometrica"
    if "adapt_traj" in texto:
        return "trajetoria_adaptada"
    if re.search(r"(?:^|[_-])ha(?:[._-]|$)", nome) or "_ha." in nome:
        return "altura_ambiguidade"
    if "kz" in nome or "vertical_wavenumber" in nome:
        return "kz"
    if "baseline" in texto:
        return "baseline"
    if nome.endswith(".grille"):
        return "grille"
    if "roi_paracou" in nome and sufixo == ".txt":
        return "lut"
    if "slc" in nome and sufixo in (".dat", ".bin", ".slc"):
        return "slc"
    if "landsat" in texto or re.search(r"l[te]05_", nome) or "_sr_b" in nome:
        return "landsat"
    if "tropisar_report" in texto or (
        sufixo == ".pdf"
        and any(t in nome for t in ("tropisar", "format", "report"))
    ):
        return "relatorio_tropisar"
    if sufixo in (
        ".xml", ".hdr", ".ann", ".par", ".cfg",
        ".json", ".txt", ".md",
    ):
        return "metadado"
    return "outro"


def _inferir_banda_caminho(path: Path) -> str:
    t = str(path).lower().replace("-", "_")
    nome = path.name.lower()
    if "lcons" in nome or any(x in t for x in ("/l_band/", "\\l_band\\", "/sar_l/", "\\sar_l\\")):
        return "L"
    if "pcons" in nome or any(x in t for x in ("/p_band/", "\\p_band\\", "/sar_p/", "\\sar_p\\")):
        return "P"
    return ""


def _inferir_aquisicao_caminho(path: Path) -> str:
    m = re.search(r"tropi\d{4}", str(path).lower())
    return m.group(0) if m else ""


def _inferir_polarizacao_caminho(path: Path) -> str:
    nome = path.name.lower()
    for pol in ("hh", "hv", "vh", "vv"):
        if re.search(rf"(?:_|-){pol}(?:_|-|\.|$)", nome):
            return pol.upper()
    return ""


def inventariar_dados_brutos_completo(
    cfg: ConfigDados = CFG.dados,
    hash_completo_todos: Optional[bool] = None,
) -> pd.DataFrame:
    """Cataloga todo o conteúdo de Dados_Brutos sem mover ou renomear arquivos."""
    raiz = cfg.dir_brutos
    if not raiz.exists():
        raise FileNotFoundError(f"Dados_Brutos ausente: {raiz}")

    cfg.dir_inventario.mkdir(parents=True, exist_ok=True)
    completo_todos = (
        cfg.inventario_hash_completo_todos
        if hash_completo_todos is None else bool(hash_completo_todos)
    )
    limite = cfg.inventario_hash_completo_limite_mb * 2**20

    linhas = []
    todos = sorted(p for p in raiz.rglob("*") if p.is_file())
    auxiliares_macos = [p for p in todos if _arquivo_auxiliar_macos(p)]
    arquivos = [p for p in todos if not _arquivo_auxiliar_macos(p)]

    pd.DataFrame({
        "caminho_relativo": [
            str(p.relative_to(raiz)) for p in auxiliares_macos
        ],
        "motivo": "AppleDouble/metadata auxiliar do macOS",
    }).to_csv(
        cfg.dir_inventario / "arquivos_auxiliares_macos_ignorados.csv",
        index=False,
    )
    for p in tqdm(arquivos, desc="Inventário integral de Dados_Brutos"):
        tamanho = p.stat().st_size
        categoria = _classificar_arquivo_acervo(p)
        fazer_completo = completo_todos or tamanho <= limite or categoria in {
            "candidato_chm", "dsm", "dtm", "altura_ambiguidade",
            "roi_analysis", "biomassa_in_situ", "relatorio_tropisar",
        }
        linhas.append({
            "caminho_relativo": str(p.relative_to(raiz)),
            "nome": p.name,
            "extensao": p.suffix.lower(),
            "tamanho_bytes": int(tamanho),
            "tamanho_mb": float(tamanho / 2**20),
            "categoria": categoria,
            "banda": _inferir_banda_caminho(p),
            "aquisicao": _inferir_aquisicao_caminho(p),
            "polarizacao": _inferir_polarizacao_caminho(p),
            "sha256_amostral": _sha256_amostral(p),
            "sha256_completo": (
                _sha256_completo(p, cfg.inventario_chunk_mb)
                if fazer_completo else ""
            ),
            "hash_completo_calculado": bool(fazer_completo),
        })

    df = pd.DataFrame(linhas)
    df.to_csv(cfg.dir_inventario / "inventario_completo.csv", index=False)

    filtros = {
        "arquivos_0506_0603.csv": df[
            df["aquisicao"].isin([cfg.aquisicao_mestre, cfg.aquisicao_escrava])
        ],
        "candidatos_chm_lidar.csv": df[
            df["categoria"].isin(["candidato_chm", "dsm", "dtm"])
        ],
        "candidatos_geometria.csv": df[
            df["categoria"].isin([
                "altura_ambiguidade", "kz", "baseline",
                "geometria_interferometrica", "trajetoria_adaptada",
            ])
        ],
        "candidatos_biomassa.csv": df[
            df["categoria"].isin(["biomassa_in_situ", "resumo_in_situ"])
        ],
        "relatorios_e_formatos.csv": df[
            df["categoria"].isin(["relatorio_tropisar", "roi_analysis"])
        ],
        "cenas_landsat_inventario.csv": df[df["categoria"] == "landsat"],
        "manifesto_sha256.csv": df[
            ["caminho_relativo", "tamanho_bytes", "sha256_amostral",
             "sha256_completo", "hash_completo_calculado"]
        ],
    }
    for nome, sub in filtros.items():
        sub.to_csv(cfg.dir_inventario / nome, index=False)

    resumo = {
        "pipeline_version": VERSAO_PIPELINE,
        "raiz": str(raiz),
        "n_arquivos": int(len(df)),
        "arquivos_auxiliares_macos_ignorados": int(len(auxiliares_macos)),
        "bytes_totais": int(df["tamanho_bytes"].sum()) if len(df) else 0,
        "categorias": (
            {str(k): int(v) for k, v in df["categoria"].value_counts().items()}
            if len(df) else {}
        ),
        "candidatos_chm_nome": int((df["categoria"] == "candidato_chm").sum()) if len(df) else 0,
        "dsm": int((df["categoria"] == "dsm").sum()) if len(df) else 0,
        "dtm": int((df["categoria"] == "dtm").sum()) if len(df) else 0,
        "arquivos_0506_0603": int(
            df["aquisicao"].isin([cfg.aquisicao_mestre, cfg.aquisicao_escrava]).sum()
        ) if len(df) else 0,
        "hash_completo_todos": bool(completo_todos),
        "observacao_hash": (
            "Arquivos AppleDouble do macOS são ignorados. sha256_amostral não substitui o hash completo; arquivos críticos e "
            "arquivos abaixo do limite recebem SHA-256 completo automaticamente."
        ),
    }
    (cfg.dir_inventario / "resumo_inventario.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nInventário concluído:")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"Saídas: {cfg.dir_inventario}")
    return df


def _pontuar_candidato_sar(
    path: Path,
    banda: str,
    aquisicao: str,
    tipo: str,
    pol: Optional[str] = None,
) -> int:
    """Pontua somente depois do filtro por banda e pasta temporal exata."""
    if _arquivo_auxiliar_macos(path):
        return -10_000

    texto = str(path).lower().replace("-", "_")
    nome = path.name.lower()
    cons = "lcons" if banda == "SAR_L" else "pcons"

    score = 0
    if _esta_na_pasta_temporal_da_aquisicao(path, aquisicao):
        score += 1000
    if _esta_na_banda(path, banda):
        score += 500
    if "tomo_data" in texto:
        score -= 1000

    if tipo == "slc":
        if aquisicao.lower() in nome:
            score += 100
        if cons in nome:
            score += 80
        if "slc" in nome:
            score += 60
        if path.suffix.lower() == ".dat":
            score += 30
        if pol and re.search(
            rf"(?:_|-){pol.lower()}(?:_|-|\.|$)",
            nome,
        ):
            score += 100
    elif tipo == "grille":
        if path.name.lower().endswith(".grille"):
            score += 160
        if aquisicao.lower() in nome:
            score += 100
        if cons in nome:
            score += 80
        if "slc" in nome:
            score += 40
    elif tipo in {"roi", "lut"}:
        # ROI é controle externo: latitude, longitude, altitude, linha, coluna.
        if "roi_paracou" in nome:
            score += 120
        if path.suffix.lower() == ".txt":
            score += 30
        if cons in nome:
            score += 20
    elif tipo == "ha":
        if "_ha" in nome or re.search(
            r"(?:^|[_-])ha(?:[._-]|$)", nome
        ):
            score += 100
        if "0007" in nome:
            score += 40
        if aquisicao.replace("tropi", "") in nome:
            score += 80
        if path.suffix == ".1":
            score += 30

    return score


def _carregar_catalogo_inventario(
    cfg: ConfigDados,
) -> Optional[pd.DataFrame]:
    """Carrega o catálogo e filtra metadados auxiliares, inclusive em CSV antigo."""
    caminho = cfg.dir_inventario / "inventario_completo.csv"
    if not caminho.exists():
        return None

    df = pd.read_csv(caminho)
    if "caminho_relativo" not in df.columns:
        return None

    df = df.copy()
    rel = df["caminho_relativo"].astype(str)
    mascara_real = ~rel.map(
        lambda x: _arquivo_auxiliar_macos(Path(x))
    )
    df = df.loc[mascara_real].copy()

    df["caminho_absoluto"] = df["caminho_relativo"].map(
        lambda x: str((cfg.dir_brutos / str(x)).resolve())
    )
    df["existe_agora"] = df["caminho_absoluto"].map(
        lambda x: _arquivo_cientifico_real(Path(x))
    )
    return df[df["existe_agora"]].reset_index(drop=True)


def resolver_entrada_sar_bruta(
    cfg: ConfigDados,
    banda: str,
    aquisicao: str,
    tipo: str,
    pol: Optional[str] = None,
    catalogo: Optional[pd.DataFrame] = None,
) -> Path:
    """Resolve a entrada pelo local físico, sem confiar no nome da LUT.

    Regra canônica:
    `Paracou/<L_Band|P_Band>/temporal_data/<aquisicao>/...`

    Arquivos de `tomo_data`, outras aquisições e AppleDouble são excluídos.
    """
    sufixo = "Lcons" if banda == "SAR_L" else "Pcons"

    if tipo == "slc":
        padroes = ("*.dat", "*.bin", "*.slc")
        categorias = {"slc"}
    elif tipo == "grille":
        padroes = ("*.grille",)
        categorias = {"grille", "metadado"}
    elif tipo in {"roi", "lut"}:
        padroes = ("*.txt",)
        categorias = {"lut", "metadado"}
    elif tipo == "ha":
        padroes = ("*Ha*", "*ha*")
        categorias = {"altura_ambiguidade"}
    else:
        raise ValueError(f"Tipo de entrada SAR desconhecido: {tipo}")

    if catalogo is None:
        catalogo = _carregar_catalogo_inventario(cfg)

    candidatos: list[Path] = []
    if catalogo is not None and len(catalogo):
        sub = catalogo
        if "categoria" in sub.columns:
            sub = sub[sub["categoria"].astype(str).isin(categorias)]
        candidatos = [
            Path(x) for x in sub["caminho_absoluto"].astype(str).tolist()
        ]
    else:
        print(
            "  Aviso: inventario_completo.csv não encontrado; "
            "usando busca recursiva de fallback.",
            flush=True,
        )
        for padrao in padroes:
            candidatos.extend(
                p for p in cfg.dir_brutos.rglob(padrao)
                if _arquivo_cientifico_real(p)
            )

    candidatos = sorted({
        p.resolve()
        for p in candidatos
        if _arquivo_cientifico_real(p)
        and _esta_na_banda(p, banda)
        and _esta_na_pasta_temporal_da_aquisicao(p, aquisicao)
        and not _contem_componente(p, "tomo_data")
    })

    if tipo == "slc":
        candidatos = [
            p for p in candidatos
            if "slc" in p.name.lower()
            and (
                p.name.lower().endswith(".dat")
                or p.suffix.lower() in {".bin", ".slc"}
            )
            and sufixo.lower() in p.name.lower()
            and aquisicao.lower() in p.name.lower()
            and pol is not None
            and re.search(
                rf"(?:_|-){pol.lower()}(?:_|-|\.|$)",
                p.name.lower(),
            )
        ]
    elif tipo == "grille":
        candidatos = [
            p for p in candidatos
            if p.name.lower().endswith(".grille")
            and aquisicao.lower() in p.name.lower()
            and sufixo.lower() in p.name.lower()
            and "slc" in p.name.lower()
        ]
    elif tipo in {"roi", "lut"}:
        candidatos = [
            p for p in candidatos
            if "roi_paracou" in p.name.lower()
            and p.suffix.lower() == ".txt"
        ]
    elif tipo == "ha":
        codigo = aquisicao.replace("tropi", "")
        candidatos = [
            p for p in candidatos
            if (
                "_ha" in p.name.lower()
                or re.search(
                    r"(?:^|[_-])ha(?:[._-]|$)",
                    p.name.lower(),
                )
            )
            and codigo in p.name.lower()
        ]

    if not candidatos:
        raise FileNotFoundError(
            "Nenhum candidato real encontrado na pasta canônica "
            f"{banda}/temporal_data/{aquisicao} para "
            f"{tipo} {pol or ''}."
        )

    avaliados = sorted(
        (
            _pontuar_candidato_sar(
                p, banda, aquisicao, tipo, pol
            ),
            p,
        )
        for p in candidatos
    )
    melhor_score = avaliados[-1][0]
    melhores = [
        p for score, p in avaliados if score == melhor_score
    ]

    if len(melhores) != 1:
        raise RuntimeError(
            f"Resolução ainda ambígua para "
            f"{tipo} {banda} {aquisicao} {pol or ''}. "
            f"Candidatos reais na pasta temporal correta:\n"
            + "\n".join(map(str, melhores))
        )
    return melhores[0]


def inventariar_arquivos(cfg: ConfigDados) -> pd.DataFrame:
    """Monta a tabela das entradas mínimas reutilizando o catálogo integral."""
    linhas = []
    catalogo = _carregar_catalogo_inventario(cfg)

    if catalogo is not None:
        print(
            f"  Catálogo integral carregado: {len(catalogo)} arquivos.",
            flush=True,
        )
    else:
        print(
            "  Catálogo integral não encontrado. "
            "A auditoria usará fallback recursivo.",
            flush=True,
        )

    try:
        dsm, dtm = localizar_dsm_dtm_originais(cfg)
    except Exception as exc:
        dsm = cfg.dir_lidar_original / "DSM_AUSENTE.tif"
        dtm = cfg.dir_lidar_original / "DTM_AUSENTE.asc"
        erro_lidar = repr(exc)
    else:
        erro_lidar = ""

    for tipo, p in (("DSM_original", dsm), ("DTM_original", dtm)):
        linhas.append({
            "tipo": tipo,
            "banda": "-",
            "aquisicao": "Paracou_April_2009",
            "arquivo": str(p),
            "existe": p.exists(),
            "MB": p.stat().st_size / 2**20 if p.exists() else np.nan,
            "erro_resolucao": erro_lidar,
        })
        print(
            f"  {tipo}: {'OK' if p.exists() else 'AUSENTE'} | {p}",
            flush=True,
        )

    linhas.append({
        "tipo": "CHM_processado",
        "banda": "-",
        "aquisicao": "derivado_DSM_DTM",
        "arquivo": str(cfg.caminho_chm_canonico),
        "existe": cfg.caminho_chm_canonico.exists(),
        "MB": (
            cfg.caminho_chm_canonico.stat().st_size / 2**20
            if cfg.caminho_chm_canonico.exists() else np.nan
        ),
        "obrigatorio_na_auditoria_inicial": False,
        "erro_resolucao": "",
    })

    cenas = descobrir_cenas_landsat(cfg)
    for cena in cenas:
        for banda in (1, 2, 3, 4, 5, 7):
            p = achar_arquivo_unico(
                cena, (f"*_SR_B{banda}.TIF", f"*B{banda}.TIF")
            )
            linhas.append({
                "tipo": "Landsat_SR",
                "banda": f"B{banda}",
                "aquisicao": cena.name,
                "arquivo": (
                    str(p) if p else
                    str(cena / f"AUSENTE_SR_B{banda}.TIF")
                ),
                "existe": bool(p and p.exists()),
                "MB": (
                    p.stat().st_size / 2**20
                    if p and p.exists() else np.nan
                ),
                "erro_resolucao": "",
            })
        qa = achar_arquivo_unico(
            cena, ("*_QA_PIXEL.TIF", "*QA_PIXEL*.TIF")
        )
        linhas.append({
            "tipo": "Landsat_QA_PIXEL",
            "banda": "QA",
            "aquisicao": cena.name,
            "arquivo": (
                str(qa) if qa else
                str(cena / "AUSENTE_QA_PIXEL.TIF")
            ),
            "existe": bool(qa and qa.exists()),
            "MB": (
                qa.stat().st_size / 2**20
                if qa and qa.exists() else np.nan
            ),
            "erro_resolucao": "",
        })

    total_logico = 2 * len(cfg.aquisicoes_ortho) * (1 + len(POLARIZACOES))
    contador = 0
    for banda_dir in ("SAR_L", "SAR_P"):
        for aquisicao in cfg.aquisicoes_ortho:
            contador += 1
            print(
                f"  [{contador:02d}/{total_logico:02d}] "
                f"Resolvendo GRILLE {banda_dir}/{aquisicao}...",
                flush=True,
            )
            try:
                lut = resolver_entrada_sar_bruta(
                    cfg, banda_dir, aquisicao, "grille",
                    catalogo=catalogo,
                )
                erro = ""
            except Exception as exc:
                lut = (
                    cfg.dir_brutos
                    / f"AUSENTE_{banda_dir}_{aquisicao}_SLC.grille"
                )
                erro = repr(exc)

            linhas.append({
                "tipo": "GRILLE",
                "banda": banda_dir,
                "aquisicao": aquisicao,
                "arquivo": str(lut),
                "existe": lut.exists(),
                "MB": (
                    lut.stat().st_size / 2**20
                    if lut.exists() else np.nan
                ),
                "erro_resolucao": erro,
            })
            print(
                f"      {'OK' if lut.exists() else 'FALHA'} | {lut}",
                flush=True,
            )

            for pol in POLARIZACOES:
                contador += 1
                print(
                    f"  [{contador:02d}/{total_logico:02d}] "
                    f"Resolvendo SLC {banda_dir}/{aquisicao}/{pol}...",
                    flush=True,
                )
                try:
                    p = resolver_entrada_sar_bruta(
                        cfg, banda_dir, aquisicao, "slc", pol,
                        catalogo=catalogo,
                    )
                    erro = ""
                except Exception as exc:
                    p = (
                        cfg.dir_brutos
                        / f"AUSENTE_{banda_dir}_{aquisicao}_{pol}_SLC.dat"
                    )
                    erro = repr(exc)

                linhas.append({
                    "tipo": "SLC",
                    "banda": banda_dir,
                    "aquisicao": aquisicao,
                    "arquivo": str(p),
                    "existe": p.exists(),
                    "MB": (
                        p.stat().st_size / 2**20
                        if p.exists() else np.nan
                    ),
                    "erro_resolucao": erro,
                })
                print(
                    f"      {'OK' if p.exists() else 'FALHA'} | {p}",
                    flush=True,
                )
    return pd.DataFrame(linhas)


def procurar_metadados_fisicos(cfg: ConfigDados) -> list[Path]:
    """Lista metadados pelo inventário; evita novas varreduras do HD."""
    catalogo = _carregar_catalogo_inventario(cfg)
    extensoes = {".xml", ".hdr", ".ann", ".txt", ".json", ".par", ".cfg"}

    if catalogo is not None and len(catalogo):
        selecionados = []
        for caminho in catalogo["caminho_absoluto"].astype(str):
            p = Path(caminho)
            nome = p.name.lower()
            if (
                not _arquivo_auxiliar_macos(p)
                and (
                p.suffix.lower() in extensoes
                or "baseline" in nome
                or "kz" in nome
                )
            ):
                selecionados.append(p)
        return sorted(set(selecionados))

    print(
        "  Aviso: catálogo ausente ao procurar metadados; "
        "usando uma única busca recursiva.",
        flush=True,
    )
    padroes = (
        "*.xml", "*.hdr", "*.ann", "*.txt",
        "*.json", "*.par", "*.cfg", "*baseline*", "*kz*",
    )
    achados = []
    for padrao in padroes:
        achados.extend(cfg.dir_brutos.rglob(padrao))
    return sorted({
        p.resolve() for p in achados
        if _arquivo_cientifico_real(p)
    })


def auditar_ambiente(cfg: ConfigDados = CFG.dados, estrito: bool = True) -> pd.DataFrame:
    print("="*78)
    print("AUDITORIA PRÉVIA TCCT v2")
    print("="*78)
    print(f"Base: {cfg.raiz}")
    df = inventariar_arquivos(cfg)
    if "obrigatorio_na_auditoria_inicial" in df.columns:
        nao_obrigatorio = (
            df["obrigatorio_na_auditoria_inicial"]
            .eq(False)
            .fillna(False)
        )
        obrigatorio = ~nao_obrigatorio
    else:
        obrigatorio = pd.Series(True, index=df.index, dtype=bool)
    faltantes = df.loc[(~df["existe"]) & obrigatorio]
    print(df.groupby(["tipo", "existe"]).size())
    if len(faltantes):
        print("\nArquivos obrigatórios ausentes ou não resolvidos:")
        colunas = [
            c for c in (
                "tipo", "banda", "aquisicao", "arquivo", "erro_resolucao"
            )
            if c in faltantes.columns
        ]
        print(faltantes[colunas].to_string(index=False))
        if estrito:
            raise FileNotFoundError(
                "A auditoria encontrou entradas obrigatórias ausentes, "
                "ambíguas ou não resolvidas."
            )

    metadados = procurar_metadados_fisicos(cfg)
    print(f"\nMetadados/cabeçalhos encontrados: {len(metadados)}")
    for p in metadados[:20]:
        print(" -", p)
    if len(metadados) > 20:
        print(f" ... e mais {len(metadados)-20}")

    cfg.dir_resultados_v2.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.dir_resultados_v2 / "inventario_arquivos.csv", index=False)
    with (cfg.dir_resultados_v2 / "config_pipeline.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)
    return df

# Execute antes das etapas pesadas:
# inventario = auditar_ambiente(estrito=True)






# ================================================================
# 2. BACK-GEOCODING DOS SLCs — PAR DEFINITIVO 0506–0603
# ================================================================
# Função legada ler_lut_robusto removida na v3.2.0: a geometria canônica usa .grille 3D.



def _candidatos_metadado_ent(caminho_slc: Path) -> list[Path]:
    """Localiza apenas cabeçalhos plausivelmente associados ao SLC."""
    nome = caminho_slc.name
    stem = caminho_slc.stem
    candidatos = {
        caminho_slc.with_suffix(".ent"),
        caminho_slc.parent / f"{stem}.ent",
        caminho_slc.parent / nome.replace("_slc.dat", "_slc.ent"),
        caminho_slc.parent / nome.replace("_slc.dat", ".ent"),
    }

    prefixo_sem_pol = re.sub(
        r"_(?:Hh|Hv|Vh|Vv)_slc\.dat$",
        "",
        nome,
        flags=re.I,
    )
    candidatos.add(caminho_slc.parent / f"{prefixo_sem_pol}.ent")

    reais = [
        p.resolve()
        for p in candidatos
        if p.exists() and _arquivo_cientifico_real(p)
    ]
    return sorted(set(reais))


def ler_dimensoes_ent(caminho_ent: Path) -> tuple[int, int]:
    """Lê linhas e colunas de um cabeçalho textual ONERA .ent."""
    texto = caminho_ent.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    padroes_linhas = (
        r"Nb_ligne_look\s*=\s*([0-9]+)",
        r"Nb_ligne\s*=\s*([0-9]+)",
        r"(?:nrows|rows|lines)\s*[=:]\s*([0-9]+)",
    )
    padroes_colunas = (
        r"Nb_case_par_ligne_look\s*=\s*([0-9]+)",
        r"Nb_case_par_ligne\s*=\s*([0-9]+)",
        r"(?:ncols|cols|samples|width)\s*[=:]\s*([0-9]+)",
    )

    linhas = None
    colunas = None

    for padrao in padroes_linhas:
        m = re.search(padrao, texto, flags=re.I)
        if m:
            linhas = int(m.group(1))
            break

    for padrao in padroes_colunas:
        m = re.search(padrao, texto, flags=re.I)
        if m:
            colunas = int(m.group(1))
            break

    if not linhas or not colunas:
        raise ValueError(
            f"Cabeçalho .ent sem dimensões reconhecíveis: {caminho_ent}"
        )
    return linhas, colunas


# Função legada inferir_layout_slc removida na v3.2.0: a geometria canônica usa .grille 3D.



def inferir_shape_slc(
    caminho: Path,
    colunas: int,
) -> tuple[int, int]:
    """Compatibilidade: retorna linhas e offset, aceitando pixels órfãos."""
    layout = inferir_layout_slc(caminho, colunas)
    return layout["linhas"], layout["offset_bytes"]


def _layout_slc_aprovado(banda: str, aquisicao: str) -> dict:
    chave = (banda, aquisicao)
    if chave not in LAYOUTS_SLC_APROVADOS:
        raise KeyError(f"Layout SLC não aprovado para {chave}.")
    item = LAYOUTS_SLC_APROVADOS[chave]
    return {
        "linhas": int(item["shape"][0]),
        "colunas": int(item["shape"][1]),
        "offset_bytes": int(item["offset_bytes"]),
    }


def ler_grille_estruturada(
    caminho: Path,
    crs_destino=None,
) -> dict:
    """Lê a grade 3D ONERA: linha, coluna, altitude, longitude, latitude."""
    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    linhas_texto = caminho.read_text(
        encoding="latin1",
        errors="replace",
    ).splitlines()
    if len(linhas_texto) < 7:
        raise RuntimeError(f"Arquivo .grille inválido: {caminho}")

    cabecalho = "\n".join(linhas_texto[:6])

    def obter_dimensao(chave: str) -> int:
        m = re.search(
            rf"{chave}\s+(\d+)",
            cabecalho,
            flags=re.IGNORECASE,
        )
        if m is None:
            raise RuntimeError(f"{chave} ausente em {caminho}")
        return int(m.group(1))

    n_linhas = obter_dimensao("nb_lig")
    n_colunas = obter_dimensao("nb_col")
    n_altitudes = obter_dimensao("nb_alt")

    dados = np.loadtxt(
        caminho,
        comments="%",
        skiprows=6,
        dtype=np.float64,
        encoding="latin1",
    )
    esperado = n_linhas * n_colunas * n_altitudes
    if dados.shape != (esperado, 5):
        raise RuntimeError(
            f"Shape inesperado em {caminho}: {dados.shape}; "
            f"esperado ({esperado}, 5)."
        )

    grade = dados.reshape(n_linhas, n_colunas, n_altitudes, 5)
    linhas_sar = grade[:, 0, 0, 0].copy()
    colunas_sar = grade[0, :, 0, 1].copy()
    altitudes = grade[0, 0, :, 2].copy()

    if not np.allclose(
        grade[:, :, :, 0], linhas_sar[:, None, None], rtol=0, atol=1e-9
    ):
        raise RuntimeError("Dimensão linha da .grille não está estruturada.")
    if not np.allclose(
        grade[:, :, :, 1], colunas_sar[None, :, None], rtol=0, atol=1e-9
    ):
        raise RuntimeError("Dimensão coluna da .grille não está estruturada.")
    if not np.allclose(
        grade[:, :, :, 2], altitudes[None, None, :], rtol=0, atol=1e-6
    ):
        raise RuntimeError("Dimensão altitude da .grille não está estruturada.")
    if np.any(np.diff(linhas_sar) <= 0) or np.any(np.diff(colunas_sar) <= 0):
        raise RuntimeError("Linhas/colunas de controle não são estritamente crescentes.")
    if np.any(np.diff(altitudes) <= 0):
        raise RuntimeError("Altitudes de controle não são estritamente crescentes.")

    lon = grade[:, :, :, 3]
    lat = grade[:, :, :, 4]
    valido = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon != 0.0)
        & (lat != 0.0)
    )
    if not valido.all():
        raise RuntimeError(
            f"A .grille contém coordenadas geográficas inválidas; "
            f"fração válida={valido.mean():.6f}."
        )

    resultado = {
        "arquivo": caminho,
        "cabecalho": cabecalho,
        "linhas_sar": linhas_sar,
        "colunas_sar": colunas_sar,
        "altitudes": altitudes,
        "longitude": lon,
        "latitude": lat,
        "n_registros": int(esperado),
    }

    if crs_destino is not None:
        transformer = Transformer.from_crs(
            "EPSG:4326",
            crs_destino,
            always_xy=True,
        )
        x, y = transformer.transform(lon.ravel(), lat.ravel())
        resultado["x"] = np.asarray(x, dtype=np.float64).reshape(lon.shape)
        resultado["y"] = np.asarray(y, dtype=np.float64).reshape(lat.shape)

    return resultado


def auditar_layouts_slc(
    cfg: ConfigDados = CFG.dados,
) -> pd.DataFrame:
    """Confirma os 16 SLCs contra layouts congelados e suas .grille."""
    registros = []
    bytes_complexo = np.dtype(np.complex64).itemsize

    for banda in cfg.bandas:
        for aquisicao in cfg.aquisicoes_ortho:
            layout = _layout_slc_aprovado(banda, aquisicao)
            grille_path = resolver_entrada_sar_bruta(
                cfg, banda, aquisicao, "grille"
            )
            grille = ler_grille_estruturada(grille_path)
            row_max = float(grille["linhas_sar"].max())
            col_max = float(grille["colunas_sar"].max())

            for pol in POLARIZACOES:
                slc = resolver_entrada_sar_bruta(
                    cfg, banda, aquisicao, "slc", pol
                )
                esperado = (
                    layout["offset_bytes"]
                    + layout["linhas"] * layout["colunas"] * bytes_complexo
                )
                real = int(slc.stat().st_size)
                if real != esperado:
                    raise RuntimeError(
                        f"SLC incompatível com layout aprovado: {slc}\n"
                        f"real={real:,} bytes; esperado={esperado:,}; "
                        f"shape=({layout['linhas']},{layout['colunas']}), "
                        f"offset={layout['offset_bytes']}."
                    )
                registros.append({
                    "banda": banda,
                    "aquisicao": aquisicao,
                    "polarizacao": pol,
                    "slc": str(slc),
                    "grille": str(grille_path),
                    "linhas": layout["linhas"],
                    "colunas": layout["colunas"],
                    "offset_bytes": layout["offset_bytes"],
                    "tamanho_real_bytes": real,
                    "tamanho_esperado_bytes": esperado,
                    "grille_nb_lig": len(grille["linhas_sar"]),
                    "grille_nb_col": len(grille["colunas_sar"]),
                    "grille_nb_alt": len(grille["altitudes"]),
                    "grille_row_max": row_max,
                    "grille_col_max": col_max,
                    "aprovado": True,
                })

    df = pd.DataFrame(registros)
    out = cfg.dir_auditoria / "BackGeocoding_Grille3D"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "layouts_16_slcs_aprovados.csv", index=False)
    (out / "layouts_16_slcs_aprovados.json").write_text(
        json.dumps(
            {
                "pipeline_version": VERSAO_PIPELINE,
                "layouts": df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nLayouts SLC aprovados:")
    print(
        df[[
            "banda", "aquisicao", "polarizacao",
            "linhas", "colunas", "offset_bytes",
        ]].to_string(index=False)
    )
    return df


_PADRAO_FLOAT_ROI = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


def ler_roi_geodesico(caminho: Path) -> pd.DataFrame:
    """Lê ROI como latitude, longitude, altitude, linha SAR, coluna SAR."""
    registros = []
    texto = caminho.read_text(encoding="latin1", errors="replace")
    for numero, linha in enumerate(texto.splitlines(), start=1):
        if linha.lstrip().startswith("*"):
            continue
        valores = [float(v) for v in _PADRAO_FLOAT_ROI.findall(linha)]
        if len(valores) < 5:
            continue
        lat, lon, alt, row_sar, col_sar = valores[:5]
        if not (
            3.0 <= lat <= 7.0
            and -56.0 <= lon <= -50.0
            and np.isfinite(alt)
            and np.isfinite(row_sar)
            and np.isfinite(col_sar)
        ):
            continue
        registros.append({
            "linha_arquivo": numero,
            "latitude": lat,
            "longitude": lon,
            "altura_roi_m": alt,
            "row_sar": row_sar,
            "col_sar": col_sar,
        })
    if not registros:
        raise RuntimeError(f"Nenhum ponto ROI válido em {caminho}")
    return pd.DataFrame(registros)


def garantir_grade_geoidal_guyana() -> Path:
    """Localiza ou baixa com curl a grade oficial usada pelo PROJ."""
    pasta = Path(datadir.get_user_data_dir(create=True))
    caminho = pasta / NOME_GRADE_GEOIDAL_GUYANA
    pasta.mkdir(parents=True, exist_ok=True)

    def valida(path: Path) -> bool:
        try:
            with rasterio.open(path) as src:
                return src.width > 0 and src.height > 0 and src.count > 0
        except Exception:
            return False

    if caminho.exists() and not valida(caminho):
        caminho.unlink()

    if not caminho.exists():
        curl = Path("/usr/bin/curl")
        if not curl.exists():
            raise FileNotFoundError(
                "Grade geoidal ausente e /usr/bin/curl indisponível. "
                f"Baixe {URL_GRADE_GEOIDAL_GUYANA} para {caminho}."
            )
        subprocess.run(
            [
                str(curl), "--fail", "--location", "--show-error",
                "--retry", "3", "--connect-timeout", "30",
                "--output", str(caminho), URL_GRADE_GEOIDAL_GUYANA,
            ],
            check=True,
        )
    if not valida(caminho):
        raise RuntimeError(f"Grade geoidal inválida: {caminho}")
    datadir.append_data_dir(str(pasta))
    return caminho


def transformador_ngg1977_para_elipsoidal() -> tuple[object, Path]:
    grade = garantir_grade_geoidal_guyana()
    grupo = TransformerGroup(
        PyprojCRS.from_epsg(9530),
        PyprojCRS.from_epsg(4967),
        always_xy=True,
    )
    if not grupo.best_available or not grupo.transformers:
        raise RuntimeError(
            "Transformação vertical RGFG95+NGG1977 → RGFG95 3D "
            "não está disponível no PROJ."
        )
    return grupo.transformers[0], grade


def amostrar_raster_bilinear(
    src,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Amostra um raster em coordenadas do seu CRS usando centros de pixel."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inv = ~src.transform
    col_corner, row_corner = inv * (x, y)
    col = np.asarray(col_corner, dtype=np.float64) - 0.5
    row = np.asarray(row_corner, dtype=np.float64) - 0.5

    r0 = np.floor(row).astype(np.int64)
    c0 = np.floor(col).astype(np.int64)
    dr = row - r0
    dc = col - c0
    valido = (
        np.isfinite(row) & np.isfinite(col)
        & (r0 >= 0) & (c0 >= 0)
        & (r0 + 1 < src.height) & (c0 + 1 < src.width)
    )
    out = np.full(row.shape, np.nan, dtype=np.float64)
    if not valido.any():
        return out

    arr = src.read(1, masked=True).filled(np.nan).astype(np.float64)
    rv = r0[valido]
    cv = c0[valido]
    drv = dr[valido]
    dcv = dc[valido]
    v00 = arr[rv, cv]
    v01 = arr[rv, cv + 1]
    v10 = arr[rv + 1, cv]
    v11 = arr[rv + 1, cv + 1]
    fin = np.isfinite(v00) & np.isfinite(v01) & np.isfinite(v10) & np.isfinite(v11)
    vals = (
        (1 - drv) * (1 - dcv) * v00
        + (1 - drv) * dcv * v01
        + drv * (1 - dcv) * v10
        + drv * dcv * v11
    )
    idx = np.flatnonzero(valido)
    out[idx[fin]] = vals[fin]
    return out


def determinar_modo_vertical_dtm(
    cfg: ConfigDados = CFG.dados,
    sobrescrever: Optional[bool] = None,
) -> dict:
    """Escolhe DTM direto ou convertido usando ROI, sem usar CHM/modelo."""
    sobrescrever = cfg.sobrescrever if sobrescrever is None else bool(sobrescrever)
    out = cfg.dir_auditoria / "BackGeocoding_Grille3D"
    out.mkdir(parents=True, exist_ok=True)
    rel_path = out / "decisao_datum_vertical_dtm.json"
    if rel_path.exists() and not sobrescrever:
        rel = json.loads(rel_path.read_text(encoding="utf-8"))
        if rel.get("pipeline_version") == VERSAO_PIPELINE:
            return rel

    if not cfg.caminho_dtm_alinhado.exists():
        raise FileNotFoundError(
            f"DTM alinhado ausente: {cfg.caminho_dtm_alinhado}. "
            "Execute primeiro a preparação do CHM."
        )

    roi_path = resolver_entrada_sar_bruta(
        cfg, "SAR_L", cfg.aquisicao_mestre, "roi"
    )
    roi = ler_roi_geodesico(roi_path)

    with rasterio.open(cfg.caminho_dtm_alinhado) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = tr.transform(
            roi["longitude"].to_numpy(float),
            roi["latitude"].to_numpy(float),
        )
        dtm_direto = amostrar_raster_bilinear(src, x, y)

    lon = roi["longitude"].to_numpy(float)
    lat = roi["latitude"].to_numpy(float)
    alt_roi = roi["altura_roi_m"].to_numpy(float)
    transformador_vertical, grade_path = transformador_ngg1977_para_elipsoidal()
    _, _, dtm_convertido = transformador_vertical.transform(lon, lat, dtm_direto)
    dtm_convertido = np.asarray(dtm_convertido, dtype=np.float64)

    valido = (
        np.isfinite(dtm_direto)
        & np.isfinite(dtm_convertido)
        & np.isfinite(alt_roi)
    )
    n = int(valido.sum())
    if n < cfg.backgeo_min_pontos_roi_dtm:
        raise RuntimeError(
            f"Apenas {n} pontos ROI possuem DTM válido; mínimo="
            f"{cfg.backgeo_min_pontos_roi_dtm}."
        )

    def metricas(pred):
        res = pred[valido] - alt_roi[valido]
        absr = np.abs(res)
        return {
            "n": n,
            "bias_m": float(np.mean(res)),
            "rmse_m": float(np.sqrt(np.mean(res**2))),
            "mediana_abs_m": float(np.median(absr)),
            "p95_abs_m": float(np.percentile(absr, 95)),
        }

    candidatos = {
        "direto": metricas(dtm_direto),
        "ngg1977_para_elipsoidal": metricas(dtm_convertido),
    }
    separacao = dtm_convertido[valido] - dtm_direto[valido]
    separacao_mediana_abs = float(np.median(np.abs(separacao)))

    modo_cfg = cfg.backgeo_modo_vertical_dtm
    if modo_cfg in {"direto", "ngg1977_para_elipsoidal"}:
        escolhido = modo_cfg
        origem = "override_configuracao"
    elif modo_cfg == "auto_roi":
        ordem = sorted(candidatos, key=lambda k: candidatos[k]["mediana_abs_m"])
        escolhido, segundo = ordem[0], ordem[1]
        ganho = (
            candidatos[segundo]["mediana_abs_m"]
            - candidatos[escolhido]["mediana_abs_m"]
        )
        ganho_min = (
            cfg.backgeo_margem_decisao_vertical_fracao_geoide
            * separacao_mediana_abs
        )
        residuo_max = (
            cfg.backgeo_max_residuo_vertical_fracao_geoide
            * separacao_mediana_abs
        )
        if ganho < ganho_min or candidatos[escolhido]["mediana_abs_m"] > residuo_max:
            raise RuntimeError(
                "Decisão vertical ambígua. Não será escolhida por heurística. "
                f"Candidatos={candidatos}; separação mediana="
                f"{separacao_mediana_abs:.3f} m; ganho={ganho:.3f} m. "
                "Revise o DTM/ROI ou defina backgeo_modo_vertical_dtm explicitamente."
            )
        origem = "controle_externo_roi"
    else:
        raise ValueError(
            "backgeo_modo_vertical_dtm deve ser auto_roi, direto ou "
            "ngg1977_para_elipsoidal."
        )

    rel = {
        "pipeline_version": VERSAO_PIPELINE,
        "roi": str(roi_path),
        "dtm_alinhado": str(cfg.caminho_dtm_alinhado),
        "grade_geoidal": str(grade_path),
        "transformacao_vertical": transformador_vertical.description,
        "modo_configurado": modo_cfg,
        "modo_escolhido": escolhido,
        "origem_decisao": origem,
        "n_pontos_validos": n,
        "separacao_convertido_menos_direto": {
            "min_m": float(np.min(separacao)),
            "mediana_m": float(np.median(separacao)),
            "max_m": float(np.max(separacao)),
            "mediana_abs_m": separacao_mediana_abs,
        },
        "candidatos": candidatos,
    }
    rel_path.write_text(
        json.dumps(rel, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nDatum vertical do DTM decidido:")
    print(json.dumps(rel, indent=2, ensure_ascii=False))
    return rel


def preparar_dtm_para_grille(
    cfg: ConfigDados = CFG.dados,
    sobrescrever: Optional[bool] = None,
) -> tuple[Path, dict]:
    """Materializa o DTM na referência vertical usada pelas .grille."""
    sobrescrever = cfg.sobrescrever if sobrescrever is None else bool(sobrescrever)
    rel_decisao = determinar_modo_vertical_dtm(cfg, sobrescrever=sobrescrever)
    modo = rel_decisao["modo_escolhido"]
    saida = cfg.caminho_dtm_para_grille
    report_path = cfg.dir_backgeo_grille3d / "relatorio_dtm_para_grille.json"
    if saida.exists() and report_path.exists() and not sobrescrever:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            rep.get("pipeline_version") == VERSAO_PIPELINE
            and rep.get("modo_vertical") == modo
        ):
            return saida, rep

    saida.parent.mkdir(parents=True, exist_ok=True)
    transformador_vertical = None
    grade_path = None
    if modo == "ngg1977_para_elipsoidal":
        transformador_vertical, grade_path = transformador_ngg1977_para_elipsoidal()

    with rasterio.open(cfg.caminho_dtm_alinhado) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=np.nan, count=1, compress="deflate")
        with rasterio.open(saida, "w", **profile) as dst:
            tr_lonlat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            for r0 in tqdm(
                range(0, src.height, cfg.backgeo_chunk_linhas),
                desc="Preparando DTM para .grille",
            ):
                r1 = min(src.height, r0 + cfg.backgeo_chunk_linhas)
                win = Window(0, r0, src.width, r1 - r0)
                arr = src.read(1, window=win, masked=True).filled(np.nan).astype(np.float64)
                if modo == "ngg1977_para_elipsoidal":
                    rr, cc = np.meshgrid(
                        np.arange(r0, r1, dtype=np.float64) + 0.5,
                        np.arange(src.width, dtype=np.float64) + 0.5,
                        indexing="ij",
                    )
                    t = src.transform
                    x = t.a * cc + t.b * rr + t.c
                    y = t.d * cc + t.e * rr + t.f
                    lon, lat = tr_lonlat.transform(x.ravel(), y.ravel())
                    _, _, h = transformador_vertical.transform(lon, lat, arr.ravel())
                    arr = np.asarray(h, dtype=np.float64).reshape(arr.shape)
                dst.write(arr.astype(np.float32), 1, window=win)

    with rasterio.open(saida) as src:
        am = src.read(1, masked=True).filled(np.nan)
        vals = am[np.isfinite(am)]
        estat = {
            "n": int(vals.size),
            "min": float(np.min(vals)),
            "mediana": float(np.median(vals)),
            "max": float(np.max(vals)),
        }
    rep = {
        "pipeline_version": VERSAO_PIPELINE,
        "entrada": str(cfg.caminho_dtm_alinhado),
        "saida": str(saida),
        "modo_vertical": modo,
        "grade_geoidal": str(grade_path) if grade_path else None,
        "decisao": rel_decisao,
        "estatisticas": estat,
    }
    report_path.write_text(
        json.dumps(rep, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return saida, rep


def construir_inversores_superficies(estrutura: dict):
    linhas = estrutura["linhas_sar"]
    colunas = estrutura["colunas_sar"]
    x = estrutura["x"]
    y = estrutura["y"]
    valores_linha = np.repeat(linhas, len(colunas))
    valores_coluna = np.tile(colunas, len(linhas))
    inversores_linha = []
    inversores_coluna = []
    for k in range(len(estrutura["altitudes"])):
        pontos = np.column_stack([x[:, :, k].ravel(), y[:, :, k].ravel()])
        validos = np.isfinite(pontos).all(axis=1)
        triangulacao = Delaunay(pontos[validos])
        inversores_linha.append(
            LinearNDInterpolator(
                triangulacao, valores_linha[validos], fill_value=np.nan
            )
        )
        inversores_coluna.append(
            LinearNDInterpolator(
                triangulacao, valores_coluna[validos], fill_value=np.nan
            )
        )
    return inversores_linha, inversores_coluna


def _caminhos_mapa_inverso(
    cfg: ConfigDados,
    banda: str,
    aquisicao: str,
    finalidade: str = "slc",
) -> dict[str, Path]:
    token = f"{banda}_{aquisicao}_{finalidade}".replace("SAR_", "")
    pasta = cfg.dir_geometria_sar / token
    return {
        "dir": pasta,
        "row": pasta / "row_sar_fracionaria.npy",
        "col": pasta / "col_sar_fracionaria.npy",
        "mask": pasta / "mascara_geometrica_valida.npy",
        "margin": pasta / "margem_borda_sar_px.npy",
        "mask_tif": pasta / "mascara_geometrica_valida.tif",
        "margin_tif": pasta / "margem_borda_sar_px.tif",
        "report": pasta / "relatorio_mapa_inverso.json",
    }


def preparar_mapa_inverso_grille(
    cfg: ConfigDados,
    banda: str,
    aquisicao: str,
    shape_radar: tuple[int, int],
    finalidade: str = "slc",
    grille_path: Optional[Path] = None,
    sobrescrever: Optional[bool] = None,
) -> dict:
    """Calcula (X,Y,h_DTM) → linha/coluna radar, sem extrapolação."""
    sobrescrever = cfg.sobrescrever if sobrescrever is None else bool(sobrescrever)
    caminhos = _caminhos_mapa_inverso(cfg, banda, aquisicao, finalidade)
    caminhos["dir"].mkdir(parents=True, exist_ok=True)
    if all(caminhos[k].exists() for k in ("row", "col", "mask", "margin", "report")) and not sobrescrever:
        rep = json.loads(caminhos["report"].read_text(encoding="utf-8"))
        if (
            rep.get("pipeline_version") == VERSAO_PIPELINE
            and tuple(rep.get("shape_radar", [])) == tuple(shape_radar)
        ):
            return {**caminhos, "report_data": rep}

    dtm_path, dtm_report = preparar_dtm_para_grille(cfg, sobrescrever=sobrescrever)
    if grille_path is None:
        grille_path = resolver_entrada_sar_bruta(
            cfg, banda, aquisicao, "grille"
        )

    with rasterio.open(dtm_path) as ref:
        estrutura = ler_grille_estruturada(grille_path, crs_destino=ref.crs)
        H, W = ref.height, ref.width
        perfil = ref.profile.copy()
        transform = ref.transform
        crs = ref.crs

        inv_rows, inv_cols = construir_inversores_superficies(estrutura)
        alts = estrutura["altitudes"]
        n_alt = len(alts)

        row_out = np.lib.format.open_memmap(
            caminhos["row"], mode="w+", dtype=np.float32, shape=(H, W)
        )
        col_out = np.lib.format.open_memmap(
            caminhos["col"], mode="w+", dtype=np.float32, shape=(H, W)
        )
        mask_out = np.lib.format.open_memmap(
            caminhos["mask"], mode="w+", dtype=np.bool_, shape=(H, W)
        )
        margin_out = np.lib.format.open_memmap(
            caminhos["margin"], mode="w+", dtype=np.float32, shape=(H, W)
        )
        row_out[:] = np.nan
        col_out[:] = np.nan
        mask_out[:] = False
        margin_out[:] = np.nan

        total_validos = 0
        total_vertical = 0
        for r0 in tqdm(
            range(0, H, cfg.backgeo_chunk_linhas),
            desc=f"Invertendo .grille {banda}/{aquisicao}/{finalidade}",
        ):
            r1 = min(H, r0 + cfg.backgeo_chunk_linhas)
            win = Window(0, r0, W, r1 - r0)
            h = ref.read(1, window=win, masked=True).filled(np.nan).astype(np.float64)
            rr, cc = np.meshgrid(
                np.arange(r0, r1, dtype=np.float64) + 0.5,
                np.arange(W, dtype=np.float64) + 0.5,
                indexing="ij",
            )
            x = transform.a * cc + transform.b * rr + transform.c
            y = transform.d * cc + transform.e * rr + transform.f
            xy = np.column_stack([x.ravel(), y.ravel()])
            hf = h.ravel()

            upper = np.searchsorted(alts, hf, side="right")
            vertical = (
                np.isfinite(hf)
                & (upper > 0)
                & (upper < n_alt)
            )
            total_vertical += int(vertical.sum())
            upper_safe = np.clip(upper, 1, n_alt - 1)
            lower = upper_safe - 1
            row_pred = np.full(hf.shape, np.nan, dtype=np.float64)
            col_pred = np.full(hf.shape, np.nan, dtype=np.float64)

            for k in np.unique(lower[vertical]):
                sel = vertical & (lower == k)
                if not sel.any():
                    continue
                pts = xy[sel]
                rl = np.asarray(inv_rows[int(k)](pts), dtype=np.float64)
                rh = np.asarray(inv_rows[int(k) + 1](pts), dtype=np.float64)
                cl = np.asarray(inv_cols[int(k)](pts), dtype=np.float64)
                ch = np.asarray(inv_cols[int(k) + 1](pts), dtype=np.float64)
                peso = (hf[sel] - alts[int(k)]) / (alts[int(k) + 1] - alts[int(k)])
                row_pred[sel] = (1.0 - peso) * rl + peso * rh
                col_pred[sel] = (1.0 - peso) * cl + peso * ch

            altura, largura = map(int, shape_radar)
            valido = (
                vertical
                & np.isfinite(row_pred)
                & np.isfinite(col_pred)
                & (row_pred >= 0.0)
                & (col_pred >= 0.0)
                & (row_pred < altura - 1.0)
                & (col_pred < largura - 1.0)
            )
            margem = np.full(hf.shape, np.nan, dtype=np.float64)
            margem[valido] = np.minimum.reduce([
                row_pred[valido],
                col_pred[valido],
                (altura - 1.0) - row_pred[valido],
                (largura - 1.0) - col_pred[valido],
            ])
            row_pred[~valido] = np.nan
            col_pred[~valido] = np.nan
            shape_chunk = (r1 - r0, W)
            row_out[r0:r1] = row_pred.reshape(shape_chunk).astype(np.float32)
            col_out[r0:r1] = col_pred.reshape(shape_chunk).astype(np.float32)
            mask_out[r0:r1] = valido.reshape(shape_chunk)
            margin_out[r0:r1] = margem.reshape(shape_chunk).astype(np.float32)
            total_validos += int(valido.sum())

        row_out.flush(); col_out.flush(); mask_out.flush(); margin_out.flush()
        del row_out, col_out, mask_out, margin_out, inv_rows, inv_cols
        gc.collect()

    mask = np.load(caminhos["mask"], mmap_mode="r")
    margem = np.load(caminhos["margin"], mmap_mode="r")
    perfil.update(dtype="uint8", nodata=0, count=1, compress="deflate")
    with rasterio.open(caminhos["mask_tif"], "w", **perfil) as dst:
        dst.write(np.asarray(mask, dtype=np.uint8), 1)
    perfil.update(dtype="float32", nodata=np.nan)
    with rasterio.open(caminhos["margin_tif"], "w", **perfil) as dst:
        dst.write(np.asarray(margem, dtype=np.float32), 1)

    margem_valida = np.asarray(margem[mask], dtype=np.float64)
    rep = {
        "pipeline_version": VERSAO_PIPELINE,
        "backgeo_schema_version": cfg.backgeo_schema_version,
        "banda": banda,
        "aquisicao": aquisicao,
        "finalidade": finalidade,
        "grille": str(grille_path),
        "dtm_para_grille": str(dtm_path),
        "modo_vertical": dtm_report["modo_vertical"],
        "shape_grade_saida": [int(mask.shape[0]), int(mask.shape[1])],
        "shape_radar": [int(shape_radar[0]), int(shape_radar[1])],
        "pixels_total": int(mask.size),
        "pixels_dentro_intervalo_vertical": int(total_vertical),
        "pixels_geometricamente_validos": int(total_validos),
        "fracao_geometricamente_valida": float(mask.mean()),
        "margem_borda_px": _quantis_amostra_distancia(margem_valida),
        "politica": "sem extrapolacao; bilinear exige quatro vizinhos dentro do raster radar",
        "arquivos": {k: str(v) for k, v in caminhos.items() if k != "dir"},
    }
    caminhos["report"].write_text(
        json.dumps(rep, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**caminhos, "report_data": rep}


def ortorretificar_slc(
    entrada: Path,
    saida: Path,
    mapa_rows_path: Path,
    mapa_cols_path: Path,
    mascara_path: Path,
    layout: dict,
    sobrescrever: bool = False,
    chunk_linhas: int = 128,
) -> None:
    """Reamostra SLC complexo por interpolação bilinear na grade LiDAR."""
    if saida.exists() and not sobrescrever:
        print(f"  ↪ Já existe; mantendo: {saida}", flush=True)
        return
    saida.parent.mkdir(parents=True, exist_ok=True)
    linhas = int(layout["linhas"])
    colunas = int(layout["colunas"])
    offset = int(layout["offset_bytes"])
    bruto = np.memmap(
        entrada,
        dtype=np.complex64,
        mode="r",
        offset=offset,
        shape=(linhas, colunas),
    )
    mapa_rows = np.load(mapa_rows_path, mmap_mode="r")
    mapa_cols = np.load(mapa_cols_path, mmap_mode="r")
    mascara = np.load(mascara_path, mmap_mode="r")
    out = np.lib.format.open_memmap(
        saida,
        mode="w+",
        dtype=np.complex64,
        shape=mapa_rows.shape,
    )
    out[:] = np.complex64(np.nan + 1j * np.nan)

    for rini in tqdm(
        range(0, mapa_rows.shape[0], chunk_linhas),
        desc=f"Reamostrando {saida.name}",
        leave=False,
    ):
        rfim = min(mapa_rows.shape[0], rini + chunk_linhas)
        rows = np.asarray(mapa_rows[rini:rfim], dtype=np.float64)
        cols = np.asarray(mapa_cols[rini:rfim], dtype=np.float64)
        valid = np.asarray(mascara[rini:rfim], dtype=bool)
        chunk = np.full(rows.shape, np.nan + 1j * np.nan, dtype=np.complex64)
        if valid.any():
            rv = rows[valid]
            cv = cols[valid]
            r0 = np.floor(rv).astype(np.int64)
            c0 = np.floor(cv).astype(np.int64)
            dr = rv - r0
            dc = cv - c0
            v00 = np.asarray(bruto[r0, c0], dtype=np.complex128)
            v01 = np.asarray(bruto[r0, c0 + 1], dtype=np.complex128)
            v10 = np.asarray(bruto[r0 + 1, c0], dtype=np.complex128)
            v11 = np.asarray(bruto[r0 + 1, c0 + 1], dtype=np.complex128)
            vals = (
                (1 - dr) * (1 - dc) * v00
                + (1 - dr) * dc * v01
                + dr * (1 - dc) * v10
                + dr * dc * v11
            )
            fin = np.isfinite(vals.real) & np.isfinite(vals.imag)
            idx = np.flatnonzero(valid)
            flat = chunk.ravel()
            flat[idx[fin]] = vals[fin].astype(np.complex64)
            chunk = flat.reshape(chunk.shape)
        out[rini:rfim] = chunk
    out.flush()
    del out, bruto, mapa_rows, mapa_cols, mascara
    gc.collect()


def executar_ortorretificacao_global(
    cfg: ConfigDados = CFG.dados,
) -> pd.DataFrame:
    """Back-geocoding das 16 imagens usando DTM + quatro .grille 3D."""
    preparar_dtm_para_grille(cfg)
    auditoria = auditar_layouts_slc(cfg)
    relatorios = []

    for banda in cfg.bandas:
        sufixo = "Lcons" if banda == "SAR_L" else "Pcons"
        for aquisicao in cfg.aquisicoes_ortho:
            layout = _layout_slc_aprovado(banda, aquisicao)
            mapa = preparar_mapa_inverso_grille(
                cfg,
                banda,
                aquisicao,
                (layout["linhas"], layout["colunas"]),
                finalidade="slc",
            )
            for pol in POLARIZACOES:
                entrada = resolver_entrada_sar_bruta(
                    cfg, banda, aquisicao, "slc", pol
                )
                saida = (
                    cfg.dir_backgeo_grille3d / banda / aquisicao
                    / f"{aquisicao}_{sufixo}_{pol}_slc_ORTORRETIFICADO_GRILLE3D.npy"
                )
                ortorretificar_slc(
                    entrada,
                    saida,
                    mapa["row"],
                    mapa["col"],
                    mapa["mask"],
                    layout,
                    sobrescrever=cfg.sobrescrever,
                    chunk_linhas=cfg.backgeo_chunk_linhas,
                )
                relatorios.append({
                    "banda": banda,
                    "aquisicao": aquisicao,
                    "polarizacao": pol,
                    "entrada": str(entrada),
                    "saida": str(saida),
                    "grille": mapa["report_data"]["grille"],
                    "mapa_row": str(mapa["row"]),
                    "mapa_col": str(mapa["col"]),
                    "mascara": str(mapa["mask"]),
                    "shape_slc": [layout["linhas"], layout["colunas"]],
                    "offset_bytes": layout["offset_bytes"],
                    "interpolacao": cfg.backgeo_interpolacao_slc,
                })

    df = pd.DataFrame(relatorios)
    out = cfg.dir_auditoria / "BackGeocoding_Grille3D"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "relatorio_backgeocoding_16_slcs.csv", index=False)
    (out / "relatorio_backgeocoding_16_slcs.json").write_text(
        json.dumps(
            {"pipeline_version": VERSAO_PIPELINE, "produtos": relatorios},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nBack-geocoding .grille 3D concluído para 16 produtos.")
    return df


def auditar_saidas_backgeocoding(
    cfg: ConfigDados = CFG.dados,
) -> pd.DataFrame:
    """Compara finitude de cada SLC com sua máscara geométrica completa."""
    with rasterio.open(cfg.caminho_dtm_para_grille) as ref:
        shape_esperado = (ref.height, ref.width)
    registros = []

    for banda in cfg.bandas:
        for aquisicao in cfg.aquisicoes_ortho:
            layout = _layout_slc_aprovado(banda, aquisicao)
            mapa = preparar_mapa_inverso_grille(
                cfg,
                banda,
                aquisicao,
                (layout["linhas"], layout["colunas"]),
                finalidade="slc",
            )
            mask = np.load(mapa["mask"], mmap_mode="r")
            for pol in POLARIZACOES:
                caminho = caminho_slc_ortho(cfg, banda, aquisicao, pol)
                if not caminho.exists():
                    raise FileNotFoundError(caminho)
                arr = np.load(caminho, mmap_mode="r")
                if arr.shape != shape_esperado or arr.dtype != np.complex64:
                    raise RuntimeError(
                        f"Saída incompatível {caminho}: shape={arr.shape}, dtype={arr.dtype}."
                    )
                falso_positivo = 0
                falso_negativo = 0
                finitos = 0
                for r0 in range(0, arr.shape[0], cfg.backgeo_chunk_linhas):
                    r1 = min(arr.shape[0], r0 + cfg.backgeo_chunk_linhas)
                    a = np.asarray(arr[r0:r1])
                    fin = np.isfinite(a.real) & np.isfinite(a.imag)
                    m = np.asarray(mask[r0:r1], dtype=bool)
                    falso_positivo += int((fin & ~m).sum())
                    falso_negativo += int((~fin & m).sum())
                    finitos += int(fin.sum())
                if falso_positivo or falso_negativo:
                    raise RuntimeError(
                        f"Finitude não coincide com máscara em {caminho}: "
                        f"falso_positivo={falso_positivo}, falso_negativo={falso_negativo}."
                    )
                passo = max(1, int(max(arr.shape) / 512))
                am = np.asarray(arr[::passo, ::passo])
                registros.append({
                    "banda": banda,
                    "aquisicao": aquisicao,
                    "polarizacao": pol,
                    "arquivo": str(caminho),
                    "shape": str(arr.shape),
                    "dtype": str(arr.dtype),
                    "pixels_finitos": finitos,
                    "fracao_finita": float(finitos / arr.size),
                    "fracao_mascara_geometrica": float(mask.mean()),
                    "falsos_positivos": falso_positivo,
                    "falsos_negativos": falso_negativo,
                    "media_magnitude_amostra": float(np.nanmean(np.abs(am))),
                })
                del arr

    df = pd.DataFrame(registros)
    if len(df) != 16:
        raise RuntimeError(f"Esperados 16 produtos; encontrados {len(df)}.")
    out = cfg.dir_auditoria / "BackGeocoding_Grille3D"
    df.to_csv(out / "auditoria_saidas_backgeocoding.csv", index=False)
    print("\nSaídas do back-geocoding .grille aprovadas:")
    print(
        df[[
            "banda", "aquisicao", "polarizacao", "shape",
            "fracao_finita", "falsos_positivos", "falsos_negativos",
        ]].to_string(index=False)
    )
    return df



# ================================================================
# 3. ETAPA 2 — LANDSAT CORRETO + COERÊNCIA NORMALIZADA + DATASET v2.4
# ================================================================
REFLECTANCE_MULT_FALLBACK = 2.75e-5
REFLECTANCE_ADD_FALLBACK = -0.2


def _ler_mtl_texto(cena: Path) -> str:
    mtl = achar_arquivo_unico(cena, ("*_MTL.txt", "*MTL*.txt"))
    return mtl.read_text(encoding="utf-8", errors="ignore") if mtl else ""


def _mtl_valor(texto: str, chave: str) -> Optional[str]:
    m = re.search(rf"^\s*{re.escape(chave)}\s*=\s*\"?([^\"\r\n]+)\"?\s*$", texto, re.M)
    return m.group(1).strip() if m else None


def extrair_data_cena_landsat(cena: Path) -> Optional[str]:
    texto = _ler_mtl_texto(cena)
    data = _mtl_valor(texto, "DATE_ACQUIRED")
    if data and re.fullmatch(r"\d{4}-\d{2}-\d{2}", data):
        return data
    m = re.search(r"_(\d{8})_", cena.name)
    if m:
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return None


def _cloud_cover_mtl(cena: Path) -> float:
    texto = _ler_mtl_texto(cena)
    for chave in ("CLOUD_COVER_LAND", "CLOUD_COVER"):
        valor = _mtl_valor(texto, chave)
        try:
            if valor is not None:
                return float(valor)
        except ValueError:
            pass
    return np.nan


def ler_coeficientes_mtl(cena: Path) -> dict:
    texto = _ler_mtl_texto(cena)
    coef = {}
    for b in (1, 2, 3, 4, 5, 7):
        m_mult = re.search(
            rf"REFLECTANCE_MULT_BAND_{b}\s*=\s*([+\-0-9.Ee]+)",
            texto,
        )
        m_add = re.search(
            rf"REFLECTANCE_ADD_BAND_{b}\s*=\s*([+\-0-9.Ee]+)",
            texto,
        )
        coef[b] = (
            float(m_mult.group(1)) if m_mult else REFLECTANCE_MULT_FALLBACK,
            float(m_add.group(1)) if m_add else REFLECTANCE_ADD_FALLBACK,
        )
    return coef


def decodificar_qa_pixel_landsat(qa: np.ndarray) -> dict[str, np.ndarray]:
    qa = qa.astype(np.uint16)
    fill = (qa & (1 << 0)) > 0
    dilated = (qa & (1 << 1)) > 0
    cloud = (qa & (1 << 3)) > 0
    shadow = (qa & (1 << 4)) > 0
    snow = (qa & (1 << 5)) > 0
    invalido = fill | dilated | cloud | shadow | snow
    return {
        "fill": fill,
        "dilated_cloud": dilated,
        "cloud": cloud,
        "cloud_shadow": shadow,
        "snow": snow,
        "valid_qa": ~invalido,
    }


def decodificar_qa_radsat_landsat(qa: np.ndarray) -> np.ndarray:
    qa = qa.astype(np.uint16)
    # Landsat-5: bits das bandas ópticas B1, B2, B3, B4, B5 e B7.
    bits = (0, 1, 2, 3, 4, 6)
    saturado = np.zeros(qa.shape, dtype=bool)
    for bit in bits:
        saturado |= (qa & (1 << bit)) > 0
    return ~saturado


def _mesma_grade(perfil_a: dict, perfil_b: dict) -> bool:
    return (
        perfil_a["width"] == perfil_b["width"]
        and perfil_a["height"] == perfil_b["height"]
        and perfil_a["crs"] == perfil_b["crs"]
        and perfil_a["transform"].almost_equals(perfil_b["transform"])
    )


def _reprojetar_array(
    arr: np.ndarray,
    src_profile: dict,
    dst_profile: dict,
    resampling: Resampling,
    src_nodata=None,
    dst_nodata=np.nan,
) -> np.ndarray:
    destino = np.full(
        (dst_profile["height"], dst_profile["width"]),
        dst_nodata,
        dtype=np.float32 if np.issubdtype(arr.dtype, np.floating) else arr.dtype,
    )
    reproject(
        source=arr,
        destination=destino,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=src_nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )
    return destino


def _carregar_cena_landsat_em_grade(
    cena: Path,
    grade_ref: Optional[dict] = None,
) -> dict:
    coef = ler_coeficientes_mtl(cena)
    bandas = []
    perfil_cena = None
    valid_dn = None

    for b in (1, 2, 3, 4, 5, 7):
        p = achar_arquivo_unico(cena, (f"*_SR_B{b}.TIF", f"*B{b}.TIF"))
        if p is None:
            raise FileNotFoundError(f"Banda B{b} ausente em {cena}")
        with rasterio.open(p) as src:
            dn = src.read(1).astype(np.float32)
            nodata = src.nodata
            perfil = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
            }
            if perfil_cena is None:
                perfil_cena = perfil
            elif not _mesma_grade(perfil_cena, perfil):
                raise RuntimeError(
                    f"Bandas da mesma cena em grades distintas: {cena}"
                )

            mult, add = coef[b]
            refl = dn * mult + add
            valido_b = np.isfinite(dn)
            if nodata is not None:
                valido_b &= dn != nodata
                refl[dn == nodata] = np.nan
            bandas.append(refl.astype(np.float32))
            valid_dn = valido_b if valid_dn is None else (valid_dn & valido_b)

    stack6 = np.stack(bandas, axis=0)
    destino = grade_ref or perfil_cena
    if grade_ref is not None and not _mesma_grade(perfil_cena, grade_ref):
        reproj = []
        for b in stack6:
            reproj.append(
                _reprojetar_array(
                    b, perfil_cena, grade_ref, Resampling.bilinear,
                    src_nodata=np.nan, dst_nodata=np.nan,
                )
            )
        stack6 = np.stack(reproj, axis=0).astype(np.float32)
        valid_dn = _reprojetar_array(
            valid_dn.astype(np.uint8),
            perfil_cena,
            grade_ref,
            Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        ).astype(bool)

    azul, vermelho, nir = stack6[0], stack6[2], stack6[3]
    eps = 1e-6
    ndvi = (nir - vermelho) / (nir + vermelho + eps)
    evi = 2.5 * (nir - vermelho) / (
        nir + 6.0 * vermelho - 7.5 * azul + 1.0 + eps
    )
    ndvi = np.clip(ndvi, -1.0, 1.0)
    evi = np.clip(evi, -2.0, 2.0)
    stack8 = np.concatenate(
        [stack6, ndvi[None], evi[None]], axis=0
    ).astype(np.float32)

    qa_path = achar_arquivo_unico(
        cena, ("*_QA_PIXEL.TIF", "*QA_PIXEL*.TIF")
    )
    if qa_path is None and CFG.dados.landsat_exigir_qa_pixel:
        raise FileNotFoundError(f"QA_PIXEL ausente em {cena}")

    if qa_path:
        with rasterio.open(qa_path) as src:
            qa = src.read(1).astype(np.uint16)
            qa_profile = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
            }
        if not _mesma_grade(qa_profile, destino):
            # QA_PIXEL=0 significa pixel limpo, não NoData.
            # Portanto, 0 jamais pode ser passado como src_nodata.
            qa = _reprojetar_array(
                qa, qa_profile, destino, Resampling.nearest,
                src_nodata=None, dst_nodata=1,
            ).astype(np.uint16)
        qa_flags = decodificar_qa_pixel_landsat(qa)
        valid_qa = qa_flags["valid_qa"]
    else:
        qa_flags = {}
        valid_qa = np.ones(valid_dn.shape, dtype=bool)

    radsat_path = achar_arquivo_unico(
        cena, ("*_QA_RADSAT.TIF", "*QA_RADSAT*.TIF")
    )
    if radsat_path and CFG.dados.landsat_usar_qa_radsat:
        with rasterio.open(radsat_path) as src:
            radsat = src.read(1).astype(np.uint16)
            rad_profile = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
            }
        if not _mesma_grade(rad_profile, destino):
            # QA_RADSAT=0 significa nenhuma saturação, não NoData.
            radsat = _reprojetar_array(
                radsat, rad_profile, destino, Resampling.nearest,
                src_nodata=None, dst_nodata=np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        valid_radsat = decodificar_qa_radsat_landsat(radsat)
    else:
        valid_radsat = np.ones(valid_dn.shape, dtype=bool)

    valid_spectral = valid_dn & np.all(np.isfinite(stack8), axis=0)
    valid = valid_spectral & valid_qa & valid_radsat

    janela = int(CFG.dados.landsat_janela_qualidade)
    if janela < 1 or janela % 2 == 0:
        raise ValueError("landsat_janela_qualidade deve ser ímpar e >= 1.")
    fracao_local = uniform_filter(
        valid.astype(np.float32),
        size=janela,
        mode="constant",
        cval=0.0,
    )
    data = extrair_data_cena_landsat(cena)
    cloud_cover = _cloud_cover_mtl(cena)

    return {
        "cena": cena,
        "data": data,
        "cloud_cover": cloud_cover,
        "stack8": stack8,
        "valid": valid,
        "valid_spectral": valid_spectral,
        "valid_qa": valid_qa,
        "valid_radsat": valid_radsat,
        "fracao_local": fracao_local.astype(np.float32),
        "profile": destino,
        "qa_flags": qa_flags,
    }


def _dias_ate_referencia(data: Optional[str], referencia: Optional[str]) -> float:
    if not data or not referencia:
        return np.nan
    try:
        return float(
            abs((pd.Timestamp(data) - pd.Timestamp(referencia)).days)
        )
    except Exception:
        return np.nan


def _salvar_geotiff_multibanda(
    path: Path,
    arr: np.ndarray,
    profile_ref: dict,
    nodata: float = -9999.0,
) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(profile_ref["height"]),
        "width": int(profile_ref["width"]),
        "transform": profile_ref["transform"],
        "crs": profile_ref["crs"],
        "dtype": "float32",
        "count": int(arr.shape[0]),
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    profile = _configurar_blocos_geotiff(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    saida = np.where(
        np.isfinite(arr),
        arr,
        nodata,
    ).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(saida)


def preparar_mosaico_landsat_qualidade(
    cfg: ConfigDados = CFG.dados,
    sobrescrever: Optional[bool] = None,
) -> dict:
    """Monta um mosaico temporal escolhendo uma cena inteira por célula Landsat."""
    sobrescrever = cfg.sobrescrever if sobrescrever is None else bool(sobrescrever)
    out = cfg.dir_landsat_processado
    out.mkdir(parents=True, exist_ok=True)

    cache_npz = out / "mosaico_landsat_qualidade.npz"
    relatorio_path = out / "relatorio_mosaico_landsat.json"
    if cache_npz.exists() and relatorio_path.exists() and not sobrescrever:
        relatorio = json.loads(relatorio_path.read_text(encoding="utf-8"))
        versao_cache = relatorio.get("pipeline_version")
        cenas_cache = relatorio.get("scenes", [])
        cache_semanticamente_compativel = (
            relatorio.get("method")
            == "quality_controlled_single_scene_per_30m_cell"
            and relatorio.get("target_leakage") is False
            and bool(cenas_cache)
            and all(
                "valid_spectral_cells" in cena
                and "valid_qa_cells" in cena
                and "valid_radsat_cells" in cena
                for cena in cenas_cache
            )
        )
        if versao_cache == VERSAO_PIPELINE or cache_semanticamente_compativel:
            dados = np.load(cache_npz, allow_pickle=False)
            print(
                "Cache Landsat semanticamente compatível reutilizado: "
                f"{versao_cache}",
                flush=True,
            )
            return {
                "stack": dados["stack"].astype(np.float32),
                "quality": dados["quality"].astype(np.float32),
                "source_index": dados["source_index"].astype(np.int16),
                "scene_stack": dados["scene_stack"].astype(np.float32),
                "scene_valid": dados["scene_valid"].astype(bool),
                "profile": {
                    "transform": rasterio.Affine(*relatorio["profile"]["transform"]),
                    "crs": CRS.from_string(relatorio["profile"]["crs"]),
                    "width": int(relatorio["profile"]["width"]),
                    "height": int(relatorio["profile"]["height"]),
                },
                "scenes": relatorio["scenes"],
                "report": relatorio,
            }

        print(
            "Cache Landsat incompatível detectado: "
            f"{versao_cache!r} != {VERSAO_PIPELINE!r}. "
            "O mosaico será recalculado automaticamente.",
            flush=True,
        )

    cenas = descobrir_cenas_landsat(cfg)
    info = []
    grade_ref = None
    for cena in cenas:
        item = _carregar_cena_landsat_em_grade(cena, grade_ref)
        if grade_ref is None:
            grade_ref = item["profile"]
        info.append(item)

    scene_stack = np.stack([x["stack8"] for x in info], axis=0)
    scene_valid = np.stack([x["valid"] for x in info], axis=0)
    local = np.stack([x["fracao_local"] for x in info], axis=0)

    scores = np.full(scene_valid.shape, -np.inf, dtype=np.float64)
    scene_records = []
    for i, item in enumerate(info):
        cloud = item["cloud_cover"]
        days = _dias_ate_referencia(
            item["data"], cfg.landsat_data_referencia_tropisar
        )
        # Prioridade: validade central -> fração clara local -> nuvens globais
        # -> proximidade temporal configurada -> desempate determinístico.
        score = local[i].astype(np.float64) * 1000.0
        if np.isfinite(cloud):
            score += (100.0 - float(cloud)) * 1e-3
        if np.isfinite(days):
            score -= float(days) * 1e-6
        score -= i * 1e-9
        score[~scene_valid[i]] = -np.inf
        scores[i] = score

        scene_records.append({
            "scene_index": i,
            "name": item["cena"].name,
            "path": str(item["cena"]),
            "date": item["data"],
            "cloud_cover_mtl_percent": (
                float(cloud) if np.isfinite(cloud) else None
            ),
            "days_to_tropisar_reference": (
                float(days) if np.isfinite(days) else None
            ),
            "valid_spectral_cells": int(item["valid_spectral"].sum()),
            "valid_qa_cells": int(item["valid_qa"].sum()),
            "valid_radsat_cells": int(item["valid_radsat"].sum()),
            "valid_cells": int(item["valid"].sum()),
            "valid_fraction": float(item["valid"].mean()),
            "mean_local_clear_fraction": float(
                item["fracao_local"][item["valid"]].mean()
            ) if item["valid"].any() else 0.0,
        })

    any_valid = np.any(scene_valid, axis=0)
    source_index = np.argmax(scores, axis=0).astype(np.int16)
    source_index[~any_valid] = -1

    h, w = any_valid.shape
    stack = np.full((8, h, w), np.nan, dtype=np.float32)
    quality = np.zeros((h, w), dtype=np.float32)
    for i in range(len(info)):
        m = source_index == i
        if np.any(m):
            stack[:, m] = scene_stack[i][:, m]
            quality[m] = local[i, m]

    quality[quality < cfg.landsat_fracao_clara_local_minima] = 0.0
    invalid_quality = quality <= 0
    stack[:, invalid_quality] = np.nan

    counts = {
        str(i): int((source_index == i).sum())
        for i in range(len(info))
    }
    recovered = np.zeros((h, w), dtype=bool)
    if len(info) >= 2:
        recovered = (~scene_valid[0]) & any_valid

    relatorio = {
        "pipeline_version": VERSAO_PIPELINE,
        "method": "quality_controlled_single_scene_per_30m_cell",
        "target_leakage": False,
        "selection_variables": [
            "QA_PIXEL",
            "spectral_NoData_and_nonfinite",
            "QA_RADSAT_when_available",
            f"local_clear_fraction_{cfg.landsat_janela_qualidade}x{cfg.landsat_janela_qualidade}",
            "scene_cloud_cover_MTL",
            "temporal_distance_if_reference_configured",
        ],
        "important_constraint": (
            "All six reflectance bands plus NDVI/EVI are copied from the same "
            "selected scene. Bands are never mixed across dates within one cell."
        ),
        "landsat_reference_date_tropisar": cfg.landsat_data_referencia_tropisar,
        "profile": {
            "transform": list(grade_ref["transform"])[:6],
            "crs": str(grade_ref["crs"]),
            "width": int(grade_ref["width"]),
            "height": int(grade_ref["height"]),
        },
        "scenes": scene_records,
        "selection_counts": counts,
        "cells_total": int(any_valid.size),
        "cells_valid_any_scene": int(any_valid.sum()),
        "cells_invalid_all_scenes": int((~any_valid).sum()),
        "cells_recovered_relative_to_scene_0": int(recovered.sum()),
        "mean_selected_local_quality": float(
            quality[quality > 0].mean()
        ) if np.any(quality > 0) else 0.0,
        "interpretation": (
            "This is a multitemporal quality mosaic, not a simultaneous "
            "observation. The selected source date is stored for every sample."
        ),
    }

    np.savez_compressed(
        cache_npz,
        stack=stack.astype(np.float32),
        quality=quality.astype(np.float32),
        source_index=source_index.astype(np.int16),
        scene_stack=scene_stack.astype(np.float32),
        scene_valid=scene_valid.astype(np.uint8),
    )
    relatorio_path.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(scene_records).to_csv(
        out / "cenas_landsat.csv", index=False
    )

    _salvar_geotiff_multibanda(
        out / "Landsat5_mosaico_8_variaveis.tif",
        stack,
        grade_ref,
    )
    _salvar_geotiff_float32(
        out / "Landsat5_qualidade_local.tif",
        quality,
        {
            **grade_ref,
            "count": 1,
            "dtype": "float32",
        },
    )
    source_profile = {
        "driver": "GTiff",
        "height": grade_ref["height"],
        "width": grade_ref["width"],
        "transform": grade_ref["transform"],
        "crs": grade_ref["crs"],
        "dtype": "int16",
        "count": 1,
        "nodata": -1,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(
        out / "Landsat5_fonte_cena.tif", "w", **source_profile
    ) as dst:
        dst.write(source_index.astype(np.int16), 1)

    plt.figure(figsize=(10, 7))
    plt.imshow(source_index, vmin=-1, vmax=max(1, len(info) - 1))
    plt.colorbar(label="Selected scene index")
    plt.title("Landsat-5 source selected for each 30 m cell")
    plt.tight_layout()
    plt.savefig(out / "mapa_fonte_cena_landsat.png", dpi=180)
    plt.close()

    print("\nMosaico Landsat-5 por qualidade concluído.")
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return {
        "stack": stack,
        "quality": quality,
        "source_index": source_index,
        "scene_stack": scene_stack,
        "scene_valid": scene_valid,
        "profile": grade_ref,
        "scenes": scene_records,
        "report": relatorio,
    }


def carregar_landsat_reflectancia(cfg: ConfigDados):
    if cfg.landsat_modo != "mosaico_qualidade":
        raise ValueError(
            "A v2.8.3 está validada para landsat_modo='mosaico_qualidade'."
        )
    mosaico = preparar_mosaico_landsat_qualidade(cfg)
    return (
        mosaico["stack"],
        mosaico["quality"],
        mosaico["profile"],
        mosaico,
    )


def caminho_slc_ortho(cfg: ConfigDados, banda: str, aquisicao: str, pol: str) -> Path:
    sufixo = "Lcons" if banda == "SAR_L" else "Pcons"
    return cfg.dir_backgeo_grille3d / banda / aquisicao / f"{aquisicao}_{sufixo}_{pol}_slc_ORTORRETIFICADO_GRILLE3D.npy"


def media_local_ponderada(arr: np.ndarray, valido: np.ndarray, kernel: int) -> np.ndarray:
    v = valido.astype(np.float32)
    num = uniform_filter(np.where(valido, arr, 0.0).astype(np.float32), size=kernel, mode="nearest")
    den = uniform_filter(v, size=kernel, mode="nearest")
    return num / np.maximum(den, 1e-6)


def calcular_coerencia_normalizada_fullgrid(
    mestre_path: Path,
    escravo_path: Path,
    saida_path: Path,
    kernel: int = 5,
    sobrescrever: bool = False,
) -> Path:
    if saida_path.exists() and not sobrescrever:
        return saida_path
    mestre = np.load(mestre_path, mmap_mode="r")
    escravo = np.load(escravo_path, mmap_mode="r")
    if mestre.shape != escravo.shape:
        raise ValueError(f"Shapes mestre/escravo diferentes: {mestre.shape} vs {escravo.shape}")

    m = np.asarray(mestre, dtype=np.complex64)
    e = np.asarray(escravo, dtype=np.complex64)
    valido = np.isfinite(m.real) & np.isfinite(m.imag) & np.isfinite(e.real) & np.isfinite(e.imag)
    valido &= (np.abs(m) > 0) & (np.abs(e) > 0)
    cruzado = m * np.conj(e)
    cross_r = media_local_ponderada(cruzado.real, valido, kernel)
    cross_i = media_local_ponderada(cruzado.imag, valido, kernel)
    p1 = media_local_ponderada(np.abs(m)**2, valido, kernel)
    p2 = media_local_ponderada(np.abs(e)**2, valido, kernel)
    den = np.sqrt(np.maximum(p1*p2, 1e-12))
    gamma = (cross_r + 1j*cross_i)/den
    mag = np.abs(gamma)
    gamma = np.where(mag > 1.0, gamma/np.maximum(mag, 1e-8), gamma).astype(np.complex64)
    gamma[~valido] = np.nan + 1j*np.nan
    saida_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(saida_path, gamma)
    del mestre, escravo, m, e, cruzado, cross_r, cross_i, p1, p2, gamma
    gc.collect()
    return saida_path


def preparar_mapas_coerencia(cfg: ConfigDados = CFG.dados) -> dict[tuple[str, str], Path]:
    saidas = {}
    par = f"{cfg.aquisicao_mestre}_{cfg.aquisicao_escrava}"
    for banda in cfg.bandas:
        pasta = cfg.dir_processados / "PolInSAR_Coerencia_Grille3D" / banda / par
        for pol in POLARIZACOES:
            m = caminho_slc_ortho(cfg, banda, cfg.aquisicao_mestre, pol)
            e = caminho_slc_ortho(cfg, banda, cfg.aquisicao_escrava, pol)
            if not m.exists() or not e.exists():
                raise FileNotFoundError(f"SLC ortorretificado ausente: {m} ou {e}")
            out = pasta / f"gamma_{banda}_{par}_{pol}_k{cfg.janela_coerencia}.npy"
            saidas[(banda, pol)] = calcular_coerencia_normalizada_fullgrid(m, e, out, cfg.janela_coerencia, cfg.sobrescrever)
            print("Coerência:", out)
    return saidas


def matrizes_polinsar_patch(k1: np.ndarray, k2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # k1/k2: [4,H,W]. Matrizes de segunda ordem médias na célula Landsat.
    a = k1.reshape(4, -1)
    b = k2.reshape(4, -1)
    valido = np.all(np.isfinite(a.real) & np.isfinite(a.imag) & np.isfinite(b.real) & np.isfinite(b.imag), axis=0)
    valido &= (np.sum(np.abs(a), axis=0) > 0) & (np.sum(np.abs(b), axis=0) > 0)
    a, b = a[:, valido], b[:, valido]
    if a.shape[1] < 16:
        nanmat = np.full((4,4), np.nan+1j*np.nan, dtype=np.complex64)
        return nanmat, nanmat.copy(), nanmat.copy()
    n = float(a.shape[1])
    t11 = np.einsum("in,jn->ij", a, np.conj(a))/n
    t22 = np.einsum("in,jn->ij", b, np.conj(b))/n
    omega = np.einsum("in,jn->ij", a, np.conj(b))/n
    return t11.astype(np.complex64), t22.astype(np.complex64), omega.astype(np.complex64)


def gamma_diagonal_matrizes(t11: np.ndarray, t22: np.ndarray, omega: np.ndarray) -> np.ndarray:
    den = np.sqrt(np.maximum(np.real(np.diag(t11))*np.real(np.diag(t22)), 1e-12))
    return (np.diag(omega)/den).astype(np.complex64)


def correlacoes_canonicas(t11: np.ndarray, t22: np.ndarray, omega: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(t11)):
        return np.full(4, np.nan, dtype=np.float32)
    eps = 1e-6
    def invsqrt_h(m):
        m = 0.5*(m+m.conj().T) + eps*np.eye(m.shape[0])
        vals, vecs = np.linalg.eigh(m)
        vals = np.maximum(vals.real, eps)
        return (vecs @ np.diag(vals**-0.5) @ vecs.conj().T)
    c = invsqrt_h(t11) @ omega @ invsqrt_h(t22)
    s = np.linalg.svd(c, compute_uv=False)
    return np.clip(s.real, 0, 1).astype(np.float32)


def redimensionar_complexo(arr: np.ndarray, tamanho: int) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr, dtype=np.complex64))
    if t.shape[-2:] == (tamanho, tamanho):
        return t
    real = F.interpolate(t.real[None], size=(tamanho,tamanho), mode="bilinear", align_corners=False)[0]
    imag = F.interpolate(t.imag[None], size=(tamanho,tamanho), mode="bilinear", align_corners=False)[0]
    return torch.complex(real, imag)


def redimensionar_real(arr: np.ndarray, tamanho: int, modo: str = "bilinear") -> torch.Tensor:
    t = torch.from_numpy(np.asarray(arr, dtype=np.float32))[None,None]
    if t.shape[-2:] == (tamanho,tamanho):
        return t[0]
    kwargs = {"size": (tamanho,tamanho), "mode": modo}
    if modo in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(t, **kwargs)[0]


def localizar_geometria(cfg: ConfigDados, banda: str, tipo: str) -> Optional[Path]:
    """Localiza apenas geometria externa em Dados_Brutos.

    Produtos derivados de versões antigas, Resultados e Dataset são excluídos
    para impedir reutilização silenciosa de kz/incidência inválidos.
    """
    token = "L" if banda == "SAR_L" else "P"
    padroes = cfg.padroes_kz if tipo == "kz" else cfg.padroes_incidencia
    candidatos = []
    for padrao in padroes:
        candidatos.extend(cfg.dir_brutos.rglob(padrao))
    candidatos = [
        p.resolve() for p in candidatos
        if _arquivo_cientifico_real(p)
        and token.lower() in str(p).lower()
    ]
    return sorted(set(candidatos))[0] if candidatos else None


def localizar_kz(cfg: ConfigDados, banda: str) -> Optional[Path]:
    if banda == "SAR_P":
        canonico = (
            cfg.dir_processados / "Geometria_RVoG_Grille3D"
            / "kz_SAR_P_0506_0603_Ha_grade_CHM_v3_2_0.npy"
        )
        if canonico.exists():
            return canonico
    return localizar_geometria(cfg, banda, "kz")


def alinhar_raster_fisico_para_chm(origem: Path, chm_path: Path, saida_npy: Path) -> Path:
    if saida_npy.exists():
        return saida_npy
    with rasterio.open(chm_path) as ref:
        shape, transform, crs = (ref.height, ref.width), ref.transform, ref.crs
    if origem.suffix.lower() == ".npy":
        arr = np.load(origem, mmap_mode="r")
        if arr.shape != shape:
            raise ValueError(f"kz NPY {origem} não coincide com CHM: {arr.shape} vs {shape}")
        np.save(saida_npy, np.asarray(arr, dtype=np.float32))
        return saida_npy
    with rasterio.open(origem) as src:
        destino = np.full(shape, np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=destino,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    saida_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(saida_npy, destino)
    return saida_npy


def carregar_ou_constante(caminho: Optional[Path], constante: Optional[float], shape: tuple[int,int]):
    if caminho is not None:
        return np.load(caminho, mmap_mode="r")
    if constante is not None:
        return float(constante)
    return None


def extrair_janela(arr, window: Window) -> np.ndarray:
    r0, c0 = int(round(window.row_off)), int(round(window.col_off))
    h, w = int(round(window.height)), int(round(window.width))
    return np.asarray(arr[r0:r0+h, c0:c0+w])




def _quantis_amostra_distancia(valores: np.ndarray) -> dict:
    valores = np.asarray(valores, dtype=np.float64)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        return {}
    qs = np.percentile(valores, [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        "min": float(qs[0]),
        "q25": float(qs[1]),
        "mediana": float(qs[2]),
        "q75": float(qs[3]),
        "q90": float(qs[4]),
        "q95": float(qs[5]),
        "q99": float(qs[6]),
        "max": float(qs[7]),
    }


# Função legada _pontos_lut_projetados removida na v3.2.0: a geometria canônica usa .grille 3D.



def preparar_geometria_sar_comum(
    cfg: ConfigDados = CFG.dados,
    sobrescrever: Optional[bool] = None,
) -> dict:
    """Interseção geométrica das quatro inversões .grille do par 0506–0603."""
    sobrescrever = cfg.sobrescrever if sobrescrever is None else bool(sobrescrever)
    out = cfg.dir_geometria_sar / "Comum_LP_0506_0603"
    out.mkdir(parents=True, exist_ok=True)
    mask_path = out / "mascara_geometrica_comum_quatro_grilles.npy"
    margin_path = out / "margem_minima_quatro_grilles_px.npy"
    mask_tif = out / "mascara_geometrica_comum_quatro_grilles.tif"
    margin_tif = out / "margem_minima_quatro_grilles_px.tif"
    report_path = out / "relatorio_geometria_sar_comum.json"

    if mask_path.exists() and margin_path.exists() and report_path.exists() and not sobrescrever:
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            rep.get("pipeline_version") == VERSAO_PIPELINE
            and rep.get("backgeo_schema_version") == cfg.backgeo_schema_version
        ):
            return {
                "mask_path": mask_path,
                "margin_path": margin_path,
                "mask_tif": mask_tif,
                "margin_tif": margin_tif,
                "report_path": report_path,
                "report": rep,
            }

    mapas = []
    for banda in cfg.bandas:
        for aquisicao in cfg.aquisicoes_ortho:
            layout = _layout_slc_aprovado(banda, aquisicao)
            mapas.append(
                preparar_mapa_inverso_grille(
                    cfg,
                    banda,
                    aquisicao,
                    (layout["linhas"], layout["colunas"]),
                    finalidade="slc",
                    sobrescrever=sobrescrever,
                )
            )

    with rasterio.open(cfg.caminho_dtm_para_grille) as ref:
        H, W = ref.height, ref.width
        profile = ref.profile.copy()

    common = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=(H, W)
    )
    margin = np.lib.format.open_memmap(
        margin_path, mode="w+", dtype=np.float32, shape=(H, W)
    )

    mask_memmaps = [np.load(m["mask"], mmap_mode="r") for m in mapas]
    margin_memmaps = [np.load(m["margin"], mmap_mode="r") for m in mapas]
    for arr in [*mask_memmaps, *margin_memmaps]:
        if arr.shape != (H, W):
            raise RuntimeError(
                f"Mapa geométrico com shape incompatível: {arr.shape}; esperado={(H, W)}."
            )

    for r0 in tqdm(
        range(0, H, cfg.backgeo_chunk_linhas),
        desc="Interseção geométrica L/P 0506/0603",
    ):
        r1 = min(H, r0 + cfg.backgeo_chunk_linhas)
        masks = [np.asarray(a[r0:r1], dtype=bool) for a in mask_memmaps]
        margins = [np.asarray(a[r0:r1], dtype=np.float32) for a in margin_memmaps]
        cm = np.logical_and.reduce(masks)
        mg = np.minimum.reduce(
            [np.where(cm, a, np.nan) for a in margins]
        ).astype(np.float32)
        common[r0:r1] = cm
        margin[r0:r1] = mg

    common.flush(); margin.flush()
    del common, margin, mask_memmaps, margin_memmaps
    gc.collect()

    common = np.load(mask_path, mmap_mode="r")
    margin = np.load(margin_path, mmap_mode="r")
    profile.update(dtype="uint8", nodata=0, count=1, compress="deflate")
    with rasterio.open(mask_tif, "w", **profile) as dst:
        dst.write(np.asarray(common, dtype=np.uint8), 1)
    profile.update(dtype="float32", nodata=np.nan)
    with rasterio.open(margin_tif, "w", **profile) as dst:
        dst.write(np.asarray(margin, dtype=np.float32), 1)

    valid_margin = np.asarray(margin[common], dtype=np.float64)
    rep = {
        "pipeline_version": VERSAO_PIPELINE,
        "backgeo_schema_version": cfg.backgeo_schema_version,
        "par": [cfg.aquisicao_mestre, cfg.aquisicao_escrava],
        "bandas": list(cfg.bandas),
        "pixels_total": int(common.size),
        "pixels_validos_comuns": int(common.sum()),
        "fracao_valida_comum": float(common.mean()),
        "margem_minima_px": _quantis_amostra_distancia(valid_margin),
        "fontes": [m["report_data"] for m in mapas],
        "policy": (
            "pixel válido somente se a inversão .grille e os quatro vizinhos "
            "de interpolação estiverem dentro dos quatro SLCs"
        ),
        "arquivos": {
            "mask_path": str(mask_path),
            "margin_path": str(margin_path),
            "mask_tif": str(mask_tif),
            "margin_tif": str(margin_tif),
        },
    }
    report_path.write_text(
        json.dumps(rep, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nGeometria SAR comum preparada:")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return {
        "mask_path": mask_path,
        "margin_path": margin_path,
        "mask_tif": mask_tif,
        "margin_tif": margin_tif,
        "report_path": report_path,
        "report": rep,
    }


def remover_produtos_obsoletos_pre_grille(
    cfg: ConfigDados = CFG.dados,
    confirmar: bool = False,
) -> list[str]:
    """Remove somente derivados produzidos pela geometria ROI/KDTree incorreta.

    Dados_Brutos, LiDAR_CHM, Landsat_Mosaico e Auditoria são preservados.
    """
    alvos = [
        cfg.dir_processados / "SAR_L",
        cfg.dir_processados / "SAR_P",
        cfg.dir_processados / "PolInSAR_Coerencia",
        cfg.dir_processados / "Suporte_SAR_Comum_30m",
        cfg.raiz / "Dataset" / "TCCT_dataset_v4_30m",
        cfg.raiz / "Dataset" / "TCCT_dataset_v5_suporte_sar_30m",
        cfg.raiz / "Resultados" / "TCCT_v3_0_SIBGRAPI",
        cfg.raiz / "Resultados" / "TCCT_v3_1_SIBGRAPI",
    ]

    geometria = cfg.dir_processados / "Geometria"
    if geometria.exists():
        alvos.extend(sorted(geometria.glob("kz_SAR_P_0506_0603_Ha_grade_CHM_v3_*.npy")))
        alvos.extend(sorted(geometria.glob("kz_SAR_P_0506_0603_Ha_grade_CHM_v3_*.json")))

    existentes = []
    vistos = set()
    for p in alvos:
        p = Path(p)
        if p.exists() and str(p.resolve()) not in vistos:
            existentes.append(p)
            vistos.add(str(p.resolve()))

    print("\nProdutos derivados obsoletos:")
    if not existentes:
        print("  Nenhum produto obsoleto encontrado.")
        return []
    for p in existentes:
        print(" -", p)

    if not confirmar:
        print("Nada foi removido. Use confirmar=True após revisar a lista.")
        return [str(p) for p in existentes]

    removidos = []
    for p in existentes:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        removidos.append(str(p))
        print("Removido:", p)

    log = cfg.dir_auditoria / "BackGeocoding_Grille3D" / "limpeza_produtos_obsoletos.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "pipeline_version": VERSAO_PIPELINE,
                "executado_em": pd.Timestamp.now().isoformat(),
                "removidos": removidos,
                "preservados": [
                    str(cfg.dir_brutos),
                    str(cfg.dir_lidar_processado),
                    str(cfg.dir_landsat_processado),
                    str(cfg.dir_auditoria),
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return removidos


# Alias explícito para impedir que notebooks antigos executem a limpeza errada.
def remover_produtos_invalidos_pre_suporte(*args, **kwargs):
    raise RuntimeError(
        "Função v3.1 desativada. Use remover_produtos_obsoletos_pre_grille()."
    )

def carregar_mascara_zero_legacy(cfg: ConfigDados) -> np.ndarray:
    """Reproduz apenas o primeiro filtro óptico do pipeline legado.

    O pipeline antigo descartava uma célula somente quando as seis bandas DN e os
    índices derivados eram todos zero. Como NDVI/EVI também são zero quando as seis
    bandas são zero, basta testar as seis bandas brutas.
    """
    cena = encontrar_cena_landsat(cfg)
    bandas_dn = []
    for b in (1, 2, 3, 4, 5, 7):
        caminho = achar_arquivo_unico(cena, (f"*_SR_B{b}.TIF", f"*B{b}.TIF"))
        if caminho is None:
            raise FileNotFoundError(f"Banda Landsat B{b} ausente em {cena}")
        with rasterio.open(caminho) as src:
            bandas_dn.append(src.read(1))
    return np.all(np.stack(bandas_dn, axis=0) == 0, axis=0)


def contar_patches_legacy(cfg: ConfigDados) -> Optional[int]:
    """Conta os .pt do dataset antigo, se ele ainda estiver disponível."""
    if not cfg.dir_dataset_legacy.exists():
        return None
    return sum(1 for _ in cfg.dir_dataset_legacy.rglob("*.pt"))


def _linha_auditoria(
    y: int,
    x: int,
    meta_ls: dict,
    motivo: str,
    aceita_legacy: bool,
    aceita_atual: bool,
    fracao_chm_valido: float = np.nan,
    media_chm_valido: float = np.nan,
    r0: Optional[int] = None,
    c0: Optional[int] = None,
    h: Optional[int] = None,
    w: Optional[int] = None,
) -> dict:
    centro_x, centro_y = meta_ls["transform"] * (x + 0.5, y + 0.5)
    return {
        "landsat_row": int(y),
        "landsat_col": int(x),
        "x": float(centro_x),
        "y": float(centro_y),
        "motivo_atual": motivo,
        "aceita_legacy_equivalente": bool(aceita_legacy),
        "aceita_pipeline_atual": bool(aceita_atual),
        "fracao_chm_valido": float(fracao_chm_valido) if np.isfinite(fracao_chm_valido) else np.nan,
        "media_chm_valido_m": float(media_chm_valido) if np.isfinite(media_chm_valido) else np.nan,
        "chm_row": r0,
        "chm_col": c0,
        "chm_h": h,
        "chm_w": w,
    }


def salvar_auditoria_filtros(
    cfg: ConfigDados,
    contadores: Counter,
    transicoes: Counter,
    motivos_removidos: Counter,
    removidos_detalhes: list[dict],
    adicionados_detalhes: list[dict],
    legacy_real: Optional[int],
) -> dict:
    """Salva contagens, transições e a lista dos patches que mudaram de decisão."""
    pasta = cfg.dir_dataset_v2
    total = int(contadores.get("total_pixels_landsat", 0))
    aceita = int(contadores.get("aceita", 0))

    ordem = [
        "total_pixels_landsat",
        "rejeitada_optico_nao_finito",
        "rejeitada_qa_nuvem_sombra_neve_fill",
        "rejeitada_fora_chm_ou_janela_invalida",
        "rejeitada_chm_incompleto_ou_fora_faixa",
        "rejeitada_chm_media_baixa",
        "rejeitada_geometria_sar_invalida",
        "aceita",
    ]
    linhas = []
    for nome in ordem:
        valor = int(contadores.get(nome, 0))
        linhas.append({
            "criterio": nome,
            "quantidade": valor,
            "percentual_total": 100.0 * valor / max(total, 1),
        })
    pd.DataFrame(linhas).to_csv(pasta / "contadores_filtros.csv", index=False)

    linhas_trans = [
        {"transicao": nome, "quantidade": int(valor)}
        for nome, valor in sorted(transicoes.items())
    ]
    pd.DataFrame(linhas_trans).to_csv(pasta / "transicoes_legacy_atual.csv", index=False)

    linhas_motivos = [
        {"motivo": nome, "quantidade": int(valor)}
        for nome, valor in sorted(motivos_removidos.items())
    ]
    pd.DataFrame(linhas_motivos).to_csv(pasta / "motivos_remocao_legacy_para_atual.csv", index=False)

    if cfg.salvar_auditoria_detalhada:
        pd.DataFrame(removidos_detalhes).to_csv(
            pasta / "patches_removidos_legacy_para_atual.csv", index=False
        )
        pd.DataFrame(adicionados_detalhes).to_csv(
            pasta / "patches_adicionados_atual.csv", index=False
        )

    legacy_simulado = int(
        transicoes.get("aceita_em_ambos", 0)
        + transicoes.get("aceita_apenas_legacy", 0)
    )
    atual_simulado = int(
        transicoes.get("aceita_em_ambos", 0)
        + transicoes.get("aceita_apenas_atual", 0)
    )
    removidos = int(transicoes.get("aceita_apenas_legacy", 0))
    adicionados = int(transicoes.get("aceita_apenas_atual", 0))
    diferenca_liquida_simulada = legacy_simulado - atual_simulado
    diferenca_legacy_real = (legacy_real - aceita) if legacy_real is not None else None

    resumo = {
        "pipeline_version": VERSAO_PIPELINE,
        "dataset_version": VERSAO_DATASET,
        "total_pixels_landsat": total,
        "aceitos_pipeline_atual": aceita,
        "legacy_referencia_documentada": int(cfg.total_legacy_referencia),
        "legacy_real_contado_em_disco": legacy_real,
        "legacy_equivalente_simulado": legacy_simulado,
        "aceitos_atual_na_matriz_de_transicao": atual_simulado,
        "aceitos_apenas_legacy": removidos,
        "aceitos_apenas_atual": adicionados,
        "diferenca_liquida_simulada": diferenca_liquida_simulada,
        "diferenca_legacy_real_menos_atual": diferenca_legacy_real,
        "contadores": {k: int(v) for k, v in contadores.items()},
        "transicoes": {k: int(v) for k, v in transicoes.items()},
        "motivos_removidos": {k: int(v) for k, v in motivos_removidos.items()},
    }
    (pasta / "auditoria_filtros.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 88, flush=True)
    print("📋 AUDITORIA DOS FILTROS DE PATCHES", flush=True)
    print("=" * 88, flush=True)
    for linha in linhas:
        print(
            f"{linha['criterio']:<48} {linha['quantidade']:>8} "
            f"({linha['percentual_total']:>6.2f}%)",
            flush=True,
        )
    print("-" * 88, flush=True)
    print(f"Legacy equivalente simulado: {legacy_simulado}", flush=True)
    print(f"Pipeline atual aceito:          {aceita}", flush=True)
    print(f"Removidos do legado:            {removidos}", flush=True)
    print(f"Adicionados pelo atual:         {adicionados}", flush=True)
    print(f"Diferença líquida simulada:     {diferenca_liquida_simulada}", flush=True)
    if legacy_real is not None:
        print(f"Patches .pt no dataset legado:  {legacy_real}", flush=True)
        print(f"Legacy real - atual:            {diferenca_legacy_real}", flush=True)
    else:
        print("Dataset legado não encontrado para contagem direta.", flush=True)
    print("-" * 88, flush=True)
    print("Motivos dos patches aceitos no legado e removidos agora:", flush=True)
    if motivos_removidos:
        for motivo, valor in motivos_removidos.most_common():
            print(f"  - {motivo:<44} {valor:>8}", flush=True)
    else:
        print("  Nenhum.", flush=True)
    print(f"Relatórios salvos em: {pasta}", flush=True)
    print("=" * 88 + "\n", flush=True)
    return resumo


def executar_geracao_dataset_v2(cfg: ConfigDados = CFG.dados) -> pd.DataFrame:
    metadata_path = cfg.dir_dataset_v2 / "metadata.json"
    index_path = cfg.dir_dataset_v2 / "index.csv"
    if index_path.exists() and metadata_path.exists() and not cfg.sobrescrever:
        meta_existente = json.loads(metadata_path.read_text(encoding="utf-8"))
        suporte_meta = meta_existente.get("sar_geometry", {})
        compativel = (
            meta_existente.get("dataset_version") == VERSAO_DATASET
            and suporte_meta.get("schema_version") == cfg.backgeo_schema_version
            and suporte_meta.get("patch_geometry_fraction_minimum")
            == float(cfg.backgeo_fracao_geometrica_minima_patch)
        )
        if compativel:
            print(f"Dataset compatível reutilizado: {cfg.dir_dataset_v2}")
            return pd.read_csv(index_path)
        raise RuntimeError(
            "Dataset existente incompatível. Use sobrescrever=True ou remova "
            f"{cfg.dir_dataset_v2}."
        )

    if cfg.dir_dataset_v2.exists() and cfg.sobrescrever:
        shutil.rmtree(cfg.dir_dataset_v2)
    cfg.dir_dataset_v2.mkdir(parents=True, exist_ok=True)
    pasta_amostras = cfg.dir_dataset_v2 / "amostras"
    pasta_amostras.mkdir(exist_ok=True)

    chm_path = encontrar_chm(cfg)
    stack_opt, qualidade_opt, meta_ls, landsat_info = carregar_landsat_reflectancia(cfg)
    mascara_zero_legacy = carregar_mascara_zero_legacy(cfg)
    legacy_real = contar_patches_legacy(cfg)
    gamma_paths = preparar_mapas_coerencia(cfg)
    suporte_info = preparar_geometria_sar_comum(cfg)
    suporte_mask = np.load(suporte_info["mask_path"], mmap_mode="r")
    suporte_margem = np.load(suporte_info["margin_path"], mmap_mode="r")

    contadores = Counter()
    transicoes = Counter()
    motivos_removidos = Counter()
    removidos_detalhes: list[dict] = []
    adicionados_detalhes: list[dict] = []

    with rasterio.open(chm_path) as src_chm:
        chm_transform, chm_crs = src_chm.transform, src_chm.crs
        chm_h, chm_w = src_chm.height, src_chm.width

        # Memmaps SAR e gamma.
        slc = {}
        gamma = {}
        for banda in cfg.bandas:
            for aq in (cfg.aquisicao_mestre, cfg.aquisicao_escrava):
                slc[(banda, aq)] = [
                    np.load(caminho_slc_ortho(cfg, banda, aq, p), mmap_mode="r")
                    for p in POLARIZACOES
                ]
            gamma[banda] = [
                np.load(gamma_paths[(banda, p)], mmap_mode="r")
                for p in POLARIZACOES
            ]

        # kz opcional, nunca fabricado.
        kz_maps = {}
        for banda, const in (("SAR_L", cfg.kz_l_constante), ("SAR_P", cfg.kz_p_constante)):
            origem = localizar_kz(cfg, banda)
            alinhado = None
            if origem:
                alinhado = alinhar_raster_fisico_para_chm(
                    origem,
                    chm_path,
                    cfg.dir_processados / "Geometria" / f"kz_{banda}_na_grade_chm.npy",
                )
            kz_maps[banda] = carregar_ou_constante(alinhado, const, (chm_h, chm_w))
            print(f"kz {banda}:", origem or const or "AUSENTE — RVoG ficará desativado", flush=True)

        incidencia_maps = {}
        for banda, const in (("SAR_L", cfg.incidencia_l_deg), ("SAR_P", cfg.incidencia_p_deg)):
            origem = localizar_geometria(cfg, banda, "incidencia")
            alinhado = None
            if origem:
                alinhado = alinhar_raster_fisico_para_chm(
                    origem,
                    chm_path,
                    cfg.dir_processados / "Geometria" / f"incidencia_{banda}_na_grade_chm.npy",
                )
            incidencia_maps[banda] = carregar_ou_constante(alinhado, const, (chm_h, chm_w))
            print(
                f"incidência {banda}:",
                origem or const or "AUSENTE — RVoG ficará desativado",
                flush=True,
            )

        linhas_indice = []
        n = 0
        for y in tqdm(range(meta_ls["height"]), desc="Gerando dataset TCCT v6 com geometria .grille"):
            for x in range(meta_ls["width"]):
                contadores["total_pixels_landsat"] += 1
                opt = stack_opt[:, y, x]

                optico_nao_finito = not np.all(np.isfinite(opt))
                qa_invalido = (qualidade_opt[y, x] <= 0) and not optico_nao_finito
                legacy_optico_ok = not bool(mascara_zero_legacy[y, x])

                motivo_atual = None
                if optico_nao_finito:
                    motivo_atual = "rejeitada_optico_nao_finito"
                elif qa_invalido:
                    motivo_atual = "rejeitada_qa_nuvem_sombra_neve_fill"

                # Só é necessário consultar geometria/CHM quando pelo menos um dos
                # dois protocolos (legado ou atual) ainda pode aceitar a célula.
                if motivo_atual is not None and not legacy_optico_ok:
                    contadores[motivo_atual] += 1
                    transicoes["rejeitada_em_ambos"] += 1
                    continue

                # Auditoria legado: reproduz literalmente a geometria e os filtros
                # da extração original, inclusive o uso da janela fracionária e a
                # leitura CHM sem máscara. Isso permite verificar se os 13.247
                # patches históricos são reproduzidos antes de atribuir motivos.
                xmin_ls, ymax_ls = meta_ls["transform"] * (x, y)
                xmax_ls, ymin_ls = meta_ls["transform"] * (x + 1, y + 1)
                aceita_legacy = False
                if legacy_optico_ok:
                    win_legacy = from_bounds(
                        xmin_ls, ymin_ls, xmax_ls, ymax_ls, chm_transform
                    )
                    col_legacy = int(win_legacy.col_off)
                    row_legacy = int(win_legacy.row_off)
                    larg_legacy = int(win_legacy.width)
                    alt_legacy = int(win_legacy.height)
                    dentro_legacy = (
                        col_legacy >= 0
                        and row_legacy >= 0
                        and col_legacy + larg_legacy <= chm_w
                        and row_legacy + alt_legacy <= chm_h
                    )
                    if dentro_legacy:
                        patch_chm_legacy = src_chm.read(1, window=win_legacy)
                        aceita_legacy = bool(
                            patch_chm_legacy.size > 0
                            and not np.isnan(patch_chm_legacy).any()
                            and float(np.mean(patch_chm_legacy)) >= cfg.chm_minimo_medio
                        )

                # Geometria atual: transforma CRS quando necessário e usa janela
                # inteira explícita para manter correspondência determinística.
                xmin, ymin, xmax, ymax = xmin_ls, ymin_ls, xmax_ls, ymax_ls
                if meta_ls["crs"] != chm_crs:
                    xmin, ymin, xmax, ymax = transform_bounds(
                        meta_ls["crs"], chm_crs, xmin, ymin, xmax, ymax, densify_pts=21
                    )
                win = from_bounds(xmin, ymin, xmax, ymax, chm_transform)
                r0, c0 = int(round(win.row_off)), int(round(win.col_off))
                h, w = int(round(win.height)), int(round(win.width))
                janela_valida = (
                    h > 0
                    and w > 0
                    and r0 >= 0
                    and c0 >= 0
                    and r0 + h <= chm_h
                    and c0 + w <= chm_w
                )

                fracao_valida = np.nan
                media_valida = np.nan
                fracao_suporte_sar = np.nan
                margem_sar_media_px = np.nan
                margem_sar_minima_px = np.nan
                suporte_patch = None
                chm = None
                valido_chm = None
                win_i = None

                if janela_valida:
                    win_i = Window(c0, r0, w, h)
                    chm_ma = src_chm.read(1, window=win_i, masked=True)
                    chm = chm_ma.filled(np.nan).astype(np.float32)
                    valido_chm = (
                        np.isfinite(chm)
                        & (chm >= 0)
                        & (chm <= cfg.chm_maximo_valido)
                    )
                    fracao_valida = float(valido_chm.mean()) if chm.size else 0.0
                    media_valida = float(
                        np.nanmean(np.where(valido_chm, chm, np.nan))
                    ) if valido_chm.any() else np.nan
                    suporte_patch = extrair_janela(suporte_mask, win_i).astype(bool)
                    margem_patch = extrair_janela(suporte_margem, win_i).astype(np.float64)
                    if suporte_patch.shape != chm.shape or margem_patch.shape != chm.shape:
                        raise RuntimeError(
                            f"Shape de suporte divergente no patch Landsat ({y},{x}): "
                            f"suporte={suporte_patch.shape}, margem={margem_patch.shape}, "
                            f"CHM={chm.shape}."
                        )
                    fracao_suporte_sar = float(suporte_patch.mean())
                    margem_sar_media_px = float(np.nanmean(margem_patch[suporte_patch])) if suporte_patch.any() else np.nan
                    margem_sar_minima_px = float(np.nanmin(margem_patch[suporte_patch])) if suporte_patch.any() else np.nan

                if motivo_atual is None:
                    if not janela_valida:
                        motivo_atual = "rejeitada_fora_chm_ou_janela_invalida"
                    elif fracao_valida < cfg.fracao_minima_chm_valido:
                        motivo_atual = "rejeitada_chm_incompleto_ou_fora_faixa"
                    elif not np.isfinite(media_valida) or media_valida < cfg.chm_minimo_medio:
                        motivo_atual = "rejeitada_chm_media_baixa"
                    elif (
                        not np.isfinite(fracao_suporte_sar)
                        or fracao_suporte_sar
                        < cfg.backgeo_fracao_geometrica_minima_patch
                    ):
                        motivo_atual = "rejeitada_geometria_sar_invalida"

                aceita_atual = motivo_atual is None

                if aceita_legacy and aceita_atual:
                    transicoes["aceita_em_ambos"] += 1
                elif aceita_legacy and not aceita_atual:
                    transicoes["aceita_apenas_legacy"] += 1
                    motivos_removidos[motivo_atual] += 1
                    if cfg.salvar_auditoria_detalhada:
                        removidos_detalhes.append(
                            _linha_auditoria(
                                y, x, meta_ls, motivo_atual, True, False,
                                fracao_valida, media_valida, r0, c0, h, w,
                            )
                        )
                elif aceita_atual and not aceita_legacy:
                    transicoes["aceita_apenas_atual"] += 1
                    if cfg.salvar_auditoria_detalhada:
                        adicionados_detalhes.append(
                            _linha_auditoria(
                                y, x, meta_ls, "aceita_apenas_atual", False, True,
                                fracao_valida, media_valida, r0, c0, h, w,
                            )
                        )
                else:
                    transicoes["rejeitada_em_ambos"] += 1

                if not aceita_atual:
                    contadores[motivo_atual] += 1
                    continue

                amostra = {
                    "tcct_dataset_version": torch.tensor(VERSAO_DATASET, dtype=torch.int16),
                    "optical": torch.from_numpy(opt.astype(np.float32)),
                    "optical_quality": torch.tensor(
                        [float(qualidade_opt[y, x])], dtype=torch.float32
                    ),
                    "landsat_source_index": torch.tensor(
                        [int(landsat_info["source_index"][y, x])],
                        dtype=torch.int16,
                    ),
                    "landsat_scene_valid": torch.from_numpy(
                        landsat_info["scene_valid"][:, y, x].astype(np.bool_)
                    ),
                    "chm": redimensionar_real(
                        np.where(valido_chm, chm, 0.0), cfg.tamanho_patch
                    ),
                    "valid_mask": redimensionar_real(
                        valido_chm.astype(np.float32),
                        cfg.tamanho_patch,
                        modo="nearest",
                    ) > 0.5,
                    "landsat_rowcol": torch.tensor([y, x], dtype=torch.int32),
                    "chm_window": torch.tensor([r0, c0, h, w], dtype=torch.int32),
                    "sar_geometry_mask": redimensionar_real(
                        suporte_patch.astype(np.float32),
                        cfg.tamanho_patch,
                        modo="nearest",
                    ) > 0.5,
                    "sar_geometry_fraction": torch.tensor(
                        [fracao_suporte_sar], dtype=torch.float32
                    ),
                    "sar_geometry_margin_mean_px": torch.tensor(
                        [margem_sar_media_px], dtype=torch.float32
                    ),
                    "sar_geometry_margin_min_px": torch.tensor(
                        [margem_sar_minima_px], dtype=torch.float32
                    ),
                }

                if cfg.landsat_salvar_stack_cenas_no_patch:
                    amostra["optical_scene_stack"] = torch.from_numpy(
                        landsat_info["scene_stack"][:, :, y, x].astype(np.float32)
                    )

                fonte_idx = int(landsat_info["source_index"][y, x])
                fonte_nome = (
                    landsat_info["scenes"][fonte_idx]["name"]
                    if fonte_idx >= 0 else ""
                )
                fonte_data = (
                    landsat_info["scenes"][fonte_idx]["date"]
                    if fonte_idx >= 0 else None
                )

                fisica_valida = True
                for banda, chave in (("SAR_L", "l"), ("SAR_P", "p")):
                    gpatch = np.stack([extrair_janela(a, win_i) for a in gamma[banda]])
                    mpatch = np.stack([
                        extrair_janela(a, win_i)
                        for a in slc[(banda, cfg.aquisicao_mestre)]
                    ])
                    epatch = np.stack([
                        extrair_janela(a, win_i)
                        for a in slc[(banda, cfg.aquisicao_escrava)]
                    ])
                    t11, t22, omega = matrizes_polinsar_patch(mpatch, epatch)
                    gdiag = gamma_diagonal_matrizes(t11, t22, omega)
                    canon = correlacoes_canonicas(t11, t22, omega)
                    gvol = np.nanmean(gdiag[[1, 2]])
                    gground = np.nanmean(gdiag[[0, 3]])
                    phi0 = np.angle(gground) if np.isfinite(gground) else np.nan
                    gvol_derot = (
                        gvol * np.exp(-1j * phi0)
                        if np.isfinite(phi0)
                        else np.nan + 1j * np.nan
                    )

                    amostra[f"sar_{chave}"] = redimensionar_complexo(
                        gpatch, cfg.tamanho_patch
                    )
                    amostra[f"gamma_patch_{chave}"] = torch.from_numpy(gdiag)
                    amostra[f"gamma_volume_{chave}"] = torch.tensor(
                        [gvol], dtype=torch.complex64
                    )
                    amostra[f"gamma_volume_derot_{chave}"] = torch.tensor(
                        [gvol_derot], dtype=torch.complex64
                    )
                    amostra[f"ground_phase_{chave}"] = torch.tensor(
                        [phi0], dtype=torch.float32
                    )
                    amostra[f"canonical_corr_{chave}"] = torch.from_numpy(canon)
                    amostra[f"sar_quality_{chave}"] = torch.tensor(
                        [float(np.nanmean(np.abs(gpatch)))], dtype=torch.float32
                    )
                    if cfg.salvar_matrizes_polinsar:
                        amostra[f"T11_{chave}"] = torch.from_numpy(t11)
                        amostra[f"T22_{chave}"] = torch.from_numpy(t22)
                        amostra[f"Omega12_{chave}"] = torch.from_numpy(omega)

                    kz_obj = kz_maps[banda]
                    if isinstance(kz_obj, float):
                        kz_val = kz_obj
                    elif kz_obj is None:
                        kz_val = np.nan
                    else:
                        kz_val = float(np.nanmean(extrair_janela(kz_obj, win_i)))
                    amostra[f"kz_{chave}"] = torch.tensor([kz_val], dtype=torch.float32)

                    theta_obj = incidencia_maps[banda]
                    if isinstance(theta_obj, float):
                        theta_val = theta_obj
                    elif theta_obj is None:
                        theta_val = np.nan
                    else:
                        theta_val = float(np.nanmean(extrair_janela(theta_obj, win_i)))
                    amostra[f"theta_{chave}"] = torch.tensor(
                        [theta_val], dtype=torch.float32
                    )
                    fisica_valida &= (
                        np.isfinite(kz_val)
                        and np.isfinite(gvol)
                        and np.isfinite(theta_val)
                    )

                amostra["physics_valid"] = torch.tensor(
                    [fisica_valida], dtype=torch.bool
                )
                nome = f"patch_{n:06d}.pt"
                destino = pasta_amostras / nome
                if cfg.sobrescrever or not destino.exists():
                    torch.save(amostra, destino)

                centro_x, centro_y = meta_ls["transform"] * (x + 0.5, y + 0.5)
                br, bc = y // cfg.tamanho_bloco_landsat, x // cfg.tamanho_bloco_landsat
                linhas_indice.append({
                    "id": n,
                    "arquivo": str(destino.relative_to(cfg.dir_dataset_v2)),
                    "landsat_row": y,
                    "landsat_col": x,
                    "landsat_source_index": fonte_idx,
                    "landsat_source_scene": fonte_nome,
                    "landsat_source_date": fonte_data,
                    "landsat_quality_local": float(qualidade_opt[y, x]),
                    "x": centro_x,
                    "y": centro_y,
                    "block_row": br,
                    "block_col": bc,
                    "group": f"{br}_{bc}",
                    "chm_row": r0,
                    "chm_col": c0,
                    "chm_h": h,
                    "chm_w": w,
                    "sar_geometry_fraction": fracao_suporte_sar,
                    "sar_geometry_margin_mean_px": margem_sar_media_px,
                    "sar_geometry_margin_min_px": margem_sar_minima_px,
                    "sar_geometry_schema_version": int(cfg.backgeo_schema_version),
                    "physics_valid": fisica_valida,
                })
                contadores["aceita"] += 1
                n += 1

    df = pd.DataFrame(linhas_indice)
    df.to_csv(cfg.dir_dataset_v2 / "index.csv", index=False)
    resumo_auditoria = salvar_auditoria_filtros(
        cfg,
        contadores,
        transicoes,
        motivos_removidos,
        removidos_detalhes,
        adicionados_detalhes,
        legacy_real,
    )
    metadados = {
        "pipeline_version": VERSAO_PIPELINE,
        "dataset_version": VERSAO_DATASET,
        "created_at": pd.Timestamp.now().isoformat(),
        "n_samples": len(df),
        "landsat_mode": cfg.landsat_modo,
        "landsat_scenes": landsat_info["scenes"],
        "landsat_selection_report": landsat_info["report"],
        "master": cfg.aquisicao_mestre,
        "slave": cfg.aquisicao_escrava,
        "coherence_window": cfg.janela_coerencia,
        "patch_size": cfg.tamanho_patch,
        "landsat_scaling": "SR = DN*0.0000275 - 0.2 (ou coeficientes MTL)",
        "input_sar": "coerência complexa normalizada por polarização",
        "sar_geometry": {
            "schema_version": int(cfg.backgeo_schema_version),
            "patch_geometry_fraction_minimum": float(
                cfg.backgeo_fracao_geometrica_minima_patch
            ),
            "mask_path": str(suporte_info["mask_path"]),
            "margin_path": str(suporte_info["margin_path"]),
            "report_path": str(suporte_info["report_path"]),
            "policy": "intersection of four .grille inversions; no extrapolated or out-of-SLC pixel enters a canonical patch",
        },
        "filter_audit": resumo_auditoria,
    }
    (cfg.dir_dataset_v2 / "metadata.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Dataset v6 com geometria .grille concluído: {len(df)} amostras em {cfg.dir_dataset_v2}",
        flush=True,
    )
    return df


# Esta etapa regenera os tensores e produz os contadores de exclusão:
# indice_v2 = executar_geracao_dataset_v2()





# ================================================================
# 3B. GEOMETRIA RVoG: METADADOS OU CALIBRAÇÃO EXCLUSIVA NO TREINO
# ================================================================
_GEOMETRIA_COLUNAS = (
    "kz_l", "kz_p", "theta_l", "theta_p", "rvog_a_l", "rvog_a_p",
    "physics_valid_l", "physics_valid_p", "physics_valid", "physics_source",
)


def _float_texto(valor: str) -> float:
    return float(valor.strip().replace(",", "."))


def _ler_texto_metadado(path: Path, limite_bytes: int = 4_000_000) -> str:
    try:
        dados = path.read_bytes()[:limite_bytes]
    except OSError:
        return ""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return dados.decode(enc)
        except UnicodeDecodeError:
            pass
    return dados.decode("latin-1", errors="ignore")


def _converter_unidade(valor: float, contexto: str, tipo: str) -> float:
    c = contexto.lower().replace("µ", "u")
    if tipo == "incidencia":
        if "rad" in c and "deg" not in c and "grau" not in c:
            return math.degrees(valor)
        return valor
    if tipo == "frequencia":
        if "ghz" in c:
            return valor * 1e9
        if "mhz" in c:
            return valor * 1e6
        if "khz" in c:
            return valor * 1e3
        return valor
    if tipo in {"comprimento_onda", "baseline", "slant_range"}:
        if " km" in c or "kilomet" in c:
            return valor * 1000.0
        if " cm" in c or "centimet" in c:
            return valor / 100.0
        if " mm" in c or "millimet" in c:
            return valor / 1000.0
        return valor
    return valor


def _extrair_candidatos_rotulados(texto: str, tipo: str) -> list[tuple[float, str]]:
    rotulos = {
        "kz": r"(?:k\s*[_-]?\s*z|vertical\s+wavenumber|nombre\s+d['’]?onde\s+vertical)",
        "incidencia": r"(?:incidence\s*angle|angle\s+d['’]?incidence|incid[eê]ncia|look\s*angle|theta)",
        "baseline": r"(?:perpendicular\s+baseline|baseline\s+perp(?:endicular)?|b\s*[_-]?\s*perp)",
        "slant_range": r"(?:slant\s+range|range\s+distance|distance\s+oblique)",
        "comprimento_onda": r"(?:wavelength|lambda|longueur\s+d['’]?onde|comprimento\s+de\s+onda)",
        "frequencia": r"(?:frequency|fr[eé]quence|frequ[eê]ncia)",
    }
    rotulo = rotulos[tipo]
    numero = r"([-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?)"
    padrao = re.compile(rotulo + r"[^\n\r0-9+-]{0,60}" + numero, re.I)
    saida = []
    for m in padrao.finditer(texto):
        ini, fim = max(0, m.start()-80), min(len(texto), m.end()+100)
        contexto = texto[ini:fim]
        try:
            v = _float_texto(m.group(1))
            v = _converter_unidade(v, contexto, tipo)
            saida.append((v, contexto.replace("\n", " ")[:240]))
        except Exception:
            continue
    return saida


def _plausivel_geometria(tipo: str, valor: float, banda: str) -> bool:
    if not np.isfinite(valor):
        return False
    if tipo == "kz":
        return 1e-4 <= abs(valor) <= 2.0
    if tipo == "incidencia":
        return 5.0 <= valor <= 80.0
    if tipo == "baseline":
        return 0.01 <= abs(valor) <= 5000.0
    if tipo == "slant_range":
        return 100.0 <= valor <= 200000.0
    if tipo == "comprimento_onda":
        if banda == "l":
            return 0.10 <= valor <= 0.40
        return 0.30 <= valor <= 1.50
    if tipo == "frequencia":
        if banda == "l":
            return 0.7e9 <= valor <= 2.0e9
        return 0.2e9 <= valor <= 1.0e9
    return True


def _arquivos_metadados_geometria(cfg: ConfigDados) -> list[Path]:
    extensoes = {".xml", ".hdr", ".ann", ".txt", ".json", ".par", ".cfg", ".aux"}
    candidatos = []
    for raiz in (cfg.dir_brutos, cfg.dir_processados):
        if not raiz.exists():
            continue
        for p in raiz.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in extensoes:
                continue
            # Evita ler SLCs e LUTs muito grandes como texto integral.
            nome = p.name.lower()
            if "slc" in nome and p.stat().st_size > 10_000_000:
                continue
            candidatos.append(p)
    return sorted(set(candidatos))


def resolver_geometria_metadados(cfg: ConfigDados, banda: str, out: Path) -> dict:
    """Procura kz/incidência diretos e, quando possível, deriva kz da geometria.

    Fórmula usada quando B_perp, lambda, R e theta estão disponíveis:
        kz = 4*pi*B_perp / (lambda*R*sin(theta))
    O sinal de B_perp é preservado.
    """
    chave = "l" if banda == "SAR_L" else "p"
    token_banda = "l" if chave == "l" else "p"
    linhas = []
    arquivos = _arquivos_metadados_geometria(cfg)
    for p in arquivos:
        caminho_lower = str(p).lower()
        # Arquivos em pastas explicitamente associadas à banda recebem prioridade.
        score_banda = 2 if f"sar_{token_banda}" in caminho_lower else 0
        score_banda += 1 if re.search(rf"(?:^|[\\/_-]){token_banda}(?:[\\/_-]|band|banda)", caminho_lower) else 0
        texto = _ler_texto_metadado(p)
        if not texto:
            continue
        score_aquisicao = int(cfg.aquisicao_mestre.lower() in caminho_lower or cfg.aquisicao_mestre.lower() in texto.lower())
        score_aquisicao += int(cfg.aquisicao_escrava.lower() in caminho_lower or cfg.aquisicao_escrava.lower() in texto.lower())
        for tipo in ("kz", "incidencia", "baseline", "slant_range", "comprimento_onda", "frequencia"):
            for valor, contexto in _extrair_candidatos_rotulados(texto, tipo):
                if _plausivel_geometria(tipo, valor, chave):
                    linhas.append({
                        "banda": chave, "tipo": tipo, "valor": float(valor),
                        "arquivo": str(p), "score": score_banda + score_aquisicao,
                        "contexto": contexto,
                    })
    df = pd.DataFrame(linhas)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"candidatos_geometria_{chave}.csv", index=False)
    if df.empty:
        return {"banda": chave, "source": "ausente", "kz": np.nan, "theta": np.nan}

    def escolher(tipo: str) -> float:
        sub = df[df.tipo == tipo].copy()
        if sub.empty:
            return np.nan
        sub = sub.sort_values(["score"], ascending=False)
        maior = sub.score.max()
        vals = sub.loc[sub.score == maior, "valor"].to_numpy(float)
        med = float(np.median(vals))
        # Valores conflitantes em mais de 20% são considerados ambíguos.
        if len(vals) > 1 and np.ptp(vals) > 0.20 * max(abs(med), 1e-8):
            return np.nan
        return med

    kz = escolher("kz")
    theta = escolher("incidencia")
    origem = "metadado_direto"
    if not np.isfinite(kz):
        bperp = escolher("baseline")
        r = escolher("slant_range")
        lamb = escolher("comprimento_onda")
        freq = escolher("frequencia")
        if not np.isfinite(lamb) and np.isfinite(freq) and freq > 0:
            lamb = 299_792_458.0 / freq
        if all(np.isfinite(v) for v in (bperp, r, lamb, theta)):
            kz = 4.0 * math.pi * bperp / (lamb * r * math.sin(math.radians(theta)))
            origem = "derivado_de_baseline"
    if not _plausivel_geometria("kz", kz, chave):
        kz = np.nan
    if not _plausivel_geometria("incidencia", theta, chave):
        theta = np.nan
    return {"banda": chave, "source": origem if np.isfinite(kz) else "ausente", "kz": float(kz), "theta": float(theta)}


def rvog_complexo_numpy(h: np.ndarray, kz: float, a: float) -> np.ndarray:
    h = np.clip(np.asarray(h, dtype=np.float64), 0.1, 60.0)
    a = float(max(a, 1e-6))
    kz = float(kz)
    q = np.exp(-a * h)
    phase = kz * h
    den1 = max(a*a + kz*kz, 1e-12)
    alpha = (a*a - 1j*a*kz) / den1
    beta = (np.exp(1j*phase) - q) / np.maximum(1.0 - q, 1e-8)
    return alpha * beta


def _huber_numpy(residuo: np.ndarray, delta: float = 0.10) -> float:
    x = np.abs(residuo)
    return float(np.mean(np.where(x <= delta, 0.5*x*x, delta*(x - 0.5*delta))))


def _objetivo_rvog_parametros(params, h, gamma, modo: str) -> float:
    kz, loga = float(params[0]), float(params[1])
    a = math.exp(loga)
    pred = rvog_complexo_numpy(h, kz, a)
    if modo == "magnitude":
        return _huber_numpy(np.abs(pred) - np.abs(gamma), delta=0.08)
    res = np.concatenate([(pred.real-gamma.real), (pred.imag-gamma.imag)])
    return _huber_numpy(res, delta=0.08)


def calibrar_rvog_efetivo(h, gamma, cfg: ConfigDados, banda: str, out: Path) -> dict:
    h = np.asarray(h, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.complex128)
    mag = np.abs(gamma)
    valid = (
        np.isfinite(h) & np.isfinite(gamma.real) & np.isfinite(gamma.imag)
        & (mag >= cfg.gamma_min_calibracao) & (mag <= cfg.gamma_max_calibracao)
        & (h >= cfg.chm_minimo_medio) & (h <= cfg.chm_maximo_valido)
    )
    h, gamma = h[valid], gamma[valid]
    if len(h) < 64:
        return {"success": False, "reason": f"apenas {len(h)} observações coerentes", "banda": banda}

    rng = np.random.default_rng(42)
    if len(h) > cfg.max_amostras_calibracao_rvog:
        idx = rng.choice(len(h), cfg.max_amostras_calibracao_rvog, replace=False)
        h, gamma = h[idx], gamma[idx]

    kzmin, kzmax = cfg.kz_abs_min_calibracao, cfg.kz_abs_max_calibracao
    amin, amax = cfg.atenuacao_min_calibracao, cfg.atenuacao_max_calibracao
    kzs = np.r_[np.linspace(-kzmax, -kzmin, 36), np.linspace(kzmin, kzmax, 36)]
    avals = np.geomspace(amin, amax, 22)

    resultados = []
    for modo in ("complexa", "magnitude"):
        melhores = []
        for kz in kzs:
            for a in avals:
                loss = _objetivo_rvog_parametros((kz, math.log(a)), h, gamma, modo)
                melhores.append((loss, kz, a))
        melhores.sort(key=lambda x: x[0])
        candidatos = []
        for _, kz0, a0 in melhores[:6]:
            res = minimize(
                _objetivo_rvog_parametros,
                x0=np.array([kz0, math.log(a0)]),
                args=(h, gamma, modo),
                method="L-BFGS-B",
                bounds=[(-kzmax, kzmax), (math.log(amin), math.log(amax))],
                options={"maxiter": 250},
            )
            kz = float(res.x[0]); a = float(math.exp(res.x[1]))
            if abs(kz) < kzmin:
                continue
            pred = rvog_complexo_numpy(h, kz, a)
            rmse_complexo = float(np.sqrt(np.mean(np.abs(pred-gamma)**2)))
            rmse_magnitude = float(np.sqrt(np.mean((np.abs(pred)-np.abs(gamma))**2)))
            candidatos.append({
                "modo": modo, "loss": float(res.fun), "kz": kz, "a": a,
                "rmse_complexo": rmse_complexo, "rmse_magnitude": rmse_magnitude,
                "n": int(len(h)), "success_opt": bool(res.success),
            })
        if candidatos:
            resultados.append(min(candidatos, key=lambda d: d["loss"]))

    if not resultados:
        return {"success": False, "reason": "otimização sem solução", "banda": banda}

    complexo = next((r for r in resultados if r["modo"] == "complexa"), None)
    magnitude = next((r for r in resultados if r["modo"] == "magnitude"), None)
    if complexo and complexo["rmse_complexo"] <= cfg.rmse_complexo_max_calibracao:
        escolhido = complexo
        modo_loss = "complexa_derotada"
    elif magnitude and magnitude["rmse_magnitude"] <= cfg.rmse_magnitude_max_calibracao:
        escolhido = magnitude
        modo_loss = "magnitude"
    else:
        escolhido = min(resultados, key=lambda d: d["rmse_magnitude"])
        return {
            "success": False, "reason": "ajuste RVoG efetivo acima do erro máximo",
            "banda": banda, "diagnostico": escolhido,
        }

    escolhido = dict(escolhido)
    escolhido.update({"success": True, "banda": banda, "modo_loss": modo_loss, "source": "calibracao_exclusiva_treino"})
    out.mkdir(parents=True, exist_ok=True)
    (out / f"calibracao_rvog_{banda}.json").write_text(
        json.dumps(escolhido, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return escolhido


def extrair_observacoes_rvog(dataset, indices: Sequence[int], max_n: int | None = None):
    indices = np.asarray(indices, dtype=int)
    if max_n and len(indices) > max_n:
        rng = np.random.default_rng(CFG.treino.seed)
        indices = rng.choice(indices, max_n, replace=False)
    dados = {"l": {"h": [], "gamma": []}, "p": {"h": [], "gamma": []}}
    for i in tqdm(indices, desc="Coletando coerência para calibrar RVoG", leave=False):
        a = dataset[int(i)]
        y = a["chm"].numpy().astype(np.float64)
        m = a["valid_mask"].numpy().astype(bool)
        h = float(np.mean(y[m])) if np.any(m) else np.nan
        for b in ("l", "p"):
            z = a[f"gamma_volume_derot_{b}"].reshape(-1)[0].item()
            dados[b]["h"].append(h)
            dados[b]["gamma"].append(complex(z))
    for b in dados:
        dados[b]["h"] = np.asarray(dados[b]["h"], dtype=np.float64)
        dados[b]["gamma"] = np.asarray(dados[b]["gamma"], dtype=np.complex128)
    return dados


def _overlay_constante(dataset, kz_l=np.nan, kz_p=np.nan, theta_l=np.nan, theta_p=np.nan,
                       a_l=np.nan, a_p=np.nan, source="ausente") -> pd.DataFrame:
    df = dataset.index[["id"]].copy() if "id" in dataset.index.columns else pd.DataFrame({"id": np.arange(len(dataset))})
    for b, kz, theta, a in (("l", kz_l, theta_l, a_l), ("p", kz_p, theta_p, a_p)):
        df[f"kz_{b}"] = float(kz)
        df[f"theta_{b}"] = float(theta)
        df[f"rvog_a_{b}"] = float(a)
        df[f"physics_valid_{b}"] = False
    df["physics_valid"] = False
    df["physics_source"] = source
    return df


def _validar_gamma_amostra(amostra: dict, banda: str) -> bool:
    try:
        z = amostra[f"gamma_volume_derot_{banda}"].reshape(-1)[0]
        mag = float(torch.abs(z).item())
        return np.isfinite(mag) and CFG.dados.gamma_min_calibracao <= mag <= CFG.dados.gamma_max_calibracao
    except Exception:
        return False


def _mapa_geometria_alinhado(cfg: ConfigDados, banda: str, tipo: str):
    origem = localizar_kz(cfg, banda) if tipo == "kz" else localizar_geometria(cfg, banda, "incidencia")
    if origem is None:
        return None, None
    try:
        chm_path = encontrar_chm(cfg)
        saida = cfg.dir_processados / "Geometria" / f"{tipo}_{banda}_na_grade_chm_v27.npy"
        alinhado = alinhar_raster_fisico_para_chm(origem, chm_path, saida)
        return np.load(alinhado, mmap_mode="r"), origem
    except Exception as exc:
        warnings.warn(f"Não foi possível alinhar {tipo} de {banda}: {exc}")
        return None, origem


def _media_mapa_por_linha_indice(mapa, row: pd.Series) -> float:
    if mapa is None:
        return np.nan
    try:
        r0, c0 = int(row["chm_row"]), int(row["chm_col"])
        h, w = int(row["chm_h"]), int(row["chm_w"])
        rec = np.asarray(mapa[r0:r0+h, c0:c0+w], dtype=np.float64)
        return float(np.nanmean(rec)) if np.isfinite(rec).any() else np.nan
    except Exception:
        return np.nan



def _caminho_cfg_relativo(cfg: ConfigDados, valor: str) -> Path:
    """Resolve produtos físicos somente dentro de SIBGRAPI_REV.

    A v2.8.3 não procura cópias na pasta-pai, em TCCT_Geral ou em outros
    volumes. Se o arquivo não estiver em Dados_Brutos sob a raiz isolada,
    a execução falha de forma explícita.
    """
    p = Path(valor)
    candidato = p if p.is_absolute() else (cfg.raiz / p)
    try:
        candidato_resolvido = candidato.resolve()
        raiz_resolvida = cfg.raiz.resolve()
        candidato_resolvido.relative_to(raiz_resolvida)
    except ValueError as exc:
        raise RuntimeError(
            f"Produto físico fora da raiz SIBGRAPI_REV: {candidato}"
        ) from exc
    return candidato


def _validar_arquivo_ha(path: Path, altura: int, largura: int) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Produto Ha ausente: {path}")
    esperado = int(altura) * int(largura) * np.dtype(np.float32).itemsize
    tamanho = path.stat().st_size
    if tamanho != esperado:
        raise ValueError(
            f"Tamanho incompatível para {path.name}: "
            f"{tamanho:,} bytes; esperado={esperado:,} "
            f"({altura}×{largura}×float32)."
        )


def _shapes_fatorados_compativeis(
    total_pixels: int,
    min_altura: int,
    min_largura: int,
) -> list[tuple[int, int]]:
    """Enumera shapes inteiros compatíveis com tamanho e extensão da LUT."""
    if total_pixels <= 0:
        raise ValueError(f"Total de pixels inválido: {total_pixels}")
    if min_altura <= 0 or min_largura <= 0:
        raise ValueError(
            f"Dimensões mínimas inválidas: {min_altura}×{min_largura}"
        )

    candidatos = set()
    limite = math.isqrt(total_pixels)

    for divisor in range(1, limite + 1):
        if total_pixels % divisor:
            continue

        quociente = total_pixels // divisor
        for altura, largura in (
            (divisor, quociente),
            (quociente, divisor),
        ):
            if altura >= min_altura and largura >= min_largura:
                candidatos.add((int(altura), int(largura)))

    return sorted(candidatos)


def inferir_layout_ha_p(
    ha_0506_path: Path,
    ha_0603_path: Path,
    lut=None,
    cfg: ConfigDados = CFG.dados,
) -> dict:
    """Valida o layout Ha-P congelado: 7478×4000 float32, sem cabeçalho."""
    altura, largura = LAYOUT_HA_P_APROVADO["shape"]
    dtype = np.dtype(LAYOUT_HA_P_APROVADO["dtype"])
    esperado = int(altura * largura * dtype.itemsize)
    registros = []
    for path in (Path(ha_0506_path), Path(ha_0603_path)):
        if not path.exists():
            raise FileNotFoundError(path)
        real = int(path.stat().st_size)
        if real != esperado:
            raise RuntimeError(
                f"Ha incompatível: {path}; real={real:,}, esperado={esperado:,} "
                f"para shape={altura}×{largura}, dtype={dtype}."
            )
        registros.append({"arquivo": str(path), "tamanho_bytes": real})
    return {
        "pipeline_version": VERSAO_PIPELINE,
        "shape_escolhido": [int(altura), int(largura)],
        "dtype": str(dtype),
        "offset_bytes": 0,
        "tamanho_esperado_bytes": esperado,
        "origem_shape": "INF_0007_0506_0603_e_tamanho_binario_aprovados",
        "arquivos": registros,
    }


def auditar_layout_ha_p(
    cfg: ConfigDados = CFG.dados,
) -> dict:
    ha_0506 = resolver_entrada_sar_bruta(
        cfg, "SAR_P", cfg.aquisicao_mestre, "ha"
    )
    ha_0603 = resolver_entrada_sar_bruta(
        cfg, "SAR_P", cfg.aquisicao_escrava, "ha"
    )
    resumo = inferir_layout_ha_p(ha_0506, ha_0603, cfg=cfg)
    grille_0007 = resolver_entrada_sar_bruta(
        cfg, "SAR_P", "tropi0007", "grille"
    )
    estrutura = ler_grille_estruturada(grille_0007)
    resumo.update({
        "grille_referencia_ha": str(grille_0007),
        "grille_row_min": float(estrutura["linhas_sar"].min()),
        "grille_row_max": float(estrutura["linhas_sar"].max()),
        "grille_col_min": float(estrutura["colunas_sar"].min()),
        "grille_col_max": float(estrutura["colunas_sar"].max()),
        "interpretacao": (
            "Os dois Ha usam a grade radar da referência tropi0007; 0007 "
            "cancela-se na diferença de kz e não entra como modalidade do TCCT."
        ),
    })
    out = cfg.dir_auditoria / "Geometria_RVoG_Grille3D"
    out.mkdir(parents=True, exist_ok=True)
    (out / "auditoria_layout_ha_p.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nLayout Ha-P aprovado:")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return resumo


def _amostrar_float_bilinear_por_mapa(
    bruto: np.memmap,
    mapa_rows,
    mapa_cols,
    mascara,
    r0: int,
    r1: int,
) -> np.ndarray:
    rows = np.asarray(mapa_rows[r0:r1], dtype=np.float64)
    cols = np.asarray(mapa_cols[r0:r1], dtype=np.float64)
    valid = np.asarray(mascara[r0:r1], dtype=bool)
    out = np.full(rows.shape, np.nan, dtype=np.float64)
    if not valid.any():
        return out
    rv = rows[valid]
    cv = cols[valid]
    rr = np.floor(rv).astype(np.int64)
    cc = np.floor(cv).astype(np.int64)
    dr = rv - rr
    dc = cv - cc
    v00 = np.asarray(bruto[rr, cc], dtype=np.float64)
    v01 = np.asarray(bruto[rr, cc + 1], dtype=np.float64)
    v10 = np.asarray(bruto[rr + 1, cc], dtype=np.float64)
    v11 = np.asarray(bruto[rr + 1, cc + 1], dtype=np.float64)
    vals = (
        (1 - dr) * (1 - dc) * v00
        + (1 - dr) * dc * v01
        + dr * (1 - dc) * v10
        + dr * dc * v11
    )
    idx = np.flatnonzero(valid)
    flat = out.ravel()
    flat[idx] = vals
    return flat.reshape(out.shape)


def gerar_kz_p_0506_0603_na_grade_chm(
    cfg: ConfigDados,
    out: Path,
    sobrescrever: bool = False,
) -> tuple[Path, dict]:
    """Deriva kz-P dos dois Ha e os geocodifica pela .grille de tropi0007."""
    pasta = cfg.dir_processados / "Geometria_RVoG_Grille3D"
    pasta.mkdir(parents=True, exist_ok=True)
    saida = pasta / "kz_SAR_P_0506_0603_Ha_grade_CHM_v3_2_0.npy"
    resumo_path = pasta / "resumo_kz_p_ha_0506_0603_v3_2_0.json"
    if saida.exists() and resumo_path.exists() and not sobrescrever:
        resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
        if resumo.get("pipeline_version") == VERSAO_PIPELINE:
            return saida, resumo

    ha_0506_path = resolver_entrada_sar_bruta(
        cfg, "SAR_P", cfg.aquisicao_mestre, "ha"
    )
    ha_0603_path = resolver_entrada_sar_bruta(
        cfg, "SAR_P", cfg.aquisicao_escrava, "ha"
    )
    layout = inferir_layout_ha_p(ha_0506_path, ha_0603_path, cfg=cfg)
    altura, largura = map(int, layout["shape_escolhido"])
    grille_0007 = resolver_entrada_sar_bruta(
        cfg, "SAR_P", "tropi0007", "grille"
    )
    mapa = preparar_mapa_inverso_grille(
        cfg,
        "SAR_P",
        "tropi0007",
        (altura, largura),
        finalidade="ha_referencia_0007",
        grille_path=grille_0007,
        sobrescrever=sobrescrever,
    )
    comum = preparar_geometria_sar_comum(cfg, sobrescrever=False)
    common_mask = np.load(comum["mask_path"], mmap_mode="r")
    mapa_mask = np.load(mapa["mask"], mmap_mode="r")
    mapa_rows = np.load(mapa["row"], mmap_mode="r")
    mapa_cols = np.load(mapa["col"], mmap_mode="r")
    if common_mask.shape != mapa_mask.shape:
        raise RuntimeError("Máscaras SLC e Ha possuem shapes diferentes.")

    dtype = np.dtype(cfg.ha_p_dtype)
    ha1 = np.memmap(
        ha_0506_path, dtype=dtype, mode="r", shape=(altura, largura)
    )
    ha2 = np.memmap(
        ha_0603_path, dtype=dtype, mode="r", shape=(altura, largura)
    )
    # Validação determinística de byte order e plausibilidade.
    for nome, arr in (("0506", ha1), ("0603", ha2)):
        am = np.asarray(arr[::31, ::31], dtype=np.float64)
        plaus = (
            np.isfinite(am)
            & (np.abs(am) < cfg.ha_nodata_abs)
            & (np.abs(am) > cfg.ha_abs_minimo)
        )
        if plaus.mean() < 0.90:
            raise RuntimeError(
                f"Ha {nome} não passou na validação de dtype/NoData: "
                f"fração plausível={plaus.mean():.3f}."
            )

    H, W = mapa_mask.shape
    kz_out = np.lib.format.open_memmap(
        saida, mode="w+", dtype=np.float32, shape=(H, W)
    )
    kz_out[:] = np.nan
    cont = Counter()
    samples = []

    for r0 in tqdm(
        range(0, H, cfg.backgeo_chunk_linhas),
        desc="Ha-P → kz-P na grade CHM",
    ):
        r1 = min(H, r0 + cfg.backgeo_chunk_linhas)
        h1 = _amostrar_float_bilinear_por_mapa(
            ha1, mapa_rows, mapa_cols, mapa_mask, r0, r1
        )
        h2 = _amostrar_float_bilinear_por_mapa(
            ha2, mapa_rows, mapa_cols, mapa_mask, r0, r1
        )
        v1 = (
            np.isfinite(h1)
            & (np.abs(h1) < cfg.ha_nodata_abs)
            & (np.abs(h1) > cfg.ha_abs_minimo)
        )
        v2 = (
            np.isfinite(h2)
            & (np.abs(h2) < cfg.ha_nodata_abs)
            & (np.abs(h2) > cfg.ha_abs_minimo)
        )
        geom = np.asarray(common_mask[r0:r1], dtype=bool) & np.asarray(
            mapa_mask[r0:r1], dtype=bool
        )
        valid_ha = v1 & v2 & geom
        kz = np.full(h1.shape, np.nan, dtype=np.float64)
        kz[valid_ha] = (
            2.0 * np.pi / h2[valid_ha]
            - 2.0 * np.pi / h1[valid_ha]
        )
        valid_kz = (
            valid_ha
            & np.isfinite(kz)
            & (np.abs(kz) >= cfg.kz_p_abs_minimo_produto)
            & (np.abs(kz) <= cfg.kz_p_abs_maximo_produto)
        )
        kz_out[r0:r1] = np.where(valid_kz, kz, np.nan).astype(np.float32)
        cont["pixels_total"] += int(kz.size)
        cont["fora_geometria_comum"] += int((~geom).sum())
        cont["ha_0506_invalido"] += int((~v1).sum())
        cont["ha_0603_invalido"] += int((~v2).sum())
        cont["kz_valido"] += int(valid_kz.sum())
        am = kz[::17, ::17]
        av = valid_kz[::17, ::17]
        if av.any():
            samples.append(am[av])

    kz_out.flush()
    del kz_out, ha1, ha2, mapa_rows, mapa_cols, mapa_mask, common_mask
    gc.collect()

    sample = np.concatenate(samples) if samples else np.array([], dtype=np.float64)
    resumo = {
        "pipeline_version": VERSAO_PIPELINE,
        "source": "Ha_diferencial_real_P_0007_0506_0603_geocodificado_por_grille_0007",
        "formula": "kz_0506_0603 = 2*pi/Ha_0007_0603 - 2*pi/Ha_0007_0506",
        "referencia_geometrica_cancelada": "tropi0007",
        "par_tcct": [cfg.aquisicao_mestre, cfg.aquisicao_escrava],
        "ha_0506": str(ha_0506_path),
        "ha_0603": str(ha_0603_path),
        "grille_referencia_ha": str(grille_0007),
        "layout_ha": layout,
        "mapa_inverso_ha": mapa["report_data"],
        "geometria_comum_slc": str(comum["report_path"]),
        "saida_kz_grade_chm": str(saida),
        "limites_kz_abs_rad_m": [
            cfg.kz_p_abs_minimo_produto,
            cfg.kz_p_abs_maximo_produto,
        ],
        "contadores": dict(cont),
        "fracao_kz_valido": float(cont["kz_valido"] / max(cont["pixels_total"], 1)),
        "estatisticas_kz": _estatisticas_raster_valido(sample),
        "signo_a_confirmar_exclusivamente_no_treino": True,
    }
    resumo_path.write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n✅ kz real da Banda P preparado por .grille:")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return saida, resumo

def rvog_complexo_numpy_kz_variavel(
    h: np.ndarray,
    kz: np.ndarray,
    a: float,
) -> np.ndarray:
    h = np.clip(np.asarray(h, dtype=np.float64), 0.1, 60.0)
    kz = np.asarray(kz, dtype=np.float64)
    a = float(max(a, 1e-6))
    q = np.exp(-a * h)
    phase = kz * h
    den1 = np.maximum(a * a + kz * kz, 1e-12)
    alpha = (a * a - 1j * a * kz) / den1
    beta = (np.exp(1j * phase) - q) / np.maximum(1.0 - q, 1e-8)
    return alpha * beta


def calibrar_atenuacao_e_signo_kz_p(
    dataset,
    train_idx: Sequence[int],
    overlay: pd.DataFrame,
    cfg: ConfigDados,
    out: Path,
) -> dict:
    """Calibra somente a atenuação efetiva e o sinal de kz no treino.

    O módulo espacial de kz vem do produto Ha real. A calibração não altera a
    magnitude de kz e não usa a validação.
    """
    indices = np.asarray(train_idx, dtype=int)
    if len(indices) > cfg.max_amostras_calibracao_rvog:
        rng = np.random.default_rng(CFG.treino.seed)
        indices = rng.choice(
            indices,
            cfg.max_amostras_calibracao_rvog,
            replace=False,
        )

    hs, gammas, kzs = [], [], []
    for i in tqdm(indices, desc="Calibrando atenuação/signo RVoG-P", leave=False):
        kz = float(overlay.at[int(i), "kz_p"])
        if not np.isfinite(kz):
            continue
        amostra = dataset.carregar_amostra_bruta(int(i))
        try:
            gamma = complex(
                amostra["gamma_volume_derot_p"].reshape(-1)[0].item()
            )
            mag = abs(gamma)
        except Exception:
            continue
        if not (
            np.isfinite(gamma.real)
            and np.isfinite(gamma.imag)
            and cfg.gamma_min_calibracao <= mag <= cfg.gamma_max_calibracao
        ):
            continue
        y = amostra["chm"].numpy().astype(np.float64)
        m = amostra["valid_mask"].numpy().astype(bool)
        if not np.any(m):
            continue
        h = float(np.mean(y[m]))
        if not (cfg.chm_minimo_medio <= h <= cfg.chm_maximo_valido):
            continue
        hs.append(h)
        gammas.append(gamma)
        kzs.append(kz)

    h = np.asarray(hs, dtype=np.float64)
    gamma = np.asarray(gammas, dtype=np.complex128)
    kz = np.asarray(kzs, dtype=np.float64)
    if len(h) < CFG.treino.min_amostras_fisica_treino:
        return {
            "success": False,
            "reason": f"apenas {len(h)} observações com Ha+kz+coerência válidos",
            "n": int(len(h)),
            "banda": "p",
        }

    amin = cfg.atenuacao_min_calibracao
    amax = cfg.atenuacao_max_calibracao
    candidatos = []

    for sinal in (1.0, -1.0):
        kz_s = sinal * kz
        for modo in ("complexa", "magnitude"):
            def objetivo(loga):
                a = float(math.exp(float(np.asarray(loga).reshape(-1)[0])))
                pred = rvog_complexo_numpy_kz_variavel(h, kz_s, a)
                if modo == "magnitude":
                    return _huber_numpy(
                        np.abs(pred) - np.abs(gamma),
                        delta=0.08,
                    )
                resid = np.concatenate([
                    pred.real - gamma.real,
                    pred.imag - gamma.imag,
                ])
                return _huber_numpy(resid, delta=0.08)

            # Vários inícios evitam mínimos locais.
            for a0 in np.geomspace(amin, amax, 8):
                res = minimize(
                    objetivo,
                    x0=np.array([math.log(a0)]),
                    method="L-BFGS-B",
                    bounds=[(math.log(amin), math.log(amax))],
                    options={"maxiter": 200},
                )
                a = float(math.exp(res.x[0]))
                pred = rvog_complexo_numpy_kz_variavel(h, kz_s, a)
                candidatos.append({
                    "modo": modo,
                    "sinal_kz": int(sinal),
                    "a": a,
                    "loss": float(res.fun),
                    "rmse_complexo": float(
                        np.sqrt(np.mean(np.abs(pred - gamma) ** 2))
                    ),
                    "rmse_magnitude": float(
                        np.sqrt(
                            np.mean((np.abs(pred) - np.abs(gamma)) ** 2)
                        )
                    ),
                    "n": int(len(h)),
                    "success_opt": bool(res.success),
                })

    df = pd.DataFrame(candidatos).sort_values("loss")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "candidatos_atenuacao_signo_rvog_p.csv", index=False)

    melhor_complexo = (
        df[df["modo"] == "complexa"]
        .sort_values("rmse_complexo")
        .iloc[0]
        .to_dict()
    )
    melhor_magnitude = (
        df[df["modo"] == "magnitude"]
        .sort_values("rmse_magnitude")
        .iloc[0]
        .to_dict()
    )

    if (
        melhor_complexo["rmse_complexo"]
        <= cfg.rmse_complexo_max_calibracao
    ):
        escolhido = melhor_complexo
        modo_loss = "complexa_derotada"
    elif (
        melhor_magnitude["rmse_magnitude"]
        <= cfg.rmse_magnitude_max_calibracao
    ):
        escolhido = melhor_magnitude
        modo_loss = "magnitude"
    else:
        escolhido = melhor_magnitude
        resultado = {
            "success": False,
            "reason": "coerência observada não ajustou ao RVoG dentro dos limites",
            "diagnostico": escolhido,
            "melhor_complexo": melhor_complexo,
            "melhor_magnitude": melhor_magnitude,
            "n": int(len(h)),
            "banda": "p",
        }
        (out / "calibracao_atenuacao_signo_rvog_p.json").write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return resultado

    resultado = dict(escolhido)
    resultado.update({
        "success": True,
        "banda": "p",
        "modo_loss": modo_loss,
        "source": "kz_Ha_real_P_atenuacao_calibrada_somente_treino",
        "kz_magnitude_calibrada": False,
        "validacao_usada_na_calibracao": False,
    })
    (out / "calibracao_atenuacao_signo_rvog_p.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return resultado


def _preparar_geometria_rvog_v26_fallback(dataset, train_idx, val_idx, out: Path) -> dict:
    """Prepara um overlay de geometria sem alterar os arquivos .pt.

    Prioridades:
    1. Usa geometria física já presente no dataset.
    2. Busca/deriva kz e incidência em metadados da campanha.
    3. Calibra kz e atenuação efetivos SOMENTE no conjunto de treino.
    """
    modo = CFG.dados.modo_geometria_rvog.lower()
    pasta = out / "geometria_rvog"
    pasta.mkdir(parents=True, exist_ok=True)
    dataset.limpar_overlay_geometria()

    # Verifica a geometria original efetivamente salva nos patches.
    n_orig_train = contar_fisica_valida(dataset, train_idx)
    n_orig_val = contar_fisica_valida(dataset, val_idx)
    if n_orig_train >= CFG.treino.min_amostras_fisica_treino and n_orig_val >= CFG.treino.min_amostras_fisica_validacao:
        resumo = {"source": "dataset_original", "treino": n_orig_train, "validacao": n_orig_val}
        (pasta / "resumo_geometria_rvog.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
        return resumo

    if modo == "off":
        return {"source": "off", "treino": 0, "validacao": 0}

    # Tenta constantes explicitamente informadas, rasters e metadados diretos/derivados.
    geom_l = resolver_geometria_metadados(CFG.dados, "SAR_L", pasta)
    geom_p = resolver_geometria_metadados(CFG.dados, "SAR_P", pasta)
    if CFG.dados.kz_l_constante is not None:
        geom_l.update({"kz": float(CFG.dados.kz_l_constante), "source": "constante_manual_auditada"})
    if CFG.dados.kz_p_constante is not None:
        geom_p.update({"kz": float(CFG.dados.kz_p_constante), "source": "constante_manual_auditada"})
    if CFG.dados.incidencia_l_deg is not None:
        geom_l["theta"] = float(CFG.dados.incidencia_l_deg)
    if CFG.dados.incidencia_p_deg is not None:
        geom_p["theta"] = float(CFG.dados.incidencia_p_deg)

    mapas = {}
    origens_mapas = {}
    for banda, b in (("SAR_L", "l"), ("SAR_P", "p")):
        mapas[(b, "kz")], origens_mapas[(b, "kz")] = _mapa_geometria_alinhado(CFG.dados, banda, "kz")
        mapas[(b, "theta")], origens_mapas[(b, "theta")] = _mapa_geometria_alinhado(CFG.dados, banda, "incidencia")

    metadados_ok = (
        (np.isfinite(geom_l["kz"]) or mapas[("l", "kz")] is not None)
        and (np.isfinite(geom_p["kz"]) or mapas[("p", "kz")] is not None)
    )
    # Se theta faltar, ainda será possível calibrar a atenuação efetiva a no treino.

    geometria_completa = (
        (np.isfinite(geom_l["kz"]) or mapas[("l", "kz")] is not None)
        and (np.isfinite(geom_p["kz"]) or mapas[("p", "kz")] is not None)
        and (np.isfinite(geom_l["theta"]) or mapas[("l", "theta")] is not None)
        and (np.isfinite(geom_p["theta"]) or mapas[("p", "theta")] is not None)
    )
    observacoes = None
    cal_l = cal_p = None
    precisa_calibrar = (
        modo == "calibracao_treino"
        or (modo == "auto" and CFG.dados.permitir_calibracao_rvog_treino and not geometria_completa)
    )
    if precisa_calibrar:
        observacoes = extrair_observacoes_rvog(dataset, train_idx, CFG.dados.max_amostras_calibracao_rvog)

    if modo == "metadados" and not metadados_ok:
        resumo = {"source": "metadados_ausentes", "treino": 0, "validacao": 0, "geom_l": geom_l, "geom_p": geom_p}
        (pasta / "resumo_geometria_rvog.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        return resumo

    if observacoes is not None:
        cal_l = calibrar_rvog_efetivo(observacoes["l"]["h"], observacoes["l"]["gamma"], CFG.dados, "l", pasta)
        cal_p = calibrar_rvog_efetivo(observacoes["p"]["h"], observacoes["p"]["gamma"], CFG.dados, "p", pasta)

    # Escolhe por banda: kz de metadado tem prioridade; a efetiva é usada quando necessária.
    def parametros_banda(geom, cal):
        kz = geom["kz"] if np.isfinite(geom.get("kz", np.nan)) else (cal.get("kz", np.nan) if cal and cal.get("success") else np.nan)
        theta = geom["theta"] if np.isfinite(geom.get("theta", np.nan)) else np.nan
        a = np.nan
        if not np.isfinite(theta) and cal and cal.get("success"):
            a = cal["a"]
        # Mesmo com theta real, uma calibração de a não é necessária: sigma continuará aprendível.
        source = geom.get("source", "ausente") if np.isfinite(geom.get("kz", np.nan)) else (cal.get("source", "ausente") if cal else "ausente")
        return float(kz), float(theta), float(a), source

    kz_l, th_l, a_l, src_l = parametros_banda(geom_l, cal_l)
    kz_p, th_p, a_p, src_p = parametros_banda(geom_p, cal_p)
    overlay = _overlay_constante(dataset, kz_l, kz_p, th_l, th_p, a_l, a_p, source=f"L:{src_l}|P:{src_p}")

    # Validade é determinada por banda e por coerência observada; um único ramo físico válido já permite a Fase B.
    for idx in tqdm(range(len(dataset)), desc="Montando overlay físico RVoG", leave=False):
        amostra = dataset.carregar_amostra_bruta(idx)
        row_index = dataset.index_original.iloc[idx]
        for b in ("l", "p"):
            kz_mapa = _media_mapa_por_linha_indice(mapas[(b, "kz")], row_index)
            th_mapa = _media_mapa_por_linha_indice(mapas[(b, "theta")], row_index)
            if np.isfinite(kz_mapa):
                overlay.at[idx, f"kz_{b}"] = kz_mapa
            if np.isfinite(th_mapa):
                overlay.at[idx, f"theta_{b}"] = th_mapa
            kz = overlay.at[idx, f"kz_{b}"]
            theta = overlay.at[idx, f"theta_{b}"]
            aeff = overlay.at[idx, f"rvog_a_{b}"]
            ok = _validar_gamma_amostra(amostra, b) and np.isfinite(kz) and (np.isfinite(theta) or np.isfinite(aeff))
            overlay.at[idx, f"physics_valid_{b}"] = bool(ok)
        overlay.at[idx, "physics_valid"] = bool(overlay.at[idx, "physics_valid_l"] or overlay.at[idx, "physics_valid_p"])

    dataset.aplicar_overlay_geometria(overlay)
    overlay.to_csv(pasta / "overlay_geometria_rvog.csv", index=False)
    ntrain = contar_fisica_valida(dataset, train_idx)
    nval = contar_fisica_valida(dataset, val_idx)
    modos = [c.get("modo_loss") for c in (cal_l, cal_p) if c and c.get("success")]
    if modos and all(m == "complexa_derotada" for m in modos):
        CFG.treino.modo_fisica = "complexa_derotada"
    elif modos:
        CFG.treino.modo_fisica = "magnitude"
    resumo = {
        "source": "overlay_preparado", "treino": ntrain, "validacao": nval,
        "geom_l": geom_l, "geom_p": geom_p, "cal_l": cal_l, "cal_p": cal_p,
        "origens_rasters": {str(k): str(v) if v is not None else None for k, v in origens_mapas.items()},
        "modo_loss_selecionado": CFG.treino.modo_fisica,
    }
    (pasta / "resumo_geometria_rvog.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGeometria RVoG preparada:", flush=True)
    print(json.dumps(resumo, indent=2, ensure_ascii=False), flush=True)
    return resumo



def preparar_geometria_rvog(dataset, train_idx, val_idx, out: Path) -> dict:
    """Prepara a geometria RVoG; na v2.7 o padrão é kz real P derivado de Ha."""
    if CFG.dados.modo_geometria_rvog.lower() != "ha_p_real":
        return _preparar_geometria_rvog_v26_fallback(
            dataset,
            train_idx,
            val_idx,
            out,
        )

    pasta = Path(out) / "geometria_rvog"
    pasta.mkdir(parents=True, exist_ok=True)
    dataset.limpar_overlay_geometria()

    kz_path, resumo_kz = gerar_kz_p_0506_0603_na_grade_chm(
        CFG.dados,
        out,
        sobrescrever=False,
    )
    mapa_kz_p = np.load(kz_path, mmap_mode="r")

    overlay = _overlay_constante(
        dataset,
        kz_l=np.nan,
        kz_p=np.nan,
        theta_l=np.nan,
        theta_p=np.nan,
        a_l=np.nan,
        a_p=np.nan,
        source="L:sem_geometria_0506_0603|P:Ha_diferencial_real",
    )

    for idx_amostra in tqdm(
        range(len(dataset)),
        desc="Extraindo kz P real por patch",
        leave=False,
    ):
        row = dataset.index_original.iloc[idx_amostra]
        kz = _media_mapa_por_linha_indice(mapa_kz_p, row)
        overlay.at[idx_amostra, "kz_p"] = kz

    cal = calibrar_atenuacao_e_signo_kz_p(
        dataset,
        train_idx,
        overlay,
        CFG.dados,
        pasta,
    )
    if not cal.get("success", False):
        resumo = {
            "source": "Ha_P_real_mas_calibracao_atenuacao_rejeitada",
            "treino": 0,
            "validacao": 0,
            "resumo_kz": resumo_kz,
            "calibracao_p": cal,
        }
        (pasta / "overlay_geometria_rvog.csv").write_text(
            overlay.to_csv(index=False),
            encoding="utf-8",
        )
        (pasta / "resumo_geometria_rvog.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(json.dumps(resumo, indent=2, ensure_ascii=False), flush=True)
        return resumo

    sinal = float(cal["sinal_kz"])
    a_p = float(cal["a"])
    overlay["kz_p"] = overlay["kz_p"].astype(float) * sinal
    overlay["rvog_a_p"] = a_p
    overlay["physics_source"] = (
        "P:kz_Ha_real_0506_0603|"
        f"signo_treino={int(sinal)}|a_treino={a_p:.8g}|"
        "L:sem_geometria"
    )

    for idx_amostra in tqdm(
        range(len(dataset)),
        desc="Validando coerência física P",
        leave=False,
    ):
        amostra = dataset.carregar_amostra_bruta(idx_amostra)
        kz = float(overlay.at[idx_amostra, "kz_p"])
        ok_p = (
            _validar_gamma_amostra(amostra, "p")
            and np.isfinite(kz)
            and CFG.dados.kz_p_abs_minimo_produto
                <= abs(kz)
                <= CFG.dados.kz_p_abs_maximo_produto
            and np.isfinite(a_p)
        )
        overlay.at[idx_amostra, "physics_valid_l"] = False
        overlay.at[idx_amostra, "physics_valid_p"] = bool(ok_p)
        overlay.at[idx_amostra, "physics_valid"] = bool(ok_p)

    dataset.aplicar_overlay_geometria(overlay)
    overlay.to_csv(pasta / "overlay_geometria_rvog.csv", index=False)

    ntrain = contar_fisica_valida(dataset, train_idx)
    nval = contar_fisica_valida(dataset, val_idx)
    CFG.treino.modo_fisica = cal["modo_loss"]

    resumo = {
        "source": "Ha_diferencial_real_P_0506_0603",
        "treino": ntrain,
        "validacao": nval,
        "banda_fisica": "P",
        "banda_L_na_loss_fisica": False,
        "resumo_kz": resumo_kz,
        "calibracao_p": cal,
        "modo_loss_selecionado": CFG.treino.modo_fisica,
        "observacao": (
            "kz espacial derivado de produtos Ha reais; apenas o sinal e a "
            "atenuação efetiva foram escolhidos/calibrados exclusivamente no treino."
        ),
    }
    (pasta / "resumo_geometria_rvog.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nGeometria RVoG v2.7 preparada:", flush=True)
    print(json.dumps(resumo, indent=2, ensure_ascii=False), flush=True)
    return resumo



# ================================================================
# 4. DATASET, SPLITS ESPACIAIS E ESTATÍSTICAS APENAS DO TREINO
# ================================================================
class TropiSARDatasetV2(Dataset):
    def __init__(self, raiz: Path | str):
        self.raiz = Path(raiz)
        index_path = self.raiz/"index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Execute a geração v2; index ausente: {index_path}")
        self.index_original = pd.read_csv(index_path)
        if self.index_original.empty:
            raise RuntimeError("index.csv vazio")
        self.index = self.index_original.copy()
        self._overlay_geometria = None

    def __len__(self):
        return len(self.index)

    def carregar_amostra_bruta(self, idx: int):
        p = self.raiz/self.index_original.iloc[int(idx)]["arquivo"]
        amostra = torch.load(p,weights_only=True,map_location="cpu")
        versao = int(amostra["tcct_dataset_version"].item())
        if versao != VERSAO_DATASET:
            raise RuntimeError(f"Dataset incompatível: {p} é versão {versao}, esperado {VERSAO_DATASET}")
        return amostra

    def limpar_overlay_geometria(self):
        self._overlay_geometria = None
        self.index = self.index_original.copy()

    def aplicar_overlay_geometria(self, overlay: pd.DataFrame):
        ov = overlay.reset_index(drop=True).copy()
        if len(ov) != len(self.index_original):
            raise ValueError("Overlay RVoG não coincide com o número de amostras do dataset.")
        self._overlay_geometria = ov
        self.index = self.index_original.copy()
        for coluna in _GEOMETRIA_COLUNAS:
            if coluna in ov.columns:
                self.index[coluna] = ov[coluna].to_numpy()

    def __getitem__(self, idx: int):
        amostra = self.carregar_amostra_bruta(idx)
        if self._overlay_geometria is not None:
            row = self._overlay_geometria.iloc[int(idx)]
            for b in ("l", "p"):
                amostra[f"kz_{b}"] = torch.tensor([float(row[f"kz_{b}"])], dtype=torch.float32)
                amostra[f"theta_{b}"] = torch.tensor([float(row[f"theta_{b}"])], dtype=torch.float32)
                amostra[f"rvog_a_{b}"] = torch.tensor([float(row[f"rvog_a_{b}"])], dtype=torch.float32)
                amostra[f"physics_valid_{b}"] = torch.tensor([bool(row[f"physics_valid_{b}"])], dtype=torch.bool)
            amostra["physics_valid"] = torch.tensor([bool(row["physics_valid"])], dtype=torch.bool)
        else:
            # Compatibilidade com o dataset v2.4, que não possuía a atenuação efetiva.
            for b in ("l", "p"):
                amostra.setdefault(f"rvog_a_{b}", torch.tensor([float("nan")], dtype=torch.float32))
                base_valid = bool(amostra.get("physics_valid", torch.tensor([False])).reshape(-1)[0].item())
                amostra.setdefault(f"physics_valid_{b}", torch.tensor([base_valid], dtype=torch.bool))
        return amostra


@dataclass
class StatsTreino:
    optical_mean: list[float]
    optical_std: list[float]
    target_mean: float
    target_std: float
    patch_mean_std: float

    def salvar(self,path:Path):
        path.write_text(json.dumps(asdict(self),indent=2),encoding="utf-8")


def aplicar_buffer_espacial(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, buffer_blocos: int) -> np.ndarray:
    if buffer_blocos <= 0:
        return train_idx
    val_blocks = set(zip(df.iloc[val_idx].block_row.astype(int),df.iloc[val_idx].block_col.astype(int)))
    manter=[]
    for i in train_idx:
        br,bc=int(df.iloc[i].block_row),int(df.iloc[i].block_col)
        perto=any(abs(br-vr)<=buffer_blocos and abs(bc-vc)<=buffer_blocos for vr,vc in val_blocks)
        if not perto: manter.append(i)
    return np.asarray(manter,dtype=int)


def criar_folds_espaciais(dataset:TropiSARDatasetV2,cfg_dados=CFG.dados,cfg_treino=CFG.treino):
    grupos=dataset.index["group"].astype(str).to_numpy()
    gkf=GroupKFold(n_splits=cfg_treino.k_folds)
    x_dummy=np.zeros(len(dataset))
    folds=[]
    for fold,(tr,va) in enumerate(gkf.split(x_dummy,groups=grupos)):
        tr=aplicar_buffer_espacial(dataset.index,tr,va,cfg_dados.buffer_blocos)
        folds.append((tr,va))
        print(f"Fold {fold}: treino={len(tr)} validação={len(va)} grupos_val={len(set(grupos[va]))}")
    return folds



def _distancias_particoes(df: pd.DataFrame, idx_a: Sequence[int], idx_b: Sequence[int]) -> dict:
    """Calcula distâncias da partição A ao vizinho mais próximo da partição B.

    As coordenadas ``x`` e ``y`` vêm do ``index.csv``. Para Paracou elas devem
    estar em um CRS projetado; o relatório registra as unidades como unidades do
    CRS e não presume metros silenciosamente.
    """
    ia = np.asarray(idx_a, dtype=int)
    ib = np.asarray(idx_b, dtype=int)
    if len(ia) == 0 or len(ib) == 0:
        return {"n_a": len(ia), "n_b": len(ib), "min": np.nan, "q05": np.nan, "mediana": np.nan}
    a = df.iloc[ia][["x", "y"]].to_numpy(float)
    b = df.iloc[ib][["x", "y"]].to_numpy(float)
    tree = cKDTree(b)
    dist, _ = tree.query(a, k=1, workers=-1)
    return {
        "n_a": int(len(ia)),
        "n_b": int(len(ib)),
        "min": float(np.min(dist)),
        "q01": float(np.percentile(dist, 1)),
        "q05": float(np.percentile(dist, 5)),
        "mediana": float(np.median(dist)),
        "media": float(np.mean(dist)),
        "max": float(np.max(dist)),
        "unidade": "unidade_do_CRS_do_index_csv",
    }


def auditar_fold_espacial(dataset: TropiSARDatasetV2, train_idx, val_idx, fold: int, buffer_blocos: int) -> dict:
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    intersecao = np.intersect1d(train_idx, val_idx)
    grupos_treino = set(dataset.index.iloc[train_idx]["group"].astype(str))
    grupos_val = set(dataset.index.iloc[val_idx]["group"].astype(str))
    blocos_treino = dataset.index.iloc[train_idx][["block_row", "block_col"]].drop_duplicates().to_numpy(int)
    blocos_val = dataset.index.iloc[val_idx][["block_row", "block_col"]].drop_duplicates().to_numpy(int)
    distancia_bloco_min = np.inf
    if len(blocos_treino) and len(blocos_val):
        tree = cKDTree(blocos_val)
        # Distância euclidiana em unidades de bloco; o buffer foi aplicado pela
        # métrica de Chebyshev. Também calculamos a separação Chebyshev exata.
        for br, bc in blocos_treino:
            d = np.max(np.abs(blocos_val - np.array([br, bc])), axis=1)
            distancia_bloco_min = min(distancia_bloco_min, float(np.min(d)))
    rel = {
        "fold": int(fold),
        "buffer_blocos": int(buffer_blocos),
        "n_treino": int(len(train_idx)),
        "n_validacao": int(len(val_idx)),
        "intersecao_amostras": int(len(intersecao)),
        "intersecao_grupos": int(len(grupos_treino & grupos_val)),
        "grupos_treino": int(len(grupos_treino)),
        "grupos_validacao": int(len(grupos_val)),
        "distancia_chebyshev_min_blocos": float(distancia_bloco_min),
        **{f"distancia_coord_{k}": v for k, v in _distancias_particoes(dataset.index, train_idx, val_idx).items()},
    }
    if rel["intersecao_amostras"] != 0 or rel["intersecao_grupos"] != 0:
        raise AssertionError(f"Vazamento no fold {fold}: {rel}")
    if buffer_blocos > 0 and distancia_bloco_min <= buffer_blocos:
        raise AssertionError(
            f"Buffer espacial não respeitado no fold {fold}: "
            f"distância Chebyshev mínima={distancia_bloco_min}, buffer={buffer_blocos}."
        )
    return rel


def plotar_mapa_folds(dataset: TropiSARDatasetV2, folds, caminho: Path, titulo: str):
    """Gera um mapa diagnóstico dos grupos de validação dos cinco folds."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    for fold, (_, val_idx) in enumerate(folds):
        d = dataset.index.iloc[np.asarray(val_idx, dtype=int)]
        ax.scatter(d["x"], d["y"], s=8, alpha=0.65, label=f"Fold {fold}")
    ax.set_title(titulo)
    ax.set_xlabel("X coordinate of the Landsat-cell center")
    ax.set_ylabel("Y coordinate of the Landsat-cell center")
    ax.grid(True, alpha=0.25)
    ax.legend(markerscale=2, ncol=2)
    fig.tight_layout()
    fig.savefig(caminho, dpi=220)
    plt.close(fig)


def auditar_sensibilidade_buffer(dataset: TropiSARDatasetV2, buffers=None, out: Optional[Path] = None) -> pd.DataFrame:
    """Compara buffer=0 e buffer=1 sem treinar modelos.

    Essa etapa quantifica o custo amostral da faixa de exclusão e evita adotar
    um buffer por conveniência. A decisão final deve considerar separação espacial,
    número de grupos e quantidade restante de treino em cada fold.
    """
    buffers = tuple(CFG.dados.buffers_a_auditar if buffers is None else buffers)
    out = Path(out or (CFG.dados.dir_resultados_v2 / "auditoria_espacial"))
    out.mkdir(parents=True, exist_ok=True)
    linhas = []
    buffer_original = CFG.dados.buffer_blocos
    try:
        for buffer in buffers:
            cfg_tmp = copy.deepcopy(CFG.dados)
            cfg_tmp.buffer_blocos = int(buffer)
            folds = criar_folds_espaciais(dataset, cfg_tmp, CFG.treino)
            plotar_mapa_folds(
                dataset,
                folds,
                out / f"mapa_folds_buffer_{buffer}.png",
                f"Spatial folds — buffer of {buffer} block(s)",
            )
            for fold, (tr, va) in enumerate(folds):
                linhas.append(auditar_fold_espacial(dataset, tr, va, fold, int(buffer)))
    finally:
        CFG.dados.buffer_blocos = buffer_original
    df = pd.DataFrame(linhas)
    df.to_csv(out / "comparacao_buffers_folds.csv", index=False)
    resumo = df.groupby("buffer_blocos").agg(
        treino_medio=("n_treino", "mean"),
        treino_minimo=("n_treino", "min"),
        validacao_media=("n_validacao", "mean"),
        distancia_coord_min=("distancia_coord_min", "min"),
        distancia_blocos_min=("distancia_chebyshev_min_blocos", "min"),
    )
    resumo.to_csv(out / "resumo_comparacao_buffers.csv")
    print(resumo, flush=True)
    return df


def calcular_stats_treino(dataset:TropiSARDatasetV2,indices:Sequence[int]) -> StatsTreino:
    opt_sum=np.zeros(8,np.float64); opt_sq=np.zeros(8,np.float64); nopt=0
    sy=sy2=0.0; ny=0; patch_means=[]
    for i in tqdm(indices,desc="Estatísticas do treino"):
        a=dataset[int(i)]
        o=a["optical"].numpy().astype(np.float64)
        opt_sum+=o; opt_sq+=o*o; nopt+=1
        y=a["chm"].numpy().astype(np.float64)
        m=a["valid_mask"].numpy().astype(bool)
        v=y[m]
        sy+=v.sum(); sy2+=(v*v).sum(); ny+=v.size
        patch_means.append(v.mean())
    om=opt_sum/max(nopt,1); osd=np.sqrt(np.maximum(opt_sq/max(nopt,1)-om*om,1e-8))
    ym=sy/max(ny,1); ysd=np.sqrt(max(sy2/max(ny,1)-ym*ym,1e-8))
    return StatsTreino(om.tolist(),osd.tolist(),float(ym),float(ysd),float(np.std(patch_means)))


def criar_loaders(dataset, train_idx, val_idx, batch_size=None, seed=None):
    bs = batch_size or CFG.treino.batch_size
    gerador = torch.Generator()
    gerador.manual_seed(CFG.treino.seed if seed is None else int(seed))
    train = DataLoader(
        Subset(dataset, list(map(int, train_idx))),
        batch_size=bs,
        shuffle=True,
        num_workers=CFG.treino.num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=len(train_idx) >= bs,
        generator=gerador,
    )
    val = DataLoader(
        Subset(dataset, list(map(int, val_idx))),
        batch_size=bs,
        shuffle=False,
        num_workers=CFG.treino.num_workers,
        pin_memory=PIN_MEMORY,
    )
    return train, val



def _sha256_bytes(conteudo: bytes) -> str:
    return hashlib.sha256(conteudo).hexdigest()


def salvar_manifestos_folds(
    dataset: TropiSARDatasetV2,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    pasta_saida: Path,
) -> pd.DataFrame:
    """Salva os índices exatos de treino/validação e hashes reprodutíveis.

    O manifesto impede que duas execuções com o mesmo número de amostras sejam
    confundidas quando os patches ou a ordem do index.csv tiverem mudado.
    """
    pasta_saida.mkdir(parents=True, exist_ok=True)

    def bool_robusto(serie: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(serie):
            return serie.fillna(False)
        return (
            serie.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "sim", "yes"})
        )

    colunas_base = [
        c for c in (
            "id", "arquivo", "landsat_row", "landsat_col", "x", "y",
            "block_row", "block_col", "group", "chm_row", "chm_col",
            "chm_h", "chm_w", "sar_geometry_fraction",
            "sar_geometry_margin_mean_px", "sar_geometry_margin_min_px",
            "sar_geometry_schema_version", "physics_valid",
        ) if c in dataset.index.columns
    ]

    index_path = dataset.raiz / "index.csv"
    hash_index = hashlib.sha256(index_path.read_bytes()).hexdigest()
    (pasta_saida / "index_dataset.sha256.txt").write_text(
        f"{hash_index}  {index_path.name}\n",
        encoding="utf-8",
    )

    resumo = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        partes = []
        for particao, indices in (("treino", train_idx), ("validacao", val_idx)):
            df = dataset.index.iloc[np.asarray(indices, dtype=int)][colunas_base].copy()
            df.insert(0, "dataset_index", np.asarray(indices, dtype=int))
            df.insert(1, "fold", fold)
            df.insert(2, "particao", particao)
            partes.append(df)

        manifesto = pd.concat(partes, ignore_index=True)
        manifesto = manifesto.sort_values(
            ["particao", "dataset_index"],
            kind="mergesort",
        ).reset_index(drop=True)

        csv_bytes = manifesto.to_csv(index=False, lineterminator="\n").encode("utf-8")
        caminho_csv = pasta_saida / f"manifesto_fold_{fold}.csv"
        caminho_csv.write_bytes(csv_bytes)
        hash_fold = _sha256_bytes(csv_bytes)
        (pasta_saida / f"manifesto_fold_{fold}.sha256.txt").write_text(
            f"{hash_fold}  {caminho_csv.name}\n",
            encoding="utf-8",
        )

        auditoria_fold = auditar_fold_espacial(
            dataset, train_idx, val_idx, fold, CFG.dados.buffer_blocos
        )
        resumo.append({
            "fold": fold,
            **auditoria_fold,
            "n_treino": int(len(train_idx)),
            "n_validacao": int(len(val_idx)),
            "grupos_treino": int(dataset.index.iloc[train_idx]["group"].nunique()),
            "grupos_validacao": int(dataset.index.iloc[val_idx]["group"].nunique()),
            "physics_valid_treino": int(
                bool_robusto(dataset.index.iloc[train_idx]["physics_valid"]).sum()
            ) if "physics_valid" in dataset.index.columns else 0,
            "physics_valid_validacao": int(
                bool_robusto(dataset.index.iloc[val_idx]["physics_valid"]).sum()
            ) if "physics_valid" in dataset.index.columns else 0,
            "sha256_manifesto": hash_fold,
            "sha256_index_dataset": hash_index,
        })

    resumo_df = pd.DataFrame(resumo)
    resumo_df.to_csv(pasta_saida / "resumo_manifestos_folds.csv", index=False)
    print(f"Manifestos dos folds salvos em: {pasta_saida}", flush=True)
    return resumo_df


def auditar_dataset_v2(dataset:TropiSARDatasetV2,max_amostras:int=1500) -> pd.DataFrame:
    """Audita conteúdo numérico e suporte geográfico do dataset canônico."""
    colunas_suporte = {
        "sar_geometry_fraction",
        "sar_geometry_margin_min_px",
        "sar_geometry_schema_version",
    }
    faltantes = colunas_suporte.difference(dataset.index.columns)
    if faltantes:
        raise RuntimeError(
            f"Dataset sem campos obrigatórios de suporte SAR: {sorted(faltantes)}"
        )
    frac = dataset.index["sar_geometry_fraction"].to_numpy(float)
    mmin = dataset.index["sar_geometry_margin_min_px"].to_numpy(float)
    if np.any(frac + 1e-9 < CFG.dados.backgeo_fracao_geometrica_minima_patch):
        raise RuntimeError("Há patches abaixo da fração geométrica SAR mínima.")
    if np.any(~np.isfinite(mmin)) or np.any(mmin < 0.0):
        raise RuntimeError("Há patches com margem geométrica SAR inválida.")
    print(
        "Geometria SAR do dataset aprovada: "
        f"n={len(dataset)} | fração mínima={frac.min():.6f} | "
        f"margem mínima global={mmin.min():.3f} px"
    )
    n=min(max_amostras,len(dataset)); indices=np.linspace(0,len(dataset)-1,n,dtype=int)
    linhas=[]
    for i in tqdm(indices,desc="Auditoria dataset v2"):
        a=dataset[int(i)]; m=a["valid_mask"].numpy().astype(bool); y=a["chm"].numpy()[m]
        gl=a["sar_l"].numpy(); gp=a["sar_p"].numpy(); opt=a["optical"].numpy()
        linhas.append({
            "id":int(i),"chm_mean":float(np.mean(y)),"chm_std":float(np.std(y)),
            "gamma_l_mean":float(np.nanmean(np.abs(gl))),"gamma_p_mean":float(np.nanmean(np.abs(gp))),
            "gamma_l_p95":float(np.nanpercentile(np.abs(gl),95)),"gamma_p_p95":float(np.nanpercentile(np.abs(gp),95)),
            "opt_min":float(np.nanmin(opt)),"opt_max":float(np.nanmax(opt)),
            "sar_geometry_fraction":float(a["sar_geometry_fraction"].item()),
            "sar_geometry_margin_min_px":float(a["sar_geometry_margin_min_px"].item()),
            "physics_valid":bool(a["physics_valid"].item()),
        })
    df=pd.DataFrame(linhas)
    print(df.describe(include="all"))
    print("\nCorrelação simples com CHM médio:")
    print(df[["chm_mean","gamma_l_mean","gamma_p_mean"]].corr()["chm_mean"])
    out=CFG.dados.dir_resultados_v2/"auditoria_dataset_v2.csv";out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False)
    if (df[["gamma_l_mean","gamma_p_mean"]].mean()<0.05).any():
        warnings.warn("Coerência média muito baixa. Audite corregistro interferométrico mestre/escravo antes de treinar.")
    return df






# ================================================================
# 5. ARQUITETURA TCCT v2.8.3.8 — CVNN + COATENÇÃO + DECODER U-NET-STYLE GLOBAL/LOCAL
# ================================================================
class ComplexConv2d(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.wr=nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=False)
        self.wi=nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding,bias=False)
        self.br=nn.Parameter(torch.zeros(1,out_ch,1,1))
        self.bi=nn.Parameter(torch.zeros(1,out_ch,1,1))
    def forward(self,xr,xi):
        yr=self.wr(xr)-self.wi(xi)+self.br
        yi=self.wi(xr)+self.wr(xi)+self.bi
        return yr,yi


class ComplexRMSNorm2d(nn.Module):
    """Aplica o mesmo ganho a real/imag; não rotaciona a fase."""
    def __init__(self,ch,eps=1e-6):
        super().__init__(); self.eps=eps
        self.gain=nn.Parameter(torch.ones(1,ch,1,1))
    def forward(self,r,i):
        rms=torch.sqrt((r.square()+i.square()).mean(dim=(-2,-1),keepdim=True)+self.eps)
        return self.gain*r/rms,self.gain*i/rms


class ModReLU(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.bias=nn.Parameter(torch.full((1,ch,1,1),-0.05))
    def forward(self,r,i):
        mag=torch.sqrt(r.square()+i.square()+1e-8)
        amp=F.relu(mag+self.bias)
        s=amp/(mag+1e-8)
        return r*s,i*s


class ComplexBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride=1):
        super().__init__(); self.conv=ComplexConv2d(in_ch,out_ch,3,stride,1)
        self.norm=ComplexRMSNorm2d(out_ch); self.act=ModReLU(out_ch)
    def forward(self,r,i):
        return self.act(*self.norm(*self.conv(r,i)))


class CVNNEncoder(nn.Module):
    def __init__(self,in_ch=4,dim=64):
        super().__init__()
        self.b1=ComplexBlock(in_ch,24,2)   # 30→15
        self.b2=ComplexBlock(24,48,2)      # 15→8
        self.b3=ComplexBlock(48,48,1)
        self.skip=nn.Sequential(nn.Conv2d(48,32,1),nn.GroupNorm(8,32),nn.GELU())
        self.deep=nn.Sequential(nn.Conv2d(96,dim,1),nn.GroupNorm(8,dim),nn.GELU())
    def forward(self,r,i):
        r1,i1=self.b1(r,i); r2,i2=self.b2(r1,i1); r3,i3=self.b3(r2,i2)
        return self.deep(torch.cat([r3,i3],1)),self.skip(torch.cat([r1,i1],1))


class OpticalContextEncoder(nn.Module):
    """Encoder real coerente com o dado disponível: vetor óptico por célula de 30 m."""
    def __init__(self,in_ch=8,dim=64):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(in_ch,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(0.05),nn.Linear(dim,dim))
        self.pos8=nn.Parameter(torch.zeros(1,dim,8,8)); self.pos15=nn.Parameter(torch.zeros(1,32,15,15))
        self.conv=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1,groups=dim),nn.Conv2d(dim,dim,1),nn.GroupNorm(8,dim),nn.GELU())
        self.skip_proj=nn.Sequential(nn.Linear(dim,32),nn.GELU())
        nn.init.trunc_normal_(self.pos8,std=0.02); nn.init.trunc_normal_(self.pos15,std=0.02)
    def forward(self,x):
        z=self.mlp(x)
        deep=self.conv(z[:,:,None,None].expand(-1,-1,8,8)+self.pos8)
        skip=self.skip_proj(z)[:,:,None,None].expand(-1,-1,15,15)+self.pos15
        return deep,skip


class CoAtencaoTrimodal(nn.Module):
    """Coatenção trimodal com 192 tokens e gating modal explícito.

    Cada modalidade produz uma grade latente 8×8, portanto há 64 tokens ópticos,
    64 tokens da Banda L e 64 tokens da Banda P. O total é 192 tokens, e não 129.

    O vetor óptico original continua sendo um único vetor de oito atributos por
    célula Landsat. Os 64 tokens ópticos são contextos latentes aprendidos obtidos
    pela expansão desse vetor com embeddings posicionais; eles não devem ser
    interpretados como 64 observações ópticas independentes.

    A saída ``pesos_gate`` contém três pesos por amostra, normalizados por softmax,
    usados efetivamente na fusão. Esses pesos são denominados *modality-gating
    weights*. A pequena matriz 3×3 de ``atencao_modal`` vem do MultiheadAttention
    entre os resumos das três modalidades. Nenhuma dessas saídas deve ser tratada
    como importância causal.
    """
    TOKENS_POR_MODALIDADE = 64
    TOTAL_TOKENS = 192

    def __init__(self, dim=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, self.TOKENS_POR_MODALIDADE, dim))
        self.mod = nn.Parameter(torch.zeros(3, dim))
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, layers)
        self.modal_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(dim + 1, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.mod, std=0.02)

    def forward(self, fo, fl, fp, reliability):
        feats = [fo, fl, fp]
        seq = []
        for j, f in enumerate(feats):
            t = (
                f.flatten(2).transpose(1, 2)
                + self.pos
                + self.mod[j].view(1, 1, -1)
            )
            seq.append(t)

        all_tokens = self.transformer(torch.cat(seq, dim=1))
        to, tl, tp = torch.split(
            all_tokens, self.TOKENS_POR_MODALIDADE, dim=1
        )

        modal = torch.stack([to.mean(1), tl.mean(1), tp.mean(1)], dim=1)
        modal2, atencao_modal = self.modal_attn(
            modal,
            modal,
            modal,
            need_weights=True,
            average_attn_weights=False,
        )
        modal = modal + modal2
        logits = self.gate(
            torch.cat([modal, reliability.unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        pesos_gate = torch.softmax(logits, dim=1)
        mapas = [
            t.transpose(1, 2).reshape(-1, t.shape[-1], 8, 8)
            for t in (to, tl, tp)
        ]
        return mapas, pesos_gate, atencao_modal


class CabecaGlobalLocal(nn.Module):
    def __init__(self,dim=64,local_scale=0.1):
        super().__init__()
        self.global_head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,32),nn.GELU(),nn.Linear(32,1))
        self.local_head=nn.Conv2d(16,1,1)
        self.log_local_scale=nn.Parameter(torch.tensor(math.log(local_scale)))
        nn.init.normal_(self.global_head[-1].weight,std=0.01); nn.init.zeros_(self.global_head[-1].bias)
        nn.init.normal_(self.local_head.weight,std=0.01); nn.init.zeros_(self.local_head.bias)
    def forward(self,global_feat,local_feat):
        g=self.global_head(global_feat).view(-1,1,1,1)
        local=self.local_head(local_feat)
        local=local-local.mean(dim=(-2,-1),keepdim=True)
        scale=torch.exp(self.log_local_scale).clamp(0.01,1.0)
        return g+scale*local


class TCCTv2(nn.Module):
    def __init__(self,model_cfg=CFG.modelo,stats:Optional[StatsTreino]=None):
        super().__init__(); d=model_cfg.dim
        self.enc_l=CVNNEncoder(4,d); self.enc_p=CVNNEncoder(4,d); self.enc_o=OpticalContextEncoder(8,d)
        self.coatt=CoAtencaoTrimodal(d,model_cfg.num_heads,model_cfg.transformer_layers,model_cfg.dropout)
        self.fuse8=nn.Sequential(nn.Conv2d(d*3,96,3,padding=1),nn.GroupNorm(8,96),nn.GELU(),nn.Conv2d(96,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.fuse15=nn.Sequential(nn.Conv2d(32*3,48,1),nn.GroupNorm(8,48),nn.GELU())
        self.dec15=nn.Sequential(nn.Conv2d(64+48,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU())
        self.dec30=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,16,3,padding=1),nn.GroupNorm(4,16),nn.GELU())
        self.head=CabecaGlobalLocal(d,model_cfg.local_scale_inicial)
        mean=stats.target_mean if stats else 0.0; std=stats.target_std if stats else 1.0
        self.register_buffer("target_mean",torch.tensor(float(mean),dtype=torch.float32))
        self.register_buffer("target_std",torch.tensor(float(std),dtype=torch.float32))

    def forward(self,sar_l,sar_p,opt,quality,retornar_xai=False):
        lr,li=sar_l; pr,pi=sar_p
        fl,sl=self.enc_l(lr,li); fp,sp=self.enc_p(pr,pi); fo,so=self.enc_o(opt)
        mapas,pesos,atencao_modal=self.coatt(fo,fl,fp,quality)
        ponderados=[mapas[j]*pesos[:,j,None,None,None] for j in range(3)]
        z8=self.fuse8(torch.cat(ponderados,1))
        skips=[so*pesos[:,0,None,None,None],sl*pesos[:,1,None,None,None],sp*pesos[:,2,None,None,None]]
        z15=F.interpolate(z8,size=(15,15),mode="bilinear",align_corners=False)
        z15=self.dec15(torch.cat([z15,self.fuse15(torch.cat(skips,1))],1))
        z30=self.dec30(F.interpolate(z15,size=(30,30),mode="bilinear",align_corners=False))
        pooled=sum(ponderados).mean(dim=(-2,-1))
        pred_norm=self.head(pooled,z30)
        pred_m=self.target_mean+self.target_std*pred_norm
        xai = None
        if retornar_xai:
            xai = {
                "pesos_modalidade": pesos,  # alias de compatibilidade
                "pesos_gate_modalidade": pesos,
                "atencao_modal_3x3": atencao_modal,
                "ordem": ("optico", "sar_l", "sar_p"),
                "metodo": "learned_modality_gating_after_trimodal_coattention",
                "interpretacao_causal": False,
            }
        return pred_m, pred_norm, xai


class TCCTSemCoAtencao(nn.Module):
    """Ablação de fusão: mesmos encoders e decoder, sem coatenção.

    As três representações latentes são concatenadas diretamente. Não há
    Transformer trimodal, MultiheadAttention modal nem gating aprendido.
    Essa variante isola o ganho da estratégia de coatenção mantendo os
    encoders complexos, o ramo óptico, o decoder global-local e a mesma loss.
    """
    def __init__(self, model_cfg=CFG.modelo, stats: Optional[StatsTreino] = None):
        super().__init__(); d = model_cfg.dim
        self.enc_l = CVNNEncoder(4, d)
        self.enc_p = CVNNEncoder(4, d)
        self.enc_o = OpticalContextEncoder(8, d)
        self.fuse8 = nn.Sequential(
            nn.Conv2d(d * 3, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
        )
        self.fuse15 = nn.Sequential(
            nn.Conv2d(32 * 3, 48, 1), nn.GroupNorm(8, 48), nn.GELU()
        )
        self.dec15 = nn.Sequential(
            nn.Conv2d(64 + 48, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.dec30 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.head = CabecaGlobalLocal(d, model_cfg.local_scale_inicial)
        mean = stats.target_mean if stats else 0.0
        std = stats.target_std if stats else 1.0
        self.register_buffer("target_mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(float(std), dtype=torch.float32))

    def forward(self, sar_l, sar_p, opt, quality, retornar_xai=False):
        lr, li = sar_l; pr, pi = sar_p
        fl, sl = self.enc_l(lr, li)
        fp, sp = self.enc_p(pr, pi)
        fo, so = self.enc_o(opt)
        mapas = [fo, fl, fp]
        z8 = self.fuse8(torch.cat(mapas, dim=1))
        z15 = F.interpolate(z8, size=(15, 15), mode="bilinear", align_corners=False)
        z15 = self.dec15(torch.cat([z15, self.fuse15(torch.cat([so, sl, sp], dim=1))], dim=1))
        z30 = self.dec30(F.interpolate(z15, size=(30, 30), mode="bilinear", align_corners=False))
        pooled = sum(mapas).mean(dim=(-2, -1)) / 3.0
        pred_norm = self.head(pooled, z30)
        pred_m = self.target_mean + self.target_std * pred_norm
        xai = None
        if retornar_xai:
            pesos_fixos = torch.full((opt.shape[0], 3), 1.0 / 3.0, device=opt.device)
            xai = {
                "pesos_modalidade": pesos_fixos,
                "pesos_gate_modalidade": pesos_fixos,
                "ordem": ("optico", "sar_l", "sar_p"),
                "metodo": "fixed_equal_weights_control_without_coattention",
                "interpretacao_causal": False,
            }
        return pred_m, pred_norm, xai


class CNNFusaoAntecipada(nn.Module):
    def __init__(self,stats:StatsTreino):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(24,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU(),nn.Conv2d(48,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU())
        self.dec=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Upsample((30,30),mode="bilinear",align_corners=False),nn.Conv2d(32,16,3,padding=1),nn.GELU())
        self.head=CabecaGlobalLocal(64,0.1)
        self.register_buffer("target_mean",torch.tensor(stats.target_mean)); self.register_buffer("target_std",torch.tensor(stats.target_std))
    def forward(self,sar_l,sar_p,opt,quality,retornar_xai=False):
        lr,li=sar_l; pr,pi=sar_p; omap=opt[:,:,None,None].expand(-1,-1,30,30)
        x=torch.cat([omap,lr,li,pr,pi],1); z=self.net(x); loc=self.dec(z); pn=self.head(z.mean((-2,-1)),loc)
        return self.target_mean+self.target_std*pn,pn,None


class ViTDenso(nn.Module):
    def __init__(self,stats:StatsTreino,dim=64,heads=4,layers=2):
        super().__init__(); self.embed=nn.Conv2d(24,dim,3,stride=2,padding=1)
        self.pos=nn.Parameter(torch.zeros(1,225,dim)); enc=nn.TransformerEncoderLayer(dim,heads,dim*4,0.1,batch_first=True,norm_first=True,activation="gelu")
        self.tr=nn.TransformerEncoder(enc,layers); self.dec=nn.Sequential(nn.Conv2d(dim,32,3,padding=1),nn.GELU(),nn.Upsample((30,30),mode="bilinear",align_corners=False),nn.Conv2d(32,16,3,padding=1),nn.GELU())
        self.head=CabecaGlobalLocal(dim,0.1); self.register_buffer("target_mean",torch.tensor(stats.target_mean)); self.register_buffer("target_std",torch.tensor(stats.target_std)); nn.init.trunc_normal_(self.pos,std=.02)
    def forward(self,sar_l,sar_p,opt,quality,retornar_xai=False):
        lr,li=sar_l; pr,pi=sar_p; omap=opt[:,:,None,None].expand(-1,-1,30,30)
        z=self.embed(torch.cat([omap,lr,li,pr,pi],1)); t=z.flatten(2).transpose(1,2)+self.pos; t=self.tr(t); z=t.transpose(1,2).reshape(-1,t.shape[-1],15,15)
        pn=self.head(t.mean(1),self.dec(z)); return self.target_mean+self.target_std*pn,pn,None






# ================================================================
# 6. PREPARAÇÃO DE BATCH E LOSS MULTIESCALA + RVoG REAL
# ================================================================
def separar_complexo(z:torch.Tensor,device,ablation="completo"):
    z=z.squeeze(1) if z.ndim==5 and z.shape[1]==1 else z
    if z.ndim!=4 or not torch.is_complex(z): raise ValueError(f"SAR inválido: {z.shape} {z.dtype}")
    r=torch.nan_to_num(z.real.float()).to(device); i=torch.nan_to_num(z.imag.float()).to(device)
    if ablation=="amplitude_only":
        r=torch.sqrt(r.square()+i.square()+1e-8); i=torch.zeros_like(r)
    return r,i


def preparar_batch_v2(batch,stats:StatsTreino,device,ablation="completo"):
    om=torch.tensor(stats.optical_mean,device=device); osd=torch.tensor(stats.optical_std,device=device).clamp_min(1e-6)
    opt=batch["optical"].float().to(device); opt=(opt-om)/osd
    if ablation=="sem_optico": opt=torch.zeros_like(opt)
    sl=separar_complexo(batch["sar_l"],device,ablation); sp=separar_complexo(batch["sar_p"],device,ablation)
    y=batch["chm"].float().to(device); mask=batch["valid_mask"].bool().to(device)
    if y.ndim==3:y=y.unsqueeze(1)
    if mask.ndim==3:mask=mask.unsqueeze(1)
    oq=batch["optical_quality"].float().to(device).flatten(); ql=batch["sar_quality_l"].float().to(device).flatten(); qp=batch["sar_quality_p"].float().to(device).flatten()
    if ablation=="sem_optico": oq=torch.zeros_like(oq)
    quality=torch.stack([oq,ql,qp],1).clamp(0,1)
    extras={}
    for band in ("l","p"):
        for key in (f"gamma_volume_{band}",f"gamma_volume_derot_{band}"):
            z=batch[key].squeeze(-1); extras[key+"_real"]=z.real.float().to(device); extras[key+"_imag"]=z.imag.float().to(device)
        extras[f"kz_{band}"]=batch[f"kz_{band}"].float().to(device).flatten()
        extras[f"theta_{band}"]=batch[f"theta_{band}"].float().to(device).flatten()
        extras[f"rvog_a_{band}"]=batch[f"rvog_a_{band}"].float().to(device).flatten()
        extras[f"physics_valid_{band}"]=batch[f"physics_valid_{band}"].bool().to(device).flatten()
    extras["physics_valid"]=batch["physics_valid"].bool().to(device).flatten()
    return opt,sl,sp,quality,y,mask,extras


def diferencas_validas(x,mask):
    dx=x[:,:,:,1:]-x[:,:,:,:-1]; mx=mask[:,:,:,1:]&mask[:,:,:,:-1]
    dy=x[:,:,1:,:]-x[:,:,:-1,:]; my=mask[:,:,1:,:]&mask[:,:,:-1,:]
    return dx,mx,dy,my


def pearson_loss(a,b,eps=1e-6):
    a=a-a.mean(); b=b-b.mean(); den=torch.sqrt((a.square().sum()+eps)*(b.square().sum()+eps))
    if a.numel()<3 or den.item()<eps:return a.new_zeros(())
    return 1-(a*b).sum()/den

def ccc_loss(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - coeficiente de correlação de concordância de Lin.

    Diferentemente da correlação de Pearson isolada, o CCC penaliza também
    diferenças de média e de escala entre predição e alvo.
    """
    if a.numel() < 3:
        return a.new_zeros(())
    ma, mb = a.mean(), b.mean()
    va = ((a - ma) ** 2).mean()
    vb = ((b - mb) ** 2).mean()
    cov = ((a - ma) * (b - mb)).mean()
    ccc = (2.0 * cov) / (va + vb + (ma - mb) ** 2 + eps)
    return 1.0 - ccc.clamp(-1.0, 1.0)


class LossTCCTv2(nn.Module):
    def __init__(self,cfg:ConfigTreino,stats:StatsTreino):
        super().__init__(); self.cfg=cfg; self.mean=stats.target_mean; self.std=max(stats.target_std,1e-6)
        # Extinção positiva aprendível por banda; inicialização explícita e auditável.
        self.raw_sigma_l=nn.Parameter(torch.tensor(math.log(math.exp(cfg.sigma_l_inicial)-1)))
        self.raw_sigma_p=nn.Parameter(torch.tensor(math.log(math.exp(cfg.sigma_p_inicial)-1)))
        self.lambda_rvog=0.0

    @staticmethod
    def rvog_volume_a_real_imag(h,kz,a):
        h=h.clamp(0.1,60); a=a.clamp_min(1e-5)
        q=torch.exp(-a*h); phase=kz*h; cr=torch.cos(phase); ci=torch.sin(phase); den1=(a*a+kz*kz).clamp_min(1e-8)
        ar=a*a/den1; ai=-a*kz/den1
        br=(cr-q)/(1-q).clamp_min(1e-6); bi=ci/(1-q).clamp_min(1e-6)
        return ar*br-ai*bi,ar*bi+ai*br

    def parcela_fisica(self,pred_m,mask,extras):
        if self.lambda_rvog<=0 or self.cfg.modo_fisica=="off": return pred_m.new_zeros(())
        soma=pred_m.new_zeros(()); n=0
        # Evita NaN * False: valores mascarados são substituídos antes da soma.
        pred_m_mascarado = torch.where(
            mask,
            pred_m,
            torch.zeros_like(pred_m),
        )
        hpatch = (
            pred_m_mascarado.sum((1, 2, 3))
            / mask.sum((1, 2, 3)).clamp_min(1)
        )
        for band, raw_sigma in (("l", self.raw_sigma_l), ("p", self.raw_sigma_p)):
            valid = extras[f"physics_valid_{band}"]
            kz = extras[f"kz_{band}"]
            theta = extras[f"theta_{band}"]
            a_overlay = extras[f"rvog_a_{band}"]
            tem_a = torch.isfinite(a_overlay)
            tem_theta = torch.isfinite(theta)
            v = valid & torch.isfinite(kz) & (tem_a | tem_theta)
            if not v.any():
                continue
            sigma = F.softplus(raw_sigma) + 1e-4
            a_overlay_v = a_overlay[v]
            theta_v = theta[v]
            usar_overlay = torch.isfinite(a_overlay_v)
            # O ramo não selecionado de torch.where também participa do backward.
            # Nos pixels com atenuação calibrada, theta pode ser NaN; tornamos
            # esse ramo numericamente seguro sem alterar o valor selecionado.
            theta_seguro = torch.where(
                usar_overlay,
                torch.zeros_like(theta_v),
                theta_v,
            )
            a_aprendido = (
                2 * sigma
                / torch.cos(
                    torch.deg2rad(theta_seguro).clamp(
                        0,
                        math.radians(80),
                    )
                ).clamp_min(0.1)
            )
            a = torch.where(usar_overlay, a_overlay_v, a_aprendido)
            gr, gi = self.rvog_volume_a_real_imag(hpatch[v], kz[v], a)
            prefix=f"gamma_volume_derot_{band}" if self.cfg.modo_fisica=="complexa_derotada" else f"gamma_volume_{band}"
            or_=extras[prefix+"_real"][v]; oi=extras[prefix+"_imag"][v]
            if self.cfg.modo_fisica=="magnitude":
                soma=soma+F.smooth_l1_loss(torch.sqrt(gr*gr+gi*gi+1e-8),torch.sqrt(or_*or_+oi*oi+1e-8))
            else:
                soma=soma+F.smooth_l1_loss(gr,or_)+F.smooth_l1_loss(gi,oi)
            n+=1
        return soma/max(n,1)

    def forward(self, pred_m, pred_norm, target, mask, extras):
        yn = (target - self.mean) / self.std
        m = mask & torch.isfinite(target) & torch.isfinite(pred_m)
        if not m.any():
            raise FloatingPointError("Batch sem pixels válidos para a loss.")

        denominador = m.sum((1, 2, 3)).clamp_min(1)
        patch_n = torch.where(
            m,
            yn,
            torch.zeros_like(yn),
        ).sum((1, 2, 3)) / denominador
        pp_n = torch.where(
            m,
            pred_norm,
            torch.zeros_like(pred_norm),
        ).sum((1, 2, 3)) / denominador

        pesos_patch = torch.where(
            (patch_n * self.std + self.mean) >= self.cfg.limiar_estrato_alto_m,
            self.cfg.peso_estrato_alto,
            1.0,
        )
        l_pixel = F.smooth_l1_loss(pred_norm[m], yn[m])
        l_patch = (
            F.smooth_l1_loss(pp_n, patch_n, reduction="none") * pesos_patch
        ).mean()

        pdx, pmx, pdy, pmy = diferencas_validas(pred_norm, m)
        tdx, _, tdy, _ = diferencas_validas(yn, m)
        grads = []
        if pmx.any():
            grads.append(F.smooth_l1_loss(pdx[pmx], tdx[pmx]))
        if pmy.any():
            grads.append(F.smooth_l1_loss(pdy[pmy], tdy[pmy]))
        l_grad = sum(grads) / max(len(grads), 1) if grads else pred_m.new_zeros(())

        # Componente de concordância usada na otimização.
        l_ccc = ccc_loss(pp_n, patch_n)

        # Diagnósticos mantidos no CSV, mas não entram isoladamente na loss.
        l_corr = pearson_loss(pp_n, patch_n)
        stdp = pp_n.std(unbiased=False)
        stdt = patch_n.std(unbiased=False).clamp_min(1e-4)
        l_var = torch.abs(torch.log((stdp + 1e-4) / (stdt + 1e-4)))

        l_phys = self.parcela_fisica(pred_m, m, extras)
        total = (
            self.cfg.w_pixel * l_pixel
            + self.cfg.w_patch * l_patch
            + self.cfg.w_gradiente * l_grad
            + self.cfg.w_ccc * l_ccc
            + self.lambda_rvog * l_phys
        )
        componentes = {
            "pixel": l_pixel.detach(),
            "patch": l_patch.detach(),
            "grad": l_grad.detach(),
            "ccc": l_ccc.detach(),
            "corr": l_corr.detach(),
            "var": l_var.detach(),
            "rvog": l_phys.detach(),
        }
        return total, componentes






# ================================================================
# 7. DIAGNÓSTICOS, TREINO EM DUAS FASES, AVALIAÇÃO E GATING MODAL
# ================================================================
def metricas_np(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    validos = np.isfinite(y) & np.isfinite(p)
    y, p = y[validos], p[validos]
    if not len(y):
        return {
            "rmse": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "std_pred": np.nan,
            "std_true": np.nan,
        }
    return {
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "r2": float(r2_score(y, p)),
        "bias": float(np.mean(p - y)),
        "std_pred": float(np.std(p)),
        "std_true": float(np.std(y)),
    }


def baseline_media(loader, stats):
    y_true, y_pred = [], []
    for batch in loader:
        y = batch["chm"].numpy()
        mask = batch["valid_mask"].numpy().astype(bool)
        y_true.extend([yy[mm].mean() for yy, mm in zip(y, mask)])
        y_pred.extend([stats.target_mean] * len(y))
    return metricas_np(y_true, y_pred)


def criar_modelo(nome, stats):
    if nome == "tcct":
        return TCCTv2(CFG.modelo, stats)
    if nome == "tcct_sem_coatencao":
        return TCCTSemCoAtencao(CFG.modelo, stats)
    if nome == "cnn":
        return CNNFusaoAntecipada(stats)
    if nome == "vit":
        return ViTDenso(
            stats,
            CFG.modelo.dim,
            CFG.modelo.num_heads,
            CFG.modelo.transformer_layers,
        )
    raise ValueError(f"Modelo desconhecido: {nome}")


def lambda_rvog_fase_b(epoca_local: int, cfg: ConfigTreino) -> float:
    """Rampa física exclusiva da Fase B.

    A primeira época física começa em `lambda_rvog_inicial`; a rampa alcança
    `lambda_rvog_final` após `fase_b_epocas_rampa` épocas.
    """
    if cfg.modo_fisica == "off":
        return 0.0
    if cfg.fase_b_epocas_rampa <= 1:
        return float(cfg.lambda_rvog_final)
    frac = min(1.0, epoca_local / (cfg.fase_b_epocas_rampa - 1))
    return float(
        cfg.lambda_rvog_inicial
        + frac * (cfg.lambda_rvog_final - cfg.lambda_rvog_inicial)
    )


def _serie_bool_robusta(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(serie):
        return serie.fillna(False)
    return (
        serie.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "sim", "yes"})
    )


def contar_fisica_valida(dataset: TropiSARDatasetV2, indices: Sequence[int]) -> int:
    if "physics_valid" not in dataset.index.columns:
        return 0
    valores = _serie_bool_robusta(dataset.index.iloc[np.asarray(indices, dtype=int)]["physics_valid"])
    return int(valores.sum())


def _avaliar_modelo_completo(
    model,
    loader,
    stats,
    device,
    ablation="completo",
    xai=False,
    criterio=None,
):
    model.eval()
    y_pred_patch, y_true_patch = [], []
    y_pred_pixel, y_true_pixel = [], []
    xai_rows = []
    perdas = []
    componentes = {k: [] for k in ("pixel", "patch", "grad", "ccc", "corr", "var", "rvog")}

    with torch.no_grad():
        for batch in loader:
            opt, sar_l, sar_p, quality, y, mask, extras = preparar_batch_v2(
                batch, stats, device, ablation
            )
            pred_m, pred_norm, xai_saida = model(
                sar_l, sar_p, opt, quality, retornar_xai=xai
            )

            if criterio is not None:
                loss, comps = criterio(pred_m, pred_norm, y, mask, extras)
                perdas.append(float(loss.item()))
                for chave in componentes:
                    componentes[chave].append(float(comps[chave].item()))

            den = mask.sum((1, 2, 3)).clamp_min(1)
            pred_patch = (pred_m * mask).sum((1, 2, 3)) / den
            true_patch = (y * mask).sum((1, 2, 3)) / den

            y_pred_patch += pred_patch.cpu().tolist()
            y_true_patch += true_patch.cpu().tolist()
            y_pred_pixel += pred_m[mask].cpu().tolist()
            y_true_pixel += y[mask].cpu().tolist()

            if xai and xai_saida:
                pesos = xai_saida["pesos_modalidade"].cpu().numpy()
                for altura, peso in zip(true_patch.cpu().numpy(), pesos):
                    xai_rows.append({
                        "CHM_m": float(altura),
                        "Optico": float(peso[0]),
                        "SAR_L": float(peso[1]),
                        "SAR_P": float(peso[2]),
                        "metodo": xai_saida.get(
                            "metodo",
                            "learned_modality_gating_after_trimodal_coattention",
                        ),
                        "interpretacao_causal": False,
                    })

    metricas_patch = metricas_np(y_true_patch, y_pred_patch)
    metricas_pixel = metricas_np(y_true_pixel, y_pred_pixel)
    perda_media = float(np.mean(perdas)) if perdas else np.nan
    componentes_medios = {
        chave: float(np.mean(valores)) if valores else np.nan
        for chave, valores in componentes.items()
    }
    return (
        metricas_patch,
        metricas_pixel,
        pd.DataFrame(xai_rows),
        np.asarray(y_true_patch),
        np.asarray(y_pred_patch),
        perda_media,
        componentes_medios,
    )


def avaliar_modelo(model, loader, stats, device, ablation="completo", xai=False):
    resultados = _avaliar_modelo_completo(
        model,
        loader,
        stats,
        device,
        ablation=ablation,
        xai=xai,
        criterio=None,
    )
    return resultados[:5]


def executar_sobreajuste_minimo(
    dataset,
    indices,
    stats,
    device=DEVICE,
    modelo_nome="tcct",
    ablation="completo",
    seed=None,
):
    n = min(CFG.treino.amostras_sobreajuste, len(indices))
    sub = np.asarray(indices[:n], dtype=int)
    loader, _ = criar_loaders(
        dataset,
        sub,
        sub,
        batch_size=min(16, n),
        seed=CFG.treino.seed if seed is None else seed,
    )

    model = criar_modelo(modelo_nome, stats).to(device)
    criterio = LossTCCTv2(CFG.treino, stats).to(device)
    criterio.lambda_rvog = 0.0
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)

    losses = []
    model.train()
    iterator = iter(loader)
    for _ in tqdm(
        range(CFG.treino.passos_sobreajuste),
        desc="Teste de sobreajuste",
    ):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        opt, sar_l, sar_p, quality, y, mask, extras = preparar_batch_v2(
            batch, stats, device, ablation
        )
        optimizer.zero_grad(set_to_none=True)
        pred_m, pred_norm, _ = model(sar_l, sar_p, opt, quality)
        loss, _ = criterio(pred_m, pred_norm, y, mask, extras)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.treino.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))

    janela = max(5, len(losses) // 10)
    inicial = float(np.mean(losses[:janela]))
    final = float(np.mean(losses[-janela:]))
    reducao = (inicial - final) / max(abs(inicial), 1e-8)
    print(
        f"Sobreajuste: loss inicial={inicial:.4f} final={final:.4f} "
        f"redução={100 * reducao:.1f}%",
        flush=True,
    )

    del model, criterio, optimizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    if reducao < CFG.treino.reducao_minima_sobreajuste:
        raise RuntimeError(
            "O teste mínimo não reduziu a loss o suficiente. "
            "Audite alinhamento, normalização e gradientes antes do treino completo."
        )
    return reducao


def plot_historico(df, path):
    """Regenerates the training-history figure with English-only labels."""
    if df.empty:
        return

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    epocas = df["epoca"]

    ax[0, 0].plot(epocas, df["loss_treino"], label="Training")
    ax[0, 0].plot(epocas, df["loss_val"], label="Validation")
    ax[0, 0].legend()
    ax[0, 0].set_title("Composite loss")

    ax[0, 1].plot(epocas, df["rmse_patch"], label="Patch")
    ax[0, 1].plot(epocas, df["rmse_pixel"], label="Pixel")
    ax[0, 1].legend()
    ax[0, 1].set_title("RMSE (m)")

    ax[1, 0].plot(epocas, df["r2_patch"])
    ax[1, 0].axhline(0, linestyle="--")
    ax[1, 0].set_title("Patch $R^2$")

    ax[1, 1].plot(epocas, df["std_pred_patch"], label="Predicted")
    ax[1, 1].plot(epocas, df["std_true_patch"], label="Observed")
    ax[1, 1].legend()
    ax[1, 1].set_title("Patch standard deviation")

    for eixo in ax.ravel():
        eixo.grid(True, alpha=0.3)
        eixo.set_xlabel("Epoch")

    nomes_fases = {
        "fase_A_supervisionada": "Phase A — supervised training",
        "fase_B_rvog": "Phase B — RVoG refinement",
        "fine_tuning_local": "Local fine-tuning",
        "baseline_supervisionado": "Supervised baseline training",
    }
    fase_interna = (
        str(df["fase"].iloc[0])
        if "fase" in df.columns and len(df)
        else "training"
    )
    fig.suptitle(nomes_fases.get(fase_interna, fase_interna.replace("_", " ").title()))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _state_dict_cpu(modulo: nn.Module) -> dict:
    return {
        chave: valor.detach().cpu().clone()
        for chave, valor in modulo.state_dict().items()
    }


def _salvar_checkpoint(
    caminho: Path,
    model,
    criterio,
    stats,
    fase,
    epoca,
    metricas_patch,
    metricas_pixel,
    modelo_nome,
    ablation,
):
    payload = {
        "model_state_dict": _state_dict_cpu(model),
        "criterion_state_dict": _state_dict_cpu(criterio),
        "stats": asdict(stats),
        "config": asdict(CFG),
        "fase": fase,
        "epoca": int(epoca),
        "metricas_patch": metricas_patch,
        "metricas_pixel": metricas_pixel,
        "rmse_patch": float(metricas_patch["rmse"]),
        "modelo": modelo_nome,
        "ablation": ablation,
        "versao_pipeline": VERSAO_PIPELINE,
        "versao_dataset": VERSAO_DATASET,
    }
    torch.save(payload, caminho)
    return payload



def _torch_load_local(caminho: Path | str, map_location="cpu"):
    """Carrega checkpoints locais produzidos pelo próprio pipeline."""
    caminho = Path(caminho)
    try:
        return torch.load(caminho, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(caminho, map_location=map_location)


def _torch_save_atomico(payload: dict, caminho: Path | str) -> None:
    """Evita checkpoint parcialmente escrito em interrupções ou queda de energia."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    torch.save(payload, temporario)
    os.replace(temporario, caminho)


def _capturar_estado_rng() -> dict:
    estado = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        estado["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            estado["torch_mps"] = torch.mps.get_rng_state()
        except Exception:
            pass
    return estado


def _restaurar_estado_rng(estado: Optional[dict]) -> None:
    if not estado:
        return
    if "python" in estado:
        random.setstate(estado["python"])
    if "numpy" in estado:
        np.random.set_state(estado["numpy"])
    if "torch_cpu" in estado:
        torch.set_rng_state(estado["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in estado:
        torch.cuda.set_rng_state_all(estado["torch_cuda"])
    if (
        "torch_mps" in estado
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "set_rng_state")
    ):
        try:
            torch.mps.set_rng_state(estado["torch_mps"])
        except Exception:
            pass


def _mover_estado_optimizer_para_device(optimizer, device) -> None:
    for estado in optimizer.state.values():
        for chave, valor in list(estado.items()):
            if torch.is_tensor(valor):
                estado[chave] = valor.to(device)


def _assinatura_fase(
    nome_fase,
    modelo_nome,
    ablation,
    epocas,
    min_epocas,
    paciencia,
    lr,
    lr_min,
    scheduler_paciencia,
    seed,
) -> dict:
    return {
        "pipeline": VERSAO_PIPELINE,
        "dataset": VERSAO_DATASET,
        "fase": str(nome_fase),
        "modelo": str(modelo_nome),
        "ablation": str(ablation),
        "epocas": int(epocas),
        "min_epocas": int(min_epocas),
        "paciencia": int(paciencia),
        "lr": float(lr),
        "lr_min": float(lr_min),
        "scheduler_paciencia": int(scheduler_paciencia),
        "seed": int(seed),
    }


def _assinaturas_compativeis(a: Optional[dict], b: dict) -> bool:
    if not a:
        return False
    # A versão do pipeline pode mudar por hotfix de retomada sem alterar o
    # experimento. Os demais campos científicos precisam permanecer idênticos.
    chaves = (
        "dataset", "fase", "modelo", "ablation", "epocas", "min_epocas",
        "paciencia", "lr", "lr_min", "scheduler_paciencia", "seed",
    )
    return all(a.get(k) == b.get(k) for k in chaves)


def treinar_fase(
    model,
    dataset,
    train_idx,
    val_idx,
    stats,
    out,
    nome_fase,
    modelo_nome,
    ablation,
    epocas,
    min_epocas,
    paciencia,
    lr,
    lr_min,
    scheduler_paciencia,
    lambda_fn,
    seed,
    device=DEVICE,
):
    """Treina uma fase com retomada por época e restaura o melhor RMSE de patch.

    A cada época concluída, salva pesos atuais, melhor checkpoint, estados do
    otimizador/scheduler, histórico e RNG. Se a execução for interrompida no
    meio de uma época, a retomada reinicia somente essa época.
    """
    pasta_fase = Path(out) / nome_fase
    checkpoint_path = pasta_fase / "melhor_checkpoint.pth"
    checkpoint_retomada = pasta_fase / "checkpoint_retomada.pth"
    marcador_conclusao = pasta_fase / "fase_concluida.json"
    historico_path = pasta_fase / "historico.csv"

    assinatura = _assinatura_fase(
        nome_fase,
        modelo_nome,
        ablation,
        epocas,
        min_epocas,
        paciencia,
        lr,
        lr_min,
        scheduler_paciencia,
        seed,
    )

    # Um estado parcial gerado pela versão antiga não contém optimizer/RNG.
    # Ele é arquivado para não misturar duas trajetórias de treinamento.
    if (
        pasta_fase.exists()
        and not marcador_conclusao.exists()
        and not checkpoint_retomada.exists()
        and (historico_path.exists() or checkpoint_path.exists())
    ):
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        legado = pasta_fase.with_name(
            pasta_fase.name + f"_PARCIAL_PRE_RETOMADA_{timestamp}"
        )
        shutil.move(str(pasta_fase), str(legado))
        print(
            f"⚠️ Estado parcial antigo arquivado em: {legado}\n"
            "   Esta combinação reiniciará somente uma vez; as próximas "
            "interrupções serão retomáveis por época.",
            flush=True,
        )

    pasta_fase.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = criar_loaders(
        dataset,
        train_idx,
        val_idx,
        seed=seed,
    )
    criterio = LossTCCTv2(CFG.treino, stats).to(device)

    parametros = list(model.parameters())
    if nome_fase == "fase_B_rvog" and modelo_nome in {"tcct", "tcct_sem_coatencao"}:
        parametros += list(criterio.parameters())

    optimizer = optim.AdamW(
        parametros,
        lr=lr,
        weight_decay=CFG.treino.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_paciencia,
        min_lr=lr_min,
    )

    # Fase já concluída: carrega e retorna sem treinar novamente.
    if marcador_conclusao.exists() and checkpoint_path.exists() and historico_path.exists():
        marcador = json.loads(marcador_conclusao.read_text(encoding="utf-8"))
        if _assinaturas_compativeis(marcador.get("assinatura"), assinatura):
            checkpoint = _torch_load_local(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            criterio.load_state_dict(checkpoint["criterion_state_dict"])
            historico_df = pd.read_csv(historico_path)
            melhor_epoca = int(checkpoint["epoca"])
            melhor_rmse = float(checkpoint["rmse_patch"])
            print(
                f"⏭️ {nome_fase} já concluída; checkpoint reutilizado "
                f"(época {melhor_epoca}, RMSE={melhor_rmse:.3f}).",
                flush=True,
            )
            return {
                "fase": nome_fase,
                "model_state_dict": checkpoint["model_state_dict"],
                "criterion_state_dict": checkpoint["criterion_state_dict"],
                "melhor_epoca": melhor_epoca,
                "melhor_rmse_patch": melhor_rmse,
                "metricas_patch": checkpoint["metricas_patch"],
                "metricas_pixel": checkpoint["metricas_pixel"],
                "historico": historico_df,
                "checkpoint": checkpoint_path,
                "lambda_final_checkpoint": float(
                    historico_df.loc[
                        historico_df["rmse_patch"].idxmin(),
                        "lambda_rvog",
                    ]
                ),
            }
        print(
            f"⚠️ Marcador de {nome_fase} incompatível com a configuração atual; "
            "a fase será recalculada.",
            flush=True,
        )
        marcador_conclusao.unlink(missing_ok=True)

    melhor_rmse = np.inf
    melhor_epoca = 0
    melhor_model_state = None
    melhor_criterio_state = None
    melhor_metricas_patch = None
    melhor_metricas_pixel = None
    sem_melhoria = 0
    historico = []
    epoca_inicial = 0

    # Retomada exata a partir da última época concluída.
    if checkpoint_retomada.exists():
        estado = _torch_load_local(checkpoint_retomada, map_location="cpu")
        if _assinaturas_compativeis(estado.get("assinatura"), assinatura):
            model.load_state_dict(estado["model_state_dict_atual"])
            criterio.load_state_dict(estado["criterion_state_dict_atual"])
            optimizer.load_state_dict(estado["optimizer_state_dict"])
            _mover_estado_optimizer_para_device(optimizer, device)
            scheduler.load_state_dict(estado["scheduler_state_dict"])
            melhor_rmse = float(estado["melhor_rmse"])
            melhor_epoca = int(estado["melhor_epoca"])
            melhor_model_state = estado["melhor_model_state"]
            melhor_criterio_state = estado["melhor_criterio_state"]
            melhor_metricas_patch = estado["melhor_metricas_patch"]
            melhor_metricas_pixel = estado["melhor_metricas_pixel"]
            sem_melhoria = int(estado["sem_melhoria"])
            historico = list(estado["historico"])
            epoca_inicial = int(estado["proxima_epoca"])
            gerador = getattr(train_loader, "generator", None)
            if gerador is not None and estado.get("dataloader_generator_state") is not None:
                gerador.set_state(estado["dataloader_generator_state"])
            _restaurar_estado_rng(estado.get("rng_state"))
            print(
                f"♻️ RETOMADA — {nome_fase}: {epoca_inicial}/{epocas} épocas "
                f"já concluídas; melhor RMSE={melhor_rmse:.3f}.",
                flush=True,
            )
        else:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            incompat = checkpoint_retomada.with_name(
                checkpoint_retomada.stem
                + f"_INCOMPATIVEL_{timestamp}"
                + checkpoint_retomada.suffix
            )
            checkpoint_retomada.replace(incompat)
            print(
                f"⚠️ Checkpoint de retomada incompatível arquivado em {incompat}.",
                flush=True,
            )

    for epoca_local in range(epoca_inicial, epocas):
        inicio = time.time()
        model.train()
        criterio.lambda_rvog = (
            float(lambda_fn(epoca_local)) if lambda_fn is not None else 0.0
        )
        losses = []
        componentes_treino = {
            k: [] for k in ("pixel", "patch", "grad", "ccc", "corr", "var", "rvog")
        }

        for batch in tqdm(
            train_loader,
            desc=f"{out.name}/{modelo_nome}/{ablation}/{nome_fase} ep {epoca_local + 1}",
            leave=False,
        ):
            opt, sar_l, sar_p, quality, y, mask, extras = preparar_batch_v2(
                batch, stats, device, ablation
            )
            optimizer.zero_grad(set_to_none=True)
            pred_m, pred_norm, _ = model(sar_l, sar_p, opt, quality)
            loss, comps = criterio(pred_m, pred_norm, y, mask, extras)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Loss não finita na {nome_fase}, época {epoca_local + 1}."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parametros,
                CFG.treino.grad_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()

            losses.append(float(loss.item()))
            for chave in componentes_treino:
                componentes_treino[chave].append(float(comps[chave].item()))

        (
            metricas_patch,
            metricas_pixel,
            _,
            _,
            _,
            loss_val,
            componentes_val,
        ) = _avaliar_modelo_completo(
            model,
            val_loader,
            stats,
            device,
            ablation=ablation,
            xai=False,
            criterio=criterio,
        )
        scheduler.step(metricas_patch["rmse"])

        duracao_epoca = time.time() - inicio
        row = {
            "fase": nome_fase,
            "epoca": epoca_local + 1,
            "duracao_s": float(duracao_epoca),
            "loss_treino": float(np.mean(losses)),
            "loss_val": loss_val,
            "rmse_patch": metricas_patch["rmse"],
            "r2_patch": metricas_patch["r2"],
            "bias_patch": metricas_patch["bias"],
            "std_pred_patch": metricas_patch["std_pred"],
            "std_true_patch": metricas_patch["std_true"],
            "rmse_pixel": metricas_pixel["rmse"],
            "r2_pixel": metricas_pixel["r2"],
            "lr": optimizer.param_groups[0]["lr"],
            "lambda_rvog": criterio.lambda_rvog,
        }
        for chave, valores in componentes_treino.items():
            row[f"loss_{chave}_treino"] = float(np.mean(valores))
        for chave, valor in componentes_val.items():
            row[f"loss_{chave}_val"] = valor

        melhorou = metricas_patch["rmse"] < melhor_rmse - 1e-4
        if melhorou:
            melhor_rmse = float(metricas_patch["rmse"])
            melhor_epoca = epoca_local + 1
            sem_melhoria = 0
            melhor_model_state = _state_dict_cpu(model)
            melhor_criterio_state = _state_dict_cpu(criterio)
            melhor_metricas_patch = dict(metricas_patch)
            melhor_metricas_pixel = dict(metricas_pixel)
            _salvar_checkpoint(
                checkpoint_path,
                model,
                criterio,
                stats,
                nome_fase,
                melhor_epoca,
                melhor_metricas_patch,
                melhor_metricas_pixel,
                modelo_nome,
                ablation,
            )
            status = (
                f"⭐ NOVO MELHOR | época={melhor_epoca:03d} "
                f"RMSE={melhor_rmse:.3f}"
            )
        else:
            sem_melhoria += 1
            status = (
                f"⏳ sem melhora={sem_melhoria}/{paciencia} | "
                f"melhor=época {melhor_epoca:03d} RMSE={melhor_rmse:.3f}"
            )

        row["melhor_epoca"] = melhor_epoca
        row["melhor_rmse_patch"] = melhor_rmse
        row["sem_melhoria"] = sem_melhoria
        historico.append(row)

        df_hist = pd.DataFrame(historico)
        df_hist.to_csv(historico_path, index=False)
        plot_historico(df_hist, pasta_fase / "metricas.png")

        gerador = getattr(train_loader, "generator", None)
        payload_retomada = {
            "assinatura": assinatura,
            "epoca_concluida": int(epoca_local + 1),
            "proxima_epoca": int(epoca_local + 1),
            "model_state_dict_atual": _state_dict_cpu(model),
            "criterion_state_dict_atual": _state_dict_cpu(criterio),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "melhor_rmse": float(melhor_rmse),
            "melhor_epoca": int(melhor_epoca),
            "melhor_model_state": melhor_model_state,
            "melhor_criterio_state": melhor_criterio_state,
            "melhor_metricas_patch": melhor_metricas_patch,
            "melhor_metricas_pixel": melhor_metricas_pixel,
            "sem_melhoria": int(sem_melhoria),
            "historico": historico,
            "dataloader_generator_state": (
                gerador.get_state() if gerador is not None else None
            ),
            "rng_state": _capturar_estado_rng(),
            "salvo_em": pd.Timestamp.now().isoformat(),
        }
        _torch_save_atomico(payload_retomada, checkpoint_retomada)

        std_ratio = (
            metricas_patch["std_pred"]
            / max(metricas_patch["std_true"], 1e-6)
        )
        print(
            f"\n[{nome_fase}] Ep {epoca_local + 1:03d} "
            f"{duracao_epoca:.1f}s | "
            f"RMSE patch={metricas_patch['rmse']:.3f} "
            f"R²={metricas_patch['r2']:.3f} "
            f"viés={metricas_patch['bias']:.3f} | "
            f"RMSE pixel={metricas_pixel['rmse']:.3f} | "
            f"std ratio={std_ratio:.2f} | "
            f"lr={row['lr']:.2e} λ={criterio.lambda_rvog:.4f} | {status}",
            flush=True,
        )

        if epoca_local >= 4 and std_ratio < 0.20:
            print(
                "⚠️ Variância prevista muito baixa; possível colapso para a média.",
                flush=True,
            )

        if epoca_local + 1 >= min_epocas and sem_melhoria >= paciencia:
            print(
                f"🛑 Early stopping em {nome_fase}, época {epoca_local + 1}; "
                f"melhor época={melhor_epoca}, RMSE={melhor_rmse:.3f}.",
                flush=True,
            )
            break

    if melhor_model_state is None:
        raise RuntimeError(f"Nenhum checkpoint válido produzido em {nome_fase}.")

    model.load_state_dict(melhor_model_state)
    criterio.load_state_dict(melhor_criterio_state)

    marcador = {
        "assinatura": assinatura,
        "melhor_epoca": int(melhor_epoca),
        "melhor_rmse_patch": float(melhor_rmse),
        "epocas_registradas": int(len(historico)),
        "concluida_em": pd.Timestamp.now().isoformat(),
    }
    marcador_conclusao.write_text(
        json.dumps(marcador, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checkpoint_retomada.unlink(missing_ok=True)

    print(
        f"\n🏆 {nome_fase} ENCERRADA | melhor época={melhor_epoca:03d} | "
        f"melhor RMSE patch={melhor_rmse:.3f} | "
        f"épocas sem melhora ao final={sem_melhoria}",
        flush=True,
    )

    historico_df = pd.DataFrame(historico)
    return {
        "fase": nome_fase,
        "model_state_dict": melhor_model_state,
        "criterion_state_dict": melhor_criterio_state,
        "melhor_epoca": melhor_epoca,
        "melhor_rmse_patch": melhor_rmse,
        "metricas_patch": melhor_metricas_patch,
        "metricas_pixel": melhor_metricas_pixel,
        "historico": historico_df,
        "checkpoint": checkpoint_path,
        "lambda_final_checkpoint": float(
            historico_df.loc[
                historico_df["rmse_patch"].idxmin(),
                "lambda_rvog",
            ]
        ),
    }


def treinar_split(
    dataset,
    train_idx,
    val_idx,
    stats,
    rotulo="holdout",
    modelo_nome="tcct",
    ablation="completo",
    device=DEVICE,
    seed=None,
):
    """Treina TCCT em Fase A + Fase B; baselines permanecem supervisionados."""
    seed = CFG.treino.seed if seed is None else int(seed)
    fixar_sementes(seed)

    out = CFG.dados.dir_resultados_v2 / rotulo / modelo_nome / ablation
    out.mkdir(parents=True, exist_ok=True)
    stats.salvar(out / "stats_treino.json")
    (out / "config_execucao.json").write_text(
        json.dumps(asdict(CFG), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _, val_loader = criar_loaders(dataset, train_idx, val_idx, seed=seed)
    baseline = baseline_media(val_loader, stats)
    print("Baseline média:", baseline, flush=True)

    if CFG.treino.executar_teste_sobreajuste:
        executar_sobreajuste_minimo(
            dataset,
            train_idx,
            stats,
            device=device,
            modelo_nome=modelo_nome,
            ablation=ablation,
            seed=seed,
        )

    model = criar_modelo(modelo_nome, stats).to(device)
    resultados_fases = []

    if modelo_nome in {"tcct", "tcct_sem_coatencao"}:
        print("\n" + "=" * 88, flush=True)
        print("FASE A — PRÉ-TREINO SUPERVISIONADO", flush=True)
        print("=" * 88, flush=True)
        fase_a = treinar_fase(
            model=model,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            stats=stats,
            out=out,
            nome_fase="fase_A_supervisionada",
            modelo_nome=modelo_nome,
            ablation=ablation,
            epocas=CFG.treino.fase_a_epocas,
            min_epocas=CFG.treino.fase_a_min_epocas,
            paciencia=CFG.treino.fase_a_paciencia,
            lr=CFG.treino.fase_a_lr,
            lr_min=CFG.treino.fase_a_lr_min,
            scheduler_paciencia=CFG.treino.fase_a_scheduler_paciencia,
            lambda_fn=lambda _: 0.0,
            seed=seed,
            device=device,
        )
        resultados_fases.append(fase_a)

        # A v2.6 prepara a geometria somente depois da Fase A. A calibração
        # efetiva, quando necessária, usa exclusivamente os rótulos do treino.
        resumo_geometria = preparar_geometria_rvog(dataset, train_idx, val_idx, out)
        n_fisica_treino = contar_fisica_valida(dataset, train_idx)
        n_fisica_val = contar_fisica_valida(dataset, val_idx)
        print(
            f"\nAmostras com física válida: treino={n_fisica_treino} | "
            f"validação={n_fisica_val}",
            flush=True,
        )

        executar_b = (
            CFG.treino.executar_fase_b
            and ablation == "completo"
            and CFG.treino.modo_fisica != "off"
            and n_fisica_treino >= CFG.treino.min_amostras_fisica_treino
            and n_fisica_val >= CFG.treino.min_amostras_fisica_validacao
        )

        if executar_b:
            print("\n" + "=" * 88, flush=True)
            print("FASE B — REFINAMENTO FÍSICO RVoG", flush=True)
            print("=" * 88, flush=True)
            model.load_state_dict(fase_a["model_state_dict"])
            fase_b = treinar_fase(
                model=model,
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                stats=stats,
                out=out,
                nome_fase="fase_B_rvog",
                modelo_nome=modelo_nome,
                ablation=ablation,
                epocas=CFG.treino.fase_b_epocas,
                min_epocas=CFG.treino.fase_b_min_epocas,
                paciencia=CFG.treino.fase_b_paciencia,
                lr=CFG.treino.fase_b_lr,
                lr_min=CFG.treino.fase_b_lr_min,
                scheduler_paciencia=CFG.treino.fase_b_scheduler_paciencia,
                lambda_fn=lambda ep: lambda_rvog_fase_b(ep, CFG.treino),
                seed=seed + 10_000,
                device=device,
            )
            resultados_fases.append(fase_b)
        else:
            motivos = []
            if not CFG.treino.executar_fase_b:
                motivos.append("executar_fase_b=False")
            if ablation != "completo":
                motivos.append(f"ablação={ablation}")
            if CFG.treino.modo_fisica == "off":
                motivos.append("modo_fisica=off")
            if n_fisica_treino < CFG.treino.min_amostras_fisica_treino:
                motivos.append(
                    f"física válida no treino {n_fisica_treino}"
                    f"<{CFG.treino.min_amostras_fisica_treino}"
                )
            if n_fisica_val < CFG.treino.min_amostras_fisica_validacao:
                motivos.append(
                    f"física válida na validação {n_fisica_val}"
                    f"<{CFG.treino.min_amostras_fisica_validacao}"
                )
            print(
                "⚠️ Fase B RVoG não executada: " + "; ".join(motivos),
                flush=True,
            )
    else:
        print("\n" + "=" * 88, flush=True)
        print(f"BASELINE {modelo_nome.upper()} — TREINO SUPERVISIONADO", flush=True)
        print("=" * 88, flush=True)
        fase_baseline = treinar_fase(
            model=model,
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            stats=stats,
            out=out,
            nome_fase="supervisionado",
            modelo_nome=modelo_nome,
            ablation=ablation,
            epocas=CFG.treino.baseline_epocas,
            min_epocas=CFG.treino.baseline_min_epocas,
            paciencia=CFG.treino.baseline_paciencia,
            lr=CFG.treino.baseline_lr,
            lr_min=CFG.treino.baseline_lr_min,
            scheduler_paciencia=CFG.treino.baseline_scheduler_paciencia,
            lambda_fn=lambda _: 0.0,
            seed=seed,
            device=device,
        )
        resultados_fases.append(fase_baseline)

    # Seleção final pelo menor RMSE de patch. Se a fase física não melhorar,
    # o checkpoint supervisionado permanece como modelo final.
    melhor_resultado = min(
        resultados_fases,
        key=lambda resultado: resultado["melhor_rmse_patch"],
    )
    model.load_state_dict(melhor_resultado["model_state_dict"])

    _, val_loader = criar_loaders(dataset, train_idx, val_idx, seed=seed)
    metricas_patch, metricas_pixel, xai_df, y_true, y_pred = avaliar_modelo(
        model,
        val_loader,
        stats,
        device,
        ablation=ablation,
        xai=(modelo_nome == "tcct"),
    )

    resumo_fases = pd.DataFrame([
        {
            "fase": resultado["fase"],
            "melhor_epoca": resultado["melhor_epoca"],
            "rmse_patch": resultado["metricas_patch"]["rmse"],
            "r2_patch": resultado["metricas_patch"]["r2"],
            "bias_patch": resultado["metricas_patch"]["bias"],
            "std_pred_patch": resultado["metricas_patch"]["std_pred"],
            "std_true_patch": resultado["metricas_patch"]["std_true"],
            "rmse_pixel": resultado["metricas_pixel"]["rmse"],
            "r2_pixel": resultado["metricas_pixel"]["r2"],
            "lambda_checkpoint": resultado["lambda_final_checkpoint"],
            "selecionada_final": resultado["fase"] == melhor_resultado["fase"],
        }
        for resultado in resultados_fases
    ])
    resumo_fases.to_csv(out / "comparacao_fases.csv", index=False)

    historico_completo = pd.concat(
        [resultado["historico"] for resultado in resultados_fases],
        ignore_index=True,
    )
    historico_completo["epoca_global"] = np.arange(1, len(historico_completo) + 1)
    historico_completo.to_csv(out / "historico_completo.csv", index=False)

    xai_df.to_csv(out / "xai_final.csv", index=False)
    xai_df.to_csv(out / "pesos_gating_modalidade_final.csv", index=False)
    np.savez(out / "predicoes_patch_final.npz", y_true=y_true, y_pred=y_pred)

    checkpoint_final = {
        "model_state_dict": melhor_resultado["model_state_dict"],
        "criterion_state_dict": melhor_resultado["criterion_state_dict"],
        "stats": asdict(stats),
        "config": asdict(CFG),
        "fase_selecionada": melhor_resultado["fase"],
        "metricas_patch": metricas_patch,
        "metricas_pixel": metricas_pixel,
        "modelo": modelo_nome,
        "ablation": ablation,
        "versao_pipeline": VERSAO_PIPELINE,
        "versao_dataset": VERSAO_DATASET,
    }
    torch.save(checkpoint_final, out / "melhor_modelo_final.pth")
    # Alias mantido para compatibilidade com funções antigas de inferência.
    shutil.copy2(out / "melhor_modelo_final.pth", out / "melhor_modelo.pth")

    n_parametros = int(sum(p.numel() for p in model.parameters()))
    resumo_json = {
        "baseline_media": baseline,
        "numero_parametros_treinaveis": n_parametros,
        "fase_selecionada": melhor_resultado["fase"],
        "metricas_patch_final": metricas_patch,
        "metricas_pixel_final": metricas_pixel,
        "n_fases_executadas": len(resultados_fases),
    }
    (out / "resumo_treinamento.json").write_text(
        json.dumps(resumo_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88, flush=True)
    print(
        f"🏆 MODELO FINAL | fase={melhor_resultado['fase']} | "
        f"RMSE patch={metricas_patch['rmse']:.3f} | "
        f"R²={metricas_patch['r2']:.3f} | "
        f"viés={metricas_patch['bias']:.3f}",
        flush=True,
    )
    print("=" * 88, flush=True)
    return model, historico_completo, metricas_patch, metricas_pixel


def treinar_holdout(modelo_nome="tcct", ablation="completo", fold=0):
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    folds = criar_folds_espaciais(dataset)
    salvar_manifestos_folds(
        dataset,
        folds,
        CFG.dados.dir_resultados_v2 / "manifestos_folds",
    )
    if fold < 0 or fold >= len(folds):
        raise IndexError(f"Fold inválido: {fold}. Intervalo: 0 a {len(folds) - 1}.")
    train_idx, val_idx = folds[fold]
    stats = calcular_stats_treino(dataset, train_idx)
    return treinar_split(
        dataset,
        train_idx,
        val_idx,
        stats,
        rotulo=f"fold_{fold}",
        modelo_nome=modelo_nome,
        ablation=ablation,
        seed=CFG.treino.seed + fold,
    )


def executar_5fold(modelo_nome="tcct", ablation="completo"):
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    folds = criar_folds_espaciais(dataset)
    salvar_manifestos_folds(
        dataset,
        folds,
        CFG.dados.dir_resultados_v2 / "manifestos_folds",
    )
    resultados = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        stats = calcular_stats_treino(dataset, train_idx)
        _, _, metricas_patch, metricas_pixel = treinar_split(
            dataset,
            train_idx,
            val_idx,
            stats,
            rotulo=f"fold_{fold}",
            modelo_nome=modelo_nome,
            ablation=ablation,
            seed=CFG.treino.seed + fold,
        )
        resultados.append({
            "fold": fold,
            **{f"patch_{k}": v for k, v in metricas_patch.items()},
            **{f"pixel_{k}": v for k, v in metricas_pixel.items()},
        })

    df = pd.DataFrame(resultados)
    caminho = (
        CFG.dados.dir_resultados_v2
        / f"resumo_5fold_{modelo_nome}_{ablation}.csv"
    )
    df.to_csv(caminho, index=False)

    resumo_estatistico = df.select_dtypes(include=[np.number]).agg(["mean", "std"])
    resumo_estatistico.to_csv(
        CFG.dados.dir_resultados_v2
        / f"resumo_5fold_{modelo_nome}_{ablation}_media_std.csv"
    )
    print(df)
    print("\nMédia e desvio-padrão:")
    print(resumo_estatistico)
    return df



def _resumo_hierarquico_multiseed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa variação de seed e variação espacial entre folds.

    Primeiro, calcula média e desvio-padrão entre seeds dentro de cada fold.
    Depois, calcula a média e o desvio-padrão dos cinco valores médios de fold.
    Assim, a dispersão principal reportada na tabela representa generalização
    espacial, enquanto a instabilidade de inicialização permanece documentada.
    """
    metricas = [c for c in df.columns if c.startswith("patch_") or c.startswith("pixel_")]
    por_fold = df.groupby(["modelo", "ablation", "fold"])[metricas].agg(["mean", "std"]).reset_index()
    por_fold.columns = ["_".join([str(x) for x in col if str(x)]) for col in por_fold.columns]
    cols_mean = [c for c in por_fold.columns if c.endswith("_mean")]
    final_rows = []
    for (modelo, ablation), sub in por_fold.groupby(["modelo", "ablation"]):
        n_seeds = int(df.loc[(df["modelo"] == modelo) & (df["ablation"] == ablation), "seed"].nunique())
        row = {
            "modelo": modelo,
            "ablation": ablation,
            "n_folds": int(sub["fold"].nunique()),
            "n_seeds": n_seeds,
        }
        for col in cols_mean:
            base = col[:-5]
            row[f"{base}_mean"] = float(sub[col].mean())
            row[f"{base}_std_espacial"] = float(sub[col].std(ddof=1))
            seed_col = f"{base}_std"
            row[f"{base}_std_seed_media"] = float(sub[seed_col].mean()) if seed_col in sub else np.nan
        final_rows.append(row)
    return por_fold, pd.DataFrame(final_rows)



def _diretorio_execucao_multiseed(
    modelo_nome: str,
    ablation: str,
    seed: int,
    fold: int,
) -> Path:
    return (
        CFG.dados.dir_resultados_v2
        / "multiseed"
        / f"seed_{int(seed)}"
        / f"fold_{int(fold)}"
        / modelo_nome
        / ablation
    )


def _carregar_resultado_multiseed_concluido(
    modelo_nome: str,
    ablation: str,
    seed: int,
    fold: int,
) -> Optional[dict]:
    pasta = _diretorio_execucao_multiseed(
        modelo_nome,
        ablation,
        seed,
        fold,
    )
    obrigatorios = (
        pasta / "resumo_treinamento.json",
        pasta / "melhor_modelo_final.pth",
        pasta / "historico_completo.csv",
        pasta / "predicoes_patch_final.npz",
    )
    if not all(p.exists() for p in obrigatorios):
        return None
    try:
        resumo = json.loads(obrigatorios[0].read_text(encoding="utf-8"))
        metricas_patch = resumo["metricas_patch_final"]
        metricas_pixel = resumo["metricas_pixel_final"]
        if not np.isfinite(float(metricas_patch["rmse"])):
            return None
    except Exception:
        return None
    return {
        "modelo": modelo_nome,
        "ablation": ablation,
        "seed": int(seed),
        "fold": int(fold),
        "fase_selecionada": resumo.get("fase_selecionada", ""),
        **{f"patch_{k}": v for k, v in metricas_patch.items()},
        **{f"pixel_{k}": v for k, v in metricas_pixel.items()},
    }


def _salvar_resumos_multiseed_incrementais(
    resultados: Sequence[dict],
    modelo_nome: str,
    ablation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = CFG.dados.dir_resultados_v2 / "resumos_multiseed"
    out.mkdir(parents=True, exist_ok=True)
    base = f"{modelo_nome}_{ablation}"
    df = pd.DataFrame(resultados)
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    df = (
        df.drop_duplicates(subset=["modelo", "ablation", "seed", "fold"], keep="last")
        .sort_values(["seed", "fold"], kind="mergesort")
        .reset_index(drop=True)
    )
    df.to_csv(out / f"resultados_detalhados_{base}.csv", index=False)
    por_fold, final = _resumo_hierarquico_multiseed(df)
    por_fold.to_csv(out / f"resumo_por_fold_{base}.csv", index=False)
    final.to_csv(out / f"resumo_final_{base}.csv", index=False)
    return df, por_fold, final


def status_5fold_multiseed(
    modelo_nome="tcct",
    ablation="completo",
    seeds=None,
    folds_selecionados=None,
) -> pd.DataFrame:
    """Lista combinações concluídas e pendentes sem iniciar treinamento."""
    seeds = tuple(
        CFG.treino.seeds_experimento if seeds is None else map(int, seeds)
    )
    folds_lista = (
        list(range(CFG.treino.k_folds))
        if folds_selecionados is None
        else [int(f) for f in folds_selecionados]
    )
    linhas = []
    for seed in seeds:
        for fold in folds_lista:
            concluido = _carregar_resultado_multiseed_concluido(
                modelo_nome,
                ablation,
                seed,
                fold,
            )
            pasta = _diretorio_execucao_multiseed(
                modelo_nome,
                ablation,
                seed,
                fold,
            )
            retomada = (
                (pasta / "fase_A_supervisionada" / "checkpoint_retomada.pth").exists()
                or (pasta / "fase_B_rvog" / "checkpoint_retomada.pth").exists()
            )
            linhas.append({
                "seed": int(seed),
                "fold": int(fold),
                "status": (
                    "concluido"
                    if concluido is not None
                    else ("retomavel" if retomada else "pendente")
                ),
                "fase_selecionada": (
                    concluido.get("fase_selecionada", "")
                    if concluido is not None else ""
                ),
                "rmse_patch": (
                    concluido.get("patch_rmse", np.nan)
                    if concluido is not None else np.nan
                ),
                "pasta": str(pasta),
            })
    df = pd.DataFrame(linhas)
    print(df[["seed", "fold", "status", "fase_selecionada", "rmse_patch"]].to_string(index=False))
    print(
        "\nResumo:",
        df["status"].value_counts(dropna=False).to_dict(),
        flush=True,
    )
    return df


def executar_5fold_multiseed(
    modelo_nome="tcct",
    ablation="completo",
    seeds=None,
    folds_selecionados=None,
    retomar=True,
    sobrescrever_concluidos=False,
):
    """Executa folds×seeds com progresso explícito e retomada segura.

    - Combinações concluídas são detectadas e puladas.
    - Fases iniciadas pela v3.2.1 retomam da última época concluída.
    - Se a interrupção ocorrer no meio de uma época, somente essa época é refeita.
    - Resultados e estado global são atualizados após cada combinação.
    """
    seeds = tuple(
        CFG.treino.seeds_experimento if seeds is None else map(int, seeds)
    )
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    folds = criar_folds_espaciais(dataset)
    pasta_manifestos = CFG.dados.dir_resultados_v2 / "manifestos_folds"
    salvar_manifestos_folds(dataset, folds, pasta_manifestos)
    plotar_mapa_folds(
        dataset,
        folds,
        CFG.dados.dir_resultados_v2
        / "auditoria_espacial"
        / "mapa_folds_protocolo_final.png",
        f"Folds espaciais finais — buffer={CFG.dados.buffer_blocos}",
    )

    folds_lista = (
        list(range(len(folds)))
        if folds_selecionados is None
        else [int(f) for f in folds_selecionados]
    )
    for fold in folds_lista:
        if fold < 0 or fold >= len(folds):
            raise IndexError(f"Fold inválido: {fold}.")

    ordem = [(int(seed), int(fold)) for seed in seeds for fold in folds_lista]
    total = len(ordem)
    resultados: list[dict] = []

    pasta_status = CFG.dados.dir_resultados_v2 / "resumos_multiseed"
    pasta_status.mkdir(parents=True, exist_ok=True)
    base = f"{modelo_nome}_{ablation}"
    status_path = pasta_status / f"estado_execucao_{base}.json"
    estado_global = {
        "modelo": modelo_nome,
        "ablation": ablation,
        "total_combinacoes": total,
        "duracoes_s": {},
        "atualizado_em": pd.Timestamp.now().isoformat(),
    }
    if status_path.exists():
        try:
            anterior = json.loads(status_path.read_text(encoding="utf-8"))
            if (
                anterior.get("modelo") == modelo_nome
                and anterior.get("ablation") == ablation
            ):
                estado_global["duracoes_s"] = dict(anterior.get("duracoes_s", {}))
        except Exception:
            pass

    concluidas_inicio = 0
    for seed, fold in ordem:
        pronto = _carregar_resultado_multiseed_concluido(
            modelo_nome,
            ablation,
            seed,
            fold,
        )
        if pronto is not None and retomar and not sobrescrever_concluidos:
            resultados.append(pronto)
            concluidas_inicio += 1

    _salvar_resumos_multiseed_incrementais(
        resultados,
        modelo_nome,
        ablation,
    )

    print("\n" + "=" * 96, flush=True)
    print(
        f"PASSO 20 RETOMÁVEL — modelo={modelo_nome} | ablação={ablation}",
        flush=True,
    )
    print(
        f"Combinações totais: {total} | já concluídas: {concluidas_inicio} | "
        f"pendentes: {total - concluidas_inicio}",
        flush=True,
    )
    print(
        "Ordem: "
        + " → ".join(f"(seed={s}, fold={f})" for s, f in ordem),
        flush=True,
    )
    print("=" * 96, flush=True)

    duracoes_sessao = []
    try:
        for posicao, (seed, fold) in enumerate(ordem, start=1):
            pronto = _carregar_resultado_multiseed_concluido(
                modelo_nome,
                ablation,
                seed,
                fold,
            )
            if pronto is not None and retomar and not sobrescrever_concluidos:
                print(
                    f"\n⏭️ [{posicao:02d}/{total:02d}] seed={seed} | fold={fold} "
                    "— já concluído; pulando.",
                    flush=True,
                )
                continue

            pasta_execucao = _diretorio_execucao_multiseed(
                modelo_nome,
                ablation,
                seed,
                fold,
            )
            if sobrescrever_concluidos and pasta_execucao.exists():
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                backup = pasta_execucao.with_name(
                    pasta_execucao.name + f"_ANTES_RETREINO_{timestamp}"
                )
                shutil.move(str(pasta_execucao), str(backup))

            n_concluidas = len({
                (int(r["seed"]), int(r["fold"])) for r in resultados
            })
            pendentes_apos = total - n_concluidas - 1
            duracoes_conhecidas = [
                float(v)
                for v in estado_global.get("duracoes_s", {}).values()
                if np.isfinite(float(v)) and float(v) > 0
            ] + duracoes_sessao
            eta_texto = "ETA ainda desconhecida"
            if duracoes_conhecidas:
                media = float(np.mean(duracoes_conhecidas))
                eta_h = max(0, pendentes_apos + 1) * media / 3600.0
                eta_texto = f"ETA aproximada={eta_h:.2f} h"

            print("\n" + "#" * 96, flush=True)
            print(
                f"▶ EXECUÇÃO [{posicao:02d}/{total:02d}] | seed={seed} | "
                f"fold={fold} | concluídas={n_concluidas} | "
                f"pendentes incluindo esta={total - n_concluidas} | {eta_texto}",
                flush=True,
            )
            print("#" * 96, flush=True)

            inicio_combo = time.time()
            train_idx, val_idx = folds[fold]
            stats = calcular_stats_treino(dataset, train_idx)
            _, _, metricas_patch, metricas_pixel = treinar_split(
                dataset,
                train_idx,
                val_idx,
                stats,
                rotulo=f"multiseed/seed_{seed}/fold_{fold}",
                modelo_nome=modelo_nome,
                ablation=ablation,
                seed=seed,
            )
            duracao = time.time() - inicio_combo
            duracoes_sessao.append(duracao)

            resultado = {
                "modelo": modelo_nome,
                "ablation": ablation,
                "seed": seed,
                "fold": fold,
                **{f"patch_{k}": v for k, v in metricas_patch.items()},
                **{f"pixel_{k}": v for k, v in metricas_pixel.items()},
            }
            resumo_path = (
                _diretorio_execucao_multiseed(
                    modelo_nome, ablation, seed, fold
                )
                / "resumo_treinamento.json"
            )
            if resumo_path.exists():
                try:
                    resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
                    resultado["fase_selecionada"] = resumo.get(
                        "fase_selecionada", ""
                    )
                except Exception:
                    resultado["fase_selecionada"] = ""
            resultados.append(resultado)

            chave = f"seed_{seed}_fold_{fold}"
            estado_global["duracoes_s"][chave] = float(duracao)
            estado_global["ultima_execucao"] = {
                "seed": seed,
                "fold": fold,
                "duracao_s": float(duracao),
                "concluida_em": pd.Timestamp.now().isoformat(),
            }
            estado_global["atualizado_em"] = pd.Timestamp.now().isoformat()
            _salvar_resumos_multiseed_incrementais(
                resultados,
                modelo_nome,
                ablation,
            )
            status_path.write_text(
                json.dumps(estado_global, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            media_sessao = float(np.mean(duracoes_sessao))
            restantes = total - len({
                (int(r["seed"]), int(r["fold"])) for r in resultados
            })
            print(
                f"\n✅ seed={seed} | fold={fold} concluído em "
                f"{duracao / 60.0:.1f} min. Restantes={restantes}; "
                f"ETA da sessão≈{restantes * media_sessao / 3600.0:.2f} h.",
                flush=True,
            )
    except KeyboardInterrupt:
        estado_global["interrompido_em"] = pd.Timestamp.now().isoformat()
        estado_global["mensagem"] = (
            "Execução interrompida pelo usuário. Reabra o notebook, execute "
            "Passos 00 e 01 e rode novamente o Passo 20."
        )
        status_path.write_text(
            json.dumps(estado_global, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _salvar_resumos_multiseed_incrementais(
            resultados,
            modelo_nome,
            ablation,
        )
        print(
            "\n⏸️ Passo 20 interrompido com estado salvo. Na retomada, "
            "combinações concluídas serão puladas e a fase atual retomará da "
            "última época salva pela v3.2.1.",
            flush=True,
        )
        raise

    df, por_fold, final = _salvar_resumos_multiseed_incrementais(
        resultados,
        modelo_nome,
        ablation,
    )
    estado_global["concluido"] = len(df) == total
    estado_global["combinacoes_concluidas"] = int(len(df))
    estado_global["atualizado_em"] = pd.Timestamp.now().isoformat()
    status_path.write_text(
        json.dumps(estado_global, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nResumo final:", flush=True)
    print(final.to_string(index=False), flush=True)
    return df, por_fold, final


def criar_holdout_espacial_70_20_10(
    dataset: TropiSARDatasetV2,
    out: Optional[Path | str] = None,
) -> dict:
    """Cria faixas espaciais contíguas aproximando 70/20/10 por blocos.

    O conjunto de teste ocupa as colunas de bloco mais a leste, a validação
    ocupa a faixa adjacente e o treino permanece a oeste. Nenhum grupo espacial
    é dividido. Não há buffer, pois este protocolo adicional usa as 10.500
    amostras; por isso ele complementa, mas não substitui, a validação cruzada.
    """
    out = Path(
        out
        or (
            CFG.dados.dir_resultados_v2
            / "holdout_espacial_70_20_10"
            / "protocolo"
        )
    )
    out.mkdir(parents=True, exist_ok=True)
    df = dataset.index.copy().reset_index(drop=True)

    colunas = np.sort(df["block_col"].unique())
    if len(colunas) < 3:
        raise RuntimeError("Poucas colunas de bloco para formar 70/20/10.")

    contagens = (
        df.groupby("block_col").size().reindex(colunas, fill_value=0).to_numpy()
    )
    acumulada = np.cumsum(contagens)
    total = int(len(df))
    alvo = np.array([0.70, 0.20, 0.10]) * total

    melhor = None
    for i in range(1, len(colunas) - 1):
        n_treino = int(acumulada[i - 1])
        for j in range(i + 1, len(colunas)):
            n_val = int(acumulada[j - 1] - acumulada[i - 1])
            n_teste = total - n_treino - n_val
            if min(n_treino, n_val, n_teste) <= 0:
                continue
            observado = np.array([n_treino, n_val, n_teste], dtype=float)
            score = float(np.sum(((observado - alvo) / np.maximum(alvo, 1)) ** 2))
            if melhor is None or score < melhor[0]:
                melhor = (score, i, j, n_treino, n_val, n_teste)

    if melhor is None:
        raise RuntimeError("Não foi possível formar o holdout espacial 70/20/10.")

    _, i, j, _, _, _ = melhor
    cols_treino = set(map(int, colunas[:i]))
    cols_val = set(map(int, colunas[i:j]))
    cols_teste = set(map(int, colunas[j:]))

    idx_treino = df.index[df["block_col"].isin(cols_treino)].to_numpy(int)
    idx_val = df.index[df["block_col"].isin(cols_val)].to_numpy(int)
    idx_teste = df.index[df["block_col"].isin(cols_teste)].to_numpy(int)

    particoes = {
        "treino": idx_treino,
        "validacao": idx_val,
        "teste": idx_teste,
    }
    _particoes_disjuntas(particoes)
    todos = np.sort(np.concatenate(list(particoes.values())))
    if not np.array_equal(todos, np.arange(total)):
        raise AssertionError("O holdout não cobre exatamente todo o dataset.")

    manifesto = []
    for nome, indices in particoes.items():
        sub = df.iloc[indices].copy()
        sub.insert(0, "dataset_index", indices)
        sub.insert(1, "particao", nome)
        manifesto.append(sub)
        pd.DataFrame({"dataset_index": indices}).to_csv(
            out / f"manifesto_{nome}.csv",
            index=False,
        )
    manifesto_df = pd.concat(manifesto, ignore_index=True)
    manifesto_df.to_csv(out / "manifesto_holdout_70_20_10.csv", index=False)

    auditorias = []
    for a, b in (("treino", "validacao"), ("validacao", "teste"), ("treino", "teste")):
        auditorias.append({
            "particao_a": a,
            "particao_b": b,
            **_distancias_particoes(df, particoes[a], particoes[b]),
        })
    pd.DataFrame(auditorias).to_csv(
        out / "distancias_entre_particoes.csv",
        index=False,
    )

    nomes_particoes_figura = {
        "treino": "Training",
        "validacao": "Validation",
        "teste": "Test",
    }
    fig, ax = plt.subplots(figsize=(11, 8))
    for nome, indices in particoes.items():
        sub = df.iloc[indices]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=9,
            alpha=0.65,
            label=nomes_particoes_figura.get(nome, nome),
        )
    ax.set_title("Additional spatial holdout — approximately 70/20/10")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "mapa_holdout_70_20_10.png", dpi=220)
    plt.close(fig)

    resumo = {
        "total": total,
        "treino": int(len(idx_treino)),
        "validacao": int(len(idx_val)),
        "teste": int(len(idx_teste)),
        "fracao_treino": float(len(idx_treino) / total),
        "fracao_validacao": float(len(idx_val) / total),
        "fracao_teste": float(len(idx_teste) / total),
        "blocos_treino": sorted(cols_treino),
        "blocos_validacao": sorted(cols_val),
        "blocos_teste": sorted(cols_teste),
        "divisao_por_patch_aleatorio": False,
        "grupos_divididos_entre_particoes": False,
        "buffer_blocos": 0,
        "motivo_sem_buffer": (
            "protocolo adicional solicitado para usar todas as amostras; "
            "a avaliação principal continua sendo 5-fold espacial"
        ),
        "teste_usado_em_normalizacao": False,
        "teste_usado_em_early_stopping": False,
        "teste_usado_em_calibracao_rvog": False,
    }
    (out / "resumo_holdout_70_20_10.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)
    return {"particoes": particoes, "resumo": resumo, "pasta": out}


def executar_holdout_espacial_70_20_10(
    seed: Optional[int] = None,
    paciencia_maior: bool = True,
    device=DEVICE,
):
    """Treina em 70%, seleciona por 20% e avalia uma única vez nos 10% finais."""
    seed = int(CFG.treino.seed if seed is None else seed)
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    protocolo = criar_holdout_espacial_70_20_10(dataset)
    p = protocolo["particoes"]
    stats = calcular_stats_treino(dataset, p["treino"])

    treino_original = copy.deepcopy(CFG.treino)
    try:
        if paciencia_maior:
            CFG.treino.fase_a_epocas = max(CFG.treino.fase_a_epocas, 150)
            CFG.treino.fase_a_min_epocas = max(CFG.treino.fase_a_min_epocas, 15)
            CFG.treino.fase_a_paciencia = max(CFG.treino.fase_a_paciencia, 20)
            CFG.treino.fase_a_scheduler_paciencia = max(
                CFG.treino.fase_a_scheduler_paciencia,
                5,
            )
            CFG.treino.fase_b_epocas = max(CFG.treino.fase_b_epocas, 50)
            CFG.treino.fase_b_min_epocas = max(CFG.treino.fase_b_min_epocas, 10)
            CFG.treino.fase_b_paciencia = max(CFG.treino.fase_b_paciencia, 15)
            CFG.treino.fase_b_scheduler_paciencia = max(
                CFG.treino.fase_b_scheduler_paciencia,
                5,
            )
            CFG.treino.executar_teste_sobreajuste = False

        rotulo = f"holdout_espacial_70_20_10/seed_{seed}"
        model, historico, metricas_val, metricas_pixel_val = treinar_split(
            dataset,
            p["treino"],
            p["validacao"],
            stats,
            rotulo=rotulo,
            modelo_nome="tcct",
            ablation="completo",
            seed=seed,
            device=device,
        )

        teste_loader = DataLoader(
            Subset(dataset, list(map(int, p["teste"]))),
            batch_size=CFG.treino.batch_size,
            shuffle=False,
            num_workers=CFG.treino.num_workers,
            pin_memory=PIN_MEMORY,
        )
        metricas_teste, metricas_pixel_teste, xai, y_true, y_pred = avaliar_modelo(
            model,
            teste_loader,
            stats,
            device,
            ablation="completo",
            xai=True,
        )

        out = (
            CFG.dados.dir_resultados_v2
            / rotulo
            / "tcct"
            / "completo"
        )
        xai.to_csv(out / "xai_teste_final_10pct.csv", index=False)
        np.savez(
            out / "predicoes_teste_final_10pct.npz",
            y_true=y_true,
            y_pred=y_pred,
        )
        resumo = {
            "protocolo": protocolo["resumo"],
            "seed": seed,
            "paciencia_maior": bool(paciencia_maior),
            "metricas_validacao_selecao": metricas_val,
            "metricas_pixel_validacao": metricas_pixel_val,
            "metricas_teste_final_10pct": metricas_teste,
            "metricas_pixel_teste_final_10pct": metricas_pixel_teste,
            "teste_usado_para_selecionar_checkpoint": False,
            "teste_usado_para_ajustar_hiperparametros": False,
            "observacao": (
                "Experimento adicional de holdout; não substitui a média e "
                "o desvio-padrão do protocolo espacial 5-fold multiseed."
            ),
        }
        (out / "resumo_holdout_teste_final.json").write_text(
            json.dumps(resumo, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nTeste final intocado (10%):", metricas_teste, flush=True)
        return model, historico, resumo
    finally:
        CFG.treino = treino_original


def executar_smoke_test_revisao(fold: int = 0):
    """Executa um teste curto da arquitetura completa e da Fase B.

    Os hiperparâmetros científicos são restaurados ao final. Este teste não gera
    resultados publicáveis; ele serve apenas para confirmar shapes, gradientes,
    salvamento de checkpoints e ativação real da loss RVoG.
    """
    treino_original = copy.deepcopy(CFG.treino)
    try:
        CFG.treino.fase_a_epocas = 3
        CFG.treino.fase_a_min_epocas = 1
        CFG.treino.fase_a_paciencia = 3
        CFG.treino.fase_b_epocas = 3
        CFG.treino.fase_b_min_epocas = 1
        CFG.treino.fase_b_paciencia = 3
        CFG.treino.executar_teste_sobreajuste = True
        return treinar_holdout("tcct", "completo", fold)
    finally:
        CFG.treino = treino_original


def _particoes_disjuntas(particoes: dict[str, Sequence[int]]) -> None:
    nomes = list(particoes)
    for i, a in enumerate(nomes):
        sa = set(map(int, particoes[a]))
        if len(sa) != len(particoes[a]):
            raise AssertionError(f"Índices duplicados dentro da partição {a}.")
        for b in nomes[i + 1:]:
            inter = sa & set(map(int, particoes[b]))
            if inter:
                raise AssertionError(f"Interseção entre {a} e {b}: {len(inter)} amostras.")


def criar_protocolo_finetuning_espacial(dataset: TropiSARDatasetV2, out: Optional[Path] = None) -> dict:
    """Cria protocolo global + adaptação local + teste final intocado.

    O domínio local é uma faixa contígua na borda leste (maiores colunas de
    bloco). Ele é excluído integralmente do treinamento e da validação global.
    Dentro do domínio local, faixas contíguas em Y formam ajuste, validação local
    e teste final, separadas por linhas-buffer. O teste final não participa de
    early stopping, calibração RVoG, normalização ou escolha de hiperparâmetros.
    """
    out = Path(out or (CFG.dados.dir_resultados_v2 / "protocolo_finetuning"))
    out.mkdir(parents=True, exist_ok=True)
    df = dataset.index.copy().reset_index(drop=True)
    cols = np.sort(df["block_col"].unique())
    n_local_cols = max(1, int(math.ceil(len(cols) * CFG.dados.fracao_dominio_local)))
    local_cols = set(cols[-n_local_cols:])
    col_min_local = min(local_cols)
    buffer = int(CFG.dados.buffer_finetuning_blocos)

    mascara_local = df["block_col"].isin(local_cols)
    mascara_global = df["block_col"] < (col_min_local - buffer)
    local = df[mascara_local]
    rows = np.sort(local["block_row"].unique())
    if len(rows) < 7:
        raise RuntimeError(
            "O domínio local possui poucas linhas de bloco para ajuste, validação, "
            "teste e dois buffers. Reduza o tamanho do bloco ou a faixa local."
        )
    n = len(rows)
    n_aj = max(1, int(math.floor(n * CFG.dados.fracao_ajuste_local)))
    n_val = max(1, int(math.floor(n * CFG.dados.fracao_validacao_local)))
    if n_aj + n_val + 2 * buffer >= n:
        n_aj = max(1, n // 3)
        n_val = max(1, n // 4)
    ajuste_rows = set(rows[:n_aj])
    inicio_val = n_aj + buffer
    valid_rows = set(rows[inicio_val:inicio_val + n_val])
    inicio_teste = inicio_val + n_val + buffer
    teste_rows = set(rows[inicio_teste:])
    if not ajuste_rows or not valid_rows or not teste_rows:
        raise RuntimeError("Não foi possível formar as três partições locais.")

    idx_ajuste = df.index[mascara_local & df["block_row"].isin(ajuste_rows)].to_numpy(int)
    idx_val_local = df.index[mascara_local & df["block_row"].isin(valid_rows)].to_numpy(int)
    idx_teste = df.index[mascara_local & df["block_row"].isin(teste_rows)].to_numpy(int)
    idx_global_pool = df.index[mascara_global].to_numpy(int)

    grupos = df.iloc[idx_global_pool]["group"].astype(str).to_numpy()
    if len(np.unique(grupos)) < CFG.treino.k_folds:
        raise RuntimeError("Poucos grupos no pool global para validação interna.")
    gkf = GroupKFold(n_splits=CFG.treino.k_folds)
    rel_tr, rel_va = next(gkf.split(np.zeros(len(idx_global_pool)), groups=grupos))
    idx_global_train = idx_global_pool[rel_tr]
    idx_global_val = idx_global_pool[rel_va]
    idx_global_train = aplicar_buffer_espacial(
        df, idx_global_train, idx_global_val, CFG.dados.buffer_blocos
    )

    particoes = {
        "global_treino": idx_global_train,
        "global_validacao": idx_global_val,
        "local_ajuste": idx_ajuste,
        "local_validacao": idx_val_local,
        "teste_final": idx_teste,
    }
    _particoes_disjuntas(particoes)

    manifesto = []
    for nome, indices in particoes.items():
        d = df.iloc[np.asarray(indices, dtype=int)].copy()
        d.insert(0, "dataset_index", np.asarray(indices, dtype=int))
        d.insert(1, "particao", nome)
        manifesto.append(d)
    manifesto_df = pd.concat(manifesto, ignore_index=True)
    manifesto_df.to_csv(out / "manifesto_protocolo_finetuning.csv", index=False)
    for nome, indices in particoes.items():
        pd.DataFrame({"dataset_index": np.asarray(indices, dtype=int)}).to_csv(
            out / f"manifesto_{nome}.csv", index=False
        )

    auditorias = []
    pares = [
        ("global_treino", "global_validacao"),
        ("global_treino", "local_ajuste"),
        ("local_ajuste", "local_validacao"),
        ("local_validacao", "teste_final"),
        ("local_ajuste", "teste_final"),
    ]
    for a, b in pares:
        auditorias.append({"particao_a": a, "particao_b": b, **_distancias_particoes(df, particoes[a], particoes[b])})
    pd.DataFrame(auditorias).to_csv(out / "distancias_entre_particoes.csv", index=False)

    nomes_particoes_figura = {
        "global_treino": "Global training",
        "global_validacao": "Global validation",
        "local_ajuste": "Local adaptation",
        "local_validacao": "Local validation",
        "teste_final": "Final test",
    }
    fig, ax = plt.subplots(figsize=(11, 8.6))
    for nome, indices in particoes.items():
        d = df.iloc[np.asarray(indices, dtype=int)]
        ax.scatter(
            d["x"],
            d["y"],
            s=9,
            alpha=0.65,
            label=nomes_particoes_figura.get(nome, nome),
        )
    ax.set_title("Spatial protocol for global training and local fine-tuning")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=3,
        frameon=True,
        framealpha=0.97,
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    fig.savefig(
        out / "mapa_protocolo_finetuning.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    resumo = {nome: int(len(indices)) for nome, indices in particoes.items()}
    resumo.update({
        "buffer_global_blocos": int(CFG.dados.buffer_blocos),
        "buffer_local_blocos": buffer,
        "colunas_bloco_dominio_local": sorted(map(int, local_cols)),
        "teste_final_usado_no_treino_global": False,
        "teste_final_usado_no_early_stopping_global": False,
        "teste_final_usado_na_calibracao_rvog": False,
        "teste_final_usado_no_finetuning": False,
    })
    (out / "resumo_protocolo_finetuning.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"particoes": particoes, "resumo": resumo, "pasta": out}


def fine_tuning_localizado(
    checkpoint_global: Path | str,
    dataset: TropiSARDatasetV2,
    indices_ajuste: Sequence[int],
    indices_validacao_local: Sequence[int],
    indices_teste_final: Sequence[int],
    epocas: int = 200,
    lr: float = 1e-5,
    ablation: str = "completo",
    device=DEVICE,
):
    """Fine-tuning local sem usar o teste para selecionar a melhor época."""
    checkpoint = torch.load(
        checkpoint_global,
        map_location="cpu",
        weights_only=True,
    )
    stats = StatsTreino(**checkpoint["stats"])
    model = TCCTv2(CFG.modelo, stats).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    out = CFG.dados.dir_resultados_v2 / "protocolo_finetuning" / "fine_tuning_local"
    out.mkdir(parents=True, exist_ok=True)

    n_fisica_treino = contar_fisica_valida(dataset, indices_ajuste)
    n_fisica_val = contar_fisica_valida(dataset, indices_validacao_local)
    usar_fisica = (
        CFG.treino.modo_fisica != "off"
        and n_fisica_treino >= CFG.treino.min_amostras_fisica_treino
        and n_fisica_val >= CFG.treino.min_amostras_fisica_validacao
    )

    resultado = treinar_fase(
        model=model,
        dataset=dataset,
        train_idx=np.asarray(indices_ajuste, dtype=int),
        val_idx=np.asarray(indices_validacao_local, dtype=int),
        stats=stats,
        out=out,
        nome_fase="fine_tuning_local",
        modelo_nome="tcct",
        ablation=ablation,
        epocas=epocas,
        min_epocas=10,
        paciencia=20,
        lr=lr,
        lr_min=1e-7,
        scheduler_paciencia=5,
        lambda_fn=(
            (lambda ep: lambda_rvog_fase_b(ep, CFG.treino))
            if usar_fisica
            else (lambda _: 0.0)
        ),
        seed=CFG.treino.seed + 50_000,
        device=device,
    )
    model.load_state_dict(resultado["model_state_dict"])

    teste_loader = DataLoader(
        Subset(dataset, list(map(int, indices_teste_final))),
        batch_size=CFG.treino.batch_size,
        shuffle=False,
        num_workers=0,
    )
    metricas_teste, metricas_pixel, xai, y_true, y_pred = avaliar_modelo(
        model,
        teste_loader,
        stats,
        device,
        ablation,
        True,
    )
    xai.to_csv(out / "xai_teste_final.csv", index=False)
    np.savez(
        out / "predicoes_teste_final.npz",
        y_true=y_true,
        y_pred=y_pred,
    )
    torch.save(
        {
            "model_state_dict": resultado["model_state_dict"],
            "stats": asdict(stats),
            "metricas_teste_final": metricas_teste,
            "metricas_pixel_final": metricas_pixel,
            "usou_rvog": usar_fisica,
        },
        out / "melhor_modelo_finetune.pth",
    )
    print("Teste final após fine-tuning:", metricas_teste)
    return model, resultado["historico"], metricas_teste, metricas_pixel


def executar_finetuning_espacial_seguro(device=DEVICE):
    """Executa o protocolo completo sem permitir que o teste final contamine etapas anteriores."""
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    protocolo = criar_protocolo_finetuning_espacial(dataset)
    p = protocolo["particoes"]
    stats = calcular_stats_treino(dataset, p["global_treino"])
    rotulo = "protocolo_finetuning/modelo_global"
    treinar_split(
        dataset,
        p["global_treino"],
        p["global_validacao"],
        stats,
        rotulo=rotulo,
        modelo_nome="tcct",
        ablation="completo",
        seed=CFG.treino.seeds_experimento[0],
        device=device,
    )
    checkpoint = (
        CFG.dados.dir_resultados_v2 / rotulo / "tcct" / "completo" /
        "melhor_modelo_final.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    # A geometria/atenuação do refinamento local é preparada somente com a
    # partição de ajuste; a validação local apenas verifica cobertura física.
    preparar_geometria_rvog(
        dataset,
        p["local_ajuste"],
        p["local_validacao"],
        CFG.dados.dir_resultados_v2 / "protocolo_finetuning" / "geometria_local",
    )
    return fine_tuning_localizado(
        checkpoint,
        dataset,
        p["local_ajuste"],
        p["local_validacao"],
        p["teste_final"],
        epocas=200,
        lr=1e-5,
        ablation="completo",
        device=device,
    )




# ================================================================
# 8. BASELINE RANDOM FOREST, AGB, MAPAS E INFERÊNCIA
# ================================================================
def features_rf(amostra):
    feats=[amostra["optical"].float().numpy()]
    for b in ("l","p"):
        z=amostra[f"sar_{b}"];feats += [z.abs().mean((-2,-1)).numpy(),z.real.mean((-2,-1)).numpy(),z.imag.mean((-2,-1)).numpy(),amostra[f"canonical_corr_{b}"].numpy()]
    return np.concatenate(feats).astype(np.float32)


def baseline_random_forest(
    dataset,
    train_idx,
    val_idx,
    n_estimators=400,
    out_dir: Optional[Path | str] = None,
):
    X_train, y_train, X_val, y_val = [], [], [], []
    for destino, indices in (
        ((X_train, y_train), train_idx),
        ((X_val, y_val), val_idx),
    ):
        X, Y = destino
        for i in tqdm(indices, desc="Features RF"):
            amostra = dataset[int(i)]
            mask = amostra["valid_mask"].numpy().astype(bool)
            X.append(features_rf(amostra))
            Y.append(amostra["chm"].numpy()[mask].mean())

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=CFG.treino.seed,
        max_features="sqrt",
        min_samples_leaf=2,
    )
    rf.fit(np.asarray(X_train), np.asarray(y_train))
    pred = rf.predict(np.asarray(X_val))
    metricas = metricas_np(y_val, pred)
    print("Random Forest:", metricas, flush=True)

    if out_dir is not None:
        import joblib
        pasta = Path(out_dir)
        pasta.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, pasta / "random_forest.joblib")
        pd.DataFrame({
            "y_true": np.asarray(y_val),
            "y_pred": np.asarray(pred),
        }).to_csv(pasta / "predicoes_random_forest.csv", index=False)
        (pasta / "metricas_random_forest.json").write_text(
            json.dumps(metricas, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return rf, metricas, np.asarray(y_val), np.asarray(pred)



def executar_rf_5fold_multiseed(seeds=None, n_estimators=400):
    seeds = tuple(CFG.treino.seeds_experimento if seeds is None else map(int, seeds))
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    folds = criar_folds_espaciais(dataset)
    resultados = []
    for seed in seeds:
        for fold, (tr, va) in enumerate(folds):
            estado = CFG.treino.seed
            CFG.treino.seed = int(seed)
            try:
                _, met, _, _ = baseline_random_forest(
                    dataset, tr, va, n_estimators=n_estimators,
                    out_dir=CFG.dados.dir_resultados_v2 / "multiseed" /
                    f"seed_{seed}" / f"fold_{fold}" / "random_forest",
                )
            finally:
                CFG.treino.seed = estado
            resultados.append({
                "modelo": "random_forest", "ablation": "completo",
                "seed": int(seed), "fold": int(fold),
                **{f"patch_{k}": v for k, v in met.items()},
            })
    df = pd.DataFrame(resultados)
    out = CFG.dados.dir_resultados_v2 / "resumos_multiseed"; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "resultados_detalhados_random_forest_completo.csv", index=False)
    por_fold, final = _resumo_hierarquico_multiseed(df)
    por_fold.to_csv(out / "resumo_por_fold_random_forest_completo.csv", index=False)
    final.to_csv(out / "resumo_final_random_forest_completo.csv", index=False)
    return df, por_fold, final


def consolidar_tabela_revisao(pasta: Optional[Path] = None) -> pd.DataFrame:
    """Consolida a nova Table I com direção da métrica e média ± desvio espacial."""
    pasta = Path(pasta or (CFG.dados.dir_resultados_v2 / "resumos_multiseed"))
    arquivos = sorted(pasta.glob("resumo_final_*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum resumo final em {pasta}")
    df = pd.concat([pd.read_csv(p) for p in arquivos], ignore_index=True)
    nomes = {
        ("tcct", "completo"): "TCCT (complex phase + co-attention)",
        ("tcct", "amplitude_only"): "TCCT (amplitude only)",
        ("tcct_sem_coatencao", "completo"): "TCCT without co-attention",
        ("cnn", "completo"): "CNN early fusion",
        ("vit", "completo"): "Vision Transformer early fusion",
        ("random_forest", "completo"): "Random Forest",
    }
    linhas = []
    for _, r in df.iterrows():
        chave = (str(r["modelo"]), str(r["ablation"]))
        nome = nomes.get(chave, f"{chave[0]} / {chave[1]}")
        def fmt(metrica, casas=2):
            media = r.get(f"patch_{metrica}_mean", np.nan)
            sd = r.get(f"patch_{metrica}_std_espacial", np.nan)
            return f"{media:.{casas}f} ± {sd:.{casas}f}"
        linhas.append({
            "Model": nome,
            "RMSE (m) ↓": fmt("rmse", 2),
            "R² ↑": fmt("r2", 3),
            "Bias (m; closer to 0)": fmt("bias", 2),
            "Std. prediction (m)": fmt("std_pred", 2),
            "Folds": int(r.get("n_folds", 0)),
            "Seeds per fold": int(r.get("n_seeds", len(CFG.treino.seeds_experimento))),
        })
    tabela = pd.DataFrame(linhas)
    ordem = [nomes[k] for k in nomes]
    tabela["_ordem"] = tabela["Model"].map({n: i for i, n in enumerate(ordem)}).fillna(999)
    tabela = tabela.sort_values("_ordem").drop(columns="_ordem")
    out = CFG.dados.dir_resultados_v2 / "tabelas_revisao"; out.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(out / "Table_I_revision.csv", index=False)
    tabela.to_excel(out / "Table_I_revision.xlsx", index=False)
    latex = tabela.to_latex(index=False, escape=False, caption="Spatial cross-validation results (mean ± standard deviation across fold means).", label="tab:revision_results")
    (out / "Table_I_revision.tex").write_text(latex, encoding="utf-8")
    print(tabela.to_string(index=False), flush=True)
    return tabela


def consolidar_gating_modalidade(
    pasta_resultados: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consolidates learned modality-gating weights across folds and seeds.

    The figure displays every available observation as a translucent point cloud
    and overlays a rolling mean. These are descriptive internal gating weights,
    not causal feature-importance scores and not raw Transformer attention maps.
    """
    raiz = Path(pasta_resultados or CFG.dados.dir_resultados_v2)
    arquivos = list(
        raiz.glob(
            "multiseed/seed_*/fold_*/tcct/*/"
            "pesos_gating_modalidade_final.csv"
        )
    )
    if not arquivos:
        raise FileNotFoundError("No modality-gating weight files were found.")

    partes = []
    padrao = re.compile(r"seed_(\d+).+fold_(\d+).+tcct[/\\]([^/\\]+)")
    for caminho in arquivos:
        correspondencia = padrao.search(str(caminho))
        dados = pd.read_csv(caminho)
        dados["seed"] = int(correspondencia.group(1)) if correspondencia else -1
        dados["fold"] = int(correspondencia.group(2)) if correspondencia else -1
        dados["ablation"] = (
            correspondencia.group(3) if correspondencia else "unknown"
        )
        partes.append(dados)

    df = pd.concat(partes, ignore_index=True)
    bins = [0, 15, 20, 25, 30, 35, 40, np.inf]
    labels = ["<15", "15–20", "20–25", "25–30", "30–35", "35–40", "≥40"]
    df["CHM_class"] = pd.cut(
        df["CHM_m"], bins=bins, labels=labels, right=False
    )
    resumo = (
        df.groupby(["ablation", "CHM_class"], observed=True)[
            ["Optico", "SAR_L", "SAR_P"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    out = CFG.dados.dir_resultados_v2 / "xai_revisao"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "gating_modalidade_todas_execucoes.csv", index=False)
    resumo.to_csv(out / "gating_modalidade_por_classe_chm.csv", index=False)

    nomes_modelos = {
        "completo": "Complete TCCT",
        "amplitude_only": "Amplitude-only TCCT",
        "sem_coatencao": "TCCT without co-attention",
    }
    modalidades = (
        ("Optico", "Optical"),
        ("SAR_L", "SAR L-band"),
        ("SAR_P", "SAR P-band"),
    )
    cores_modalidade = {
        "Optico": "#0072B2",
        "SAR_L": "#D55E00",
        "SAR_P": "#009E73",
    }

    for ablation, sub in df.groupby("ablation"):
        ordenado = sub.sort_values("CHM_m").reset_index(drop=True)
        if ordenado.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 7))
        janela = max(50, min(500, len(ordenado) // 20))

        for coluna, rotulo in modalidades:
            cor = cores_modalidade[coluna]
            ax.scatter(
                ordenado["CHM_m"],
                ordenado[coluna],
                s=8,
                alpha=0.085,
                rasterized=True,
                color=cor,
                linewidths=0,
                label="_nolegend_",
                zorder=1,
            )
            tendencia = (
                ordenado[coluna]
                .rolling(janela, center=True, min_periods=1)
                .mean()
            )
            ax.plot(
                ordenado["CHM_m"],
                tendencia,
                linewidth=2.8,
                color=cor,
                label=rotulo,
                zorder=3,
            )

        nome_modelo = nomes_modelos.get(
            str(ablation), str(ablation).replace("_", " ").title()
        )
        ax.set_title(
            "Learned Modality-Gating Weight Distribution"
            f" — {nome_modelo}"
        )
        ax.set_xlabel("LiDAR-derived reference canopy height (m)")
        ax.set_ylabel("Modality-gating weight")
        ax.set_xlim(
            max(0.0, float(ordenado["CHM_m"].min())),
            float(ordenado["CHM_m"].max()),
        )
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, framealpha=0.96)
        fig.tight_layout()

        # Backward-compatible filename used by the current notebook.
        caminho_padrao = out / f"gating_weights_{ablation}.png"
        fig.savefig(caminho_padrao, dpi=300, bbox_inches="tight")
        # Explicit filename for the revised manuscript.
        fig.savefig(
            out / f"gating_weights_cloud_{ablation}.png",
            dpi=300,
            bbox_inches="tight",
        )
        if str(ablation) == "completo":
            fig.savefig(
                out / "xai_modality_gating_cloud_complete_tcct.png",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

    return df, resumo


# -----------------------------------------------------------------------------
# Strict Phase-A out-of-fold pairwise comparisons (no retraining)
# -----------------------------------------------------------------------------
def _media_chm_dataset_index(
    dataset: TropiSARDatasetV2,
    dataset_index: int,
) -> float:
    """Returns the valid-pixel mean CHM for one canonical dataset index."""
    amostra = dataset[int(dataset_index)]
    chm = amostra["chm"].detach().cpu().numpy().astype(np.float64)
    mask = amostra["valid_mask"].detach().cpu().numpy().astype(bool)
    valores = chm[mask]
    if valores.size == 0:
        raise RuntimeError(
            f"Sample {dataset_index} has no valid CHM pixel."
        )
    return float(np.mean(valores))


def _verdade_canonica_indices(
    dataset: TropiSARDatasetV2,
    indices_validacao: Sequence[int],
) -> np.ndarray:
    """Builds y_true directly from dataset indices in a declared order."""
    return np.asarray(
        [
            _media_chm_dataset_index(dataset, int(i))
            for i in indices_validacao
        ],
        dtype=np.float64,
    )


def _alinhar_predicoes_por_indice(
    pred: pd.DataFrame,
    indices_canonicos: Sequence[int],
    contexto: str,
) -> pd.DataFrame:
    """Aligns one prediction table by dataset_index and audits uniqueness."""
    obrigatorias = {"dataset_index", "y_true", "y_pred"}
    ausentes = obrigatorias - set(pred.columns)
    if ausentes:
        raise KeyError(
            f"Missing columns in {contexto}: {sorted(ausentes)}"
        )

    tabela = pred.loc[:, ["dataset_index", "y_true", "y_pred"]].copy()
    tabela["dataset_index"] = tabela["dataset_index"].astype(int)
    if tabela["dataset_index"].duplicated().any():
        duplicados = tabela.loc[
            tabela["dataset_index"].duplicated(keep=False),
            "dataset_index",
        ].tolist()[:10]
        raise AssertionError(
            f"Duplicated dataset indices in {contexto}: {duplicados}"
        )

    indices = np.asarray(indices_canonicos, dtype=int)
    faltantes = sorted(set(map(int, indices)) - set(tabela["dataset_index"]))
    extras = sorted(set(tabela["dataset_index"]) - set(map(int, indices)))
    if faltantes or extras:
        raise AssertionError(
            f"Index mismatch in {contexto}: missing={faltantes[:10]}, "
            f"extra={extras[:10]}."
        )

    tabela = (
        tabela.set_index("dataset_index")
        .loc[indices]
        .reset_index()
    )
    return tabela


def _predicoes_neural_phaseA_seed_fold(
    dataset: TropiSARDatasetV2,
    indices_validacao: Sequence[int],
    modelo_nome: str,
    ablation: str,
    seed: int,
    fold: int,
    device=DEVICE,
) -> pd.DataFrame:
    """Runs checkpoint-only inference in canonical validation-index order.

    This intentionally does not reuse ``predicoes_patch_final.npz`` because
    legacy files do not store dataset indices. Their arrays can therefore be
    correct but ordered differently after resumed runs. No training occurs.
    """
    indices = np.sort(np.asarray(indices_validacao, dtype=int))
    pasta = _diretorio_execucao_multiseed(
        modelo_nome,
        ablation,
        int(seed),
        int(fold),
    )

    nome_fase = (
        "fase_A_supervisionada"
        if modelo_nome in {"tcct", "tcct_sem_coatencao"}
        else "supervisionado"
    )
    checkpoint = pasta / nome_fase / "melhor_checkpoint.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Required supervised checkpoint not found: {checkpoint}"
        )

    payload = _torch_load_local(checkpoint, map_location="cpu")
    stats = StatsTreino(**payload["stats"])
    modelo = criar_modelo(modelo_nome, stats).to(device)
    modelo.load_state_dict(payload["model_state_dict"])
    modelo.eval()

    loader = DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=CFG.treino.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=PIN_MEMORY,
    )
    _, _, _, y_true, y_pred = avaliar_modelo(
        modelo,
        loader,
        stats,
        device,
        ablation=ablation,
        xai=False,
    )

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_canonico = _verdade_canonica_indices(dataset, indices)
    np.testing.assert_allclose(
        y_true,
        y_canonico,
        rtol=0,
        atol=1e-5,
        err_msg=(
            f"Fresh inference order mismatch for model={modelo_nome}, "
            f"ablation={ablation}, seed={seed}, fold={fold}."
        ),
    )

    del modelo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pd.DataFrame({
        "dataset_index": indices,
        "y_true": y_canonico,
        "y_pred": y_pred,
    })


def _predicoes_rf_seed_fold(
    dataset: TropiSARDatasetV2,
    indices_validacao: Sequence[int],
    seed: int,
    fold: int,
) -> pd.DataFrame:
    """Loads the trained RF and predicts in canonical dataset-index order."""
    import joblib

    indices = np.sort(np.asarray(indices_validacao, dtype=int))
    pasta = (
        CFG.dados.dir_resultados_v2
        / "multiseed"
        / f"seed_{int(seed)}"
        / f"fold_{int(fold)}"
        / "random_forest"
    )
    caminho_modelo = pasta / "random_forest.joblib"
    if not caminho_modelo.exists():
        raise FileNotFoundError(caminho_modelo)

    rf = joblib.load(caminho_modelo)
    x_val = []
    y_true = []
    for i in indices:
        amostra = dataset[int(i)]
        x_val.append(features_rf(amostra))
        chm = amostra["chm"].detach().cpu().numpy().astype(np.float64)
        mask = amostra["valid_mask"].detach().cpu().numpy().astype(bool)
        valores = chm[mask]
        if valores.size == 0:
            raise RuntimeError(f"Sample {int(i)} has no valid CHM pixel.")
        y_true.append(float(np.mean(valores)))

    y_pred = rf.predict(np.asarray(x_val, dtype=np.float32))
    return pd.DataFrame({
        "dataset_index": indices,
        "y_true": np.asarray(y_true, dtype=np.float64),
        "y_pred": np.asarray(y_pred, dtype=np.float64),
    })


def gerar_comparacoes_pareadas_oof_faseA(
    pasta_resultados: Optional[Path | str] = None,
    device=DEVICE,
) -> dict:
    """Creates strict Phase-A OOF pairwise figures without retraining.

    Predictions are regenerated from frozen checkpoints/models in a canonical
    ``dataset_index`` order. This avoids silently assigning current fold indices
    to legacy prediction arrays that did not save their original indices.
    """
    raiz = Path(pasta_resultados or CFG.dados.dir_resultados_v2)
    out = raiz / "figuras_comparacoes_oof_phaseA"
    cache = out / "cache_predicoes_v3_indexed"
    cache_execucoes = cache / "por_seed_fold"
    out.mkdir(parents=True, exist_ok=True)
    cache_execucoes.mkdir(parents=True, exist_ok=True)

    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
    folds = criar_folds_espaciais(dataset)
    seeds = tuple(map(int, CFG.treino.seeds_experimento))
    modelos = {
        "complete_tcct": ("tcct", "completo", "Complete TCCT"),
        "amplitude_only": ("tcct", "amplitude_only", "Amplitude-only TCCT"),
        "without_coattention": (
            "tcct_sem_coatencao",
            "completo",
            "TCCT without co-attention",
        ),
        "cnn": ("cnn", "completo", "Early-fusion CNN"),
        "vit": ("vit", "completo", "Vision Transformer"),
        "random_forest": (
            "random_forest",
            "completo",
            "Random Forest",
        ),
    }

    oof_por_modelo = {}
    for slug, (modelo_nome, ablation, rotulo) in modelos.items():
        cache_modelo = cache / f"oof_seed_mean_{slug}.csv"
        if cache_modelo.exists():
            oof_por_modelo[slug] = pd.read_csv(cache_modelo)
            print(f"Reusing indexed OOF cache: {cache_modelo}", flush=True)
            continue

        registros_folds = []
        for fold, (_, indices_validacao_originais) in enumerate(folds):
            indices_canonicos = np.sort(
                np.asarray(indices_validacao_originais, dtype=int)
            )
            y_ref = _verdade_canonica_indices(dataset, indices_canonicos)
            por_seed = []

            for seed in seeds:
                cache_execucao = (
                    cache_execucoes
                    / f"{slug}_seed_{int(seed)}_fold_{int(fold)}.csv"
                )
                if cache_execucao.exists():
                    pred = pd.read_csv(cache_execucao)
                else:
                    pred = (
                        _predicoes_rf_seed_fold(
                            dataset,
                            indices_canonicos,
                            seed,
                            fold,
                        )
                        if modelo_nome == "random_forest"
                        else _predicoes_neural_phaseA_seed_fold(
                            dataset,
                            indices_canonicos,
                            modelo_nome,
                            ablation,
                            seed,
                            fold,
                            device=device,
                        )
                    )
                    pred.to_csv(cache_execucao, index=False)

                pred = _alinhar_predicoes_por_indice(
                    pred,
                    indices_canonicos,
                    contexto=(
                        f"model={slug}, seed={seed}, fold={fold}"
                    ),
                )
                np.testing.assert_allclose(
                    pred["y_true"].to_numpy(dtype=np.float64),
                    y_ref,
                    rtol=0,
                    atol=1e-5,
                    err_msg=(
                        f"Target mismatch after index alignment for "
                        f"model={slug}, seed={seed}, fold={fold}."
                    ),
                )
                por_seed.append(
                    pred["y_pred"].to_numpy(dtype=np.float64)
                )

            registros_folds.append(pd.DataFrame({
                "dataset_index": indices_canonicos,
                "fold": int(fold),
                "y_true": y_ref,
                "y_pred": np.mean(
                    np.stack(por_seed, axis=0),
                    axis=0,
                ),
                "model": rotulo,
            }))

        oof = (
            pd.concat(registros_folds, ignore_index=True)
            .sort_values("dataset_index", kind="mergesort")
            .reset_index(drop=True)
        )
        if oof["dataset_index"].duplicated().any():
            raise AssertionError(
                f"Duplicated OOF sample indices for {rotulo}."
            )
        if len(oof) != len(dataset):
            raise AssertionError(
                f"OOF coverage for {rotulo} is {len(oof)}; "
                f"expected {len(dataset)}."
            )
        np.testing.assert_array_equal(
            oof["dataset_index"].to_numpy(dtype=int),
            np.arange(len(dataset), dtype=int),
        )
        oof.to_csv(cache_modelo, index=False)
        oof_por_modelo[slug] = oof

    referencia = oof_por_modelo["complete_tcct"]
    metricas_rows = []
    for slug, tabela in oof_por_modelo.items():
        np.testing.assert_array_equal(
            referencia["dataset_index"].to_numpy(dtype=int),
            tabela["dataset_index"].to_numpy(dtype=int),
        )
        np.testing.assert_allclose(
            referencia["y_true"].to_numpy(dtype=np.float64),
            tabela["y_true"].to_numpy(dtype=np.float64),
            rtol=0,
            atol=1e-5,
        )
        met = metricas_np(tabela["y_true"], tabela["y_pred"])
        metricas_rows.append({
            "slug": slug,
            "model": str(tabela["model"].iloc[0]),
            **met,
            "std_ratio": (
                float(met["std_pred"] / met["std_true"])
                if float(met["std_true"]) > 0
                else np.nan
            ),
            "n_oof": int(len(tabela)),
            "seed_aggregation": "mean within fold",
            "phase": "strict Phase A",
            "prediction_source": "fresh inference from frozen checkpoints",
            "index_alignment": "canonical dataset_index",
        })
    metricas_df = pd.DataFrame(metricas_rows)
    metricas_df.to_csv(out / "oof_phaseA_metrics.csv", index=False)

    pares = [
        "amplitude_only",
        "without_coattention",
        "cnn",
        "vit",
        "random_forest",
    ]
    caminhos = []
    for outro_slug in pares:
        outro = oof_por_modelo[outro_slug]
        nome_tcct = str(referencia["model"].iloc[0])
        nome_outro = str(outro["model"].iloc[0])
        met_tcct = metricas_df.loc[
            metricas_df["slug"] == "complete_tcct"
        ].iloc[0]
        met_outro = metricas_df.loc[
            metricas_df["slug"] == outro_slug
        ].iloc[0]

        fig, ax = plt.subplots(figsize=(8.4, 7.4))
        ax.scatter(
            referencia["y_true"],
            outro["y_pred"],
            s=13,
            alpha=0.20,
            color="#D55E00",
            linewidths=0,
            rasterized=True,
            label=(
                f"{nome_outro} (OOF $R^2$={met_outro['r2']:.3f}; "
                f"RMSE={met_outro['rmse']:.3f} m)"
            ),
        )
        ax.scatter(
            referencia["y_true"],
            referencia["y_pred"],
            s=13,
            alpha=0.20,
            color="#009E73",
            linewidths=0,
            rasterized=True,
            label=(
                f"{nome_tcct} (OOF $R^2$={met_tcct['r2']:.3f}; "
                f"RMSE={met_tcct['rmse']:.3f} m)"
            ),
        )
        minimo = float(np.nanmin([
            referencia["y_true"].min(),
            referencia["y_pred"].min(),
            outro["y_pred"].min(),
        ]))
        maximo = float(np.nanmax([
            referencia["y_true"].max(),
            referencia["y_pred"].max(),
            outro["y_pred"].max(),
        ]))
        ax.plot(
            [minimo, maximo],
            [minimo, maximo],
            "--",
            linewidth=1.8,
            color="black",
            label="1:1 line",
        )
        ax.set_title(
            f"{nome_tcct} vs {nome_outro}\n"
            "Strict Phase-A out-of-fold predictions"
        )
        ax.set_xlabel("Observed LiDAR-derived canopy height (m)")
        ax.set_ylabel("Predicted canopy height (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, framealpha=0.96, fontsize=9)
        fig.tight_layout()
        caminho = (
            out
            / f"complete_tcct_vs_{outro_slug}_oof_phaseA.png"
        )
        fig.savefig(caminho, dpi=300, bbox_inches="tight")
        plt.close(fig)
        caminhos.append(caminho)

    manifesto = pd.DataFrame({
        "figure": [str(p) for p in caminhos],
        "phase": "strict Phase A",
        "aggregation": "mean of three seeds within each spatial fold",
        "index_alignment": "canonical dataset_index",
    })
    manifesto.to_csv(
        out / "manifesto_figuras_pareadas.csv",
        index=False,
    )
    print(
        "\nStrict Phase-A pairwise OOF figures saved in:",
        out,
        flush=True,
    )
    print(metricas_df.to_string(index=False), flush=True)
    return {
        "metrics": metricas_df,
        "oof_predictions": oof_por_modelo,
        "figures": caminhos,
        "output_dir": out,
    }

def chm_para_agb_paracou(chm):
    return 162.17268*np.power(np.clip(chm,0,None),0.24862)


def metricas_agb_derivada(chm_true,chm_pred,limiar_agb=50):
    at=chm_para_agb_paracou(np.asarray(chm_true));ap=chm_para_agb_paracou(np.asarray(chm_pred));m=at>limiar_agb
    rmse=np.sqrt(np.mean((ap[m]-at[m])**2));rrmse=100*rmse/np.mean(at[m]);return {"rmse_agb":float(rmse),"rrmse_agb_pct":float(rrmse),"bias_agb":float(np.mean(ap[m]-at[m])),"n":int(m.sum())}


def exportar_mapa_30m(model,dataset,indices,stats,caminho_chm:Path,device=DEVICE,ablation="completo"):
    stack,qual,meta,_=carregar_landsat_reflectancia(CFG.dados);arr=np.full((meta["height"],meta["width"]),np.nan,np.float32)
    loader=DataLoader(Subset(dataset,list(map(int,indices))),batch_size=CFG.treino.batch_size,shuffle=False,num_workers=0);model.eval();cursor=0
    with torch.no_grad():
        for b in tqdm(loader,desc="Inferência mapa 30 m"):
            o,sl,sp,q,y,m,e=preparar_batch_v2(b,stats,device,ablation);pm,_,_=model(sl,sp,o,q);den=m.sum((1,2,3)).clamp_min(1);pred=((pm*m).sum((1,2,3))/den).cpu().numpy()
            for k,v in enumerate(pred):
                idx=int(indices[cursor+k]);row=dataset.index.iloc[idx];arr[int(row.landsat_row),int(row.landsat_col)]=v
            cursor+=len(pred)
    perfil={"driver":"GTiff","height":meta["height"],"width":meta["width"],"count":1,"dtype":"float32","crs":meta["crs"],"transform":meta["transform"],"nodata":-9999.0,"compress":"deflate"}
    caminho_chm.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(caminho_chm,"w",**perfil) as dst:dst.write(np.where(np.isfinite(arr),arr,-9999).astype(np.float32),1)
    agb=chm_para_agb_paracou(arr).astype(np.float32);caminho_agb=caminho_chm.with_name(caminho_chm.stem.replace("CHM","AGB")+caminho_chm.suffix)
    with rasterio.open(caminho_agb,"w",**perfil) as dst:dst.write(np.where(np.isfinite(agb),agb,-9999).astype(np.float32),1)
    return caminho_chm,caminho_agb



def exportar_mosaico_denso_chm(model,dataset,indices,stats,caminho_saida:Path,device=DEVICE,ablation="completo"):
    """Reconstrói a grade LiDAR/1 m com a predição 30×30 de cada célula Landsat."""
    ref_path=encontrar_chm(CFG.dados)
    with rasterio.open(ref_path) as ref:
        perfil=ref.profile.copy(); H,W=ref.height,ref.width
    soma=np.zeros((H,W),np.float32);cont=np.zeros((H,W),np.float32)
    loader=DataLoader(Subset(dataset,list(map(int,indices))),batch_size=CFG.treino.batch_size,shuffle=False,num_workers=0)
    model.eval()
    with torch.no_grad():
        for b in tqdm(loader,desc="Mosaico CHM denso"):
            o,sl,sp,q,y,m,e=preparar_batch_v2(b,stats,device,ablation);pm,_,_=model(sl,sp,o,q);pred=pm.cpu();wins=b["chm_window"].numpy()
            for z,win in zip(pred,wins):
                r0,c0,h,w=map(int,win);zz=F.interpolate(z[None],size=(h,w),mode="bilinear",align_corners=False)[0,0].numpy();soma[r0:r0+h,c0:c0+w]+=zz;cont[r0:r0+h,c0:c0+w]+=1
    arr=np.divide(soma,cont,out=np.full_like(soma,np.nan),where=cont>0);perfil.update(driver="GTiff",count=1,dtype="float32",nodata=-9999.0,compress="deflate")
    caminho_saida.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(caminho_saida,"w",**perfil) as dst:dst.write(np.where(np.isfinite(arr),arr,-9999).astype(np.float32),1)
    agb=chm_para_agb_paracou(arr);agb_path=caminho_saida.with_name(caminho_saida.stem.replace("CHM","AGB")+caminho_saida.suffix)
    with rasterio.open(agb_path,"w",**perfil) as dst:dst.write(np.where(np.isfinite(agb),agb,-9999).astype(np.float32),1)
    return caminho_saida,agb_path



# ================================================================
# 9. POST-PROCESSING FOR THE REVISION — ENGLISH FIGURES AND SAFE DIAGNOSTICS
# ================================================================
def regenerar_diagnosticos_chm_ingles(
    cfg: ConfigDados = CFG.dados,
) -> list[Path]:
    """Regenerates CHM diagnostic figures in English without preprocessing."""
    caminho_chm = cfg.caminho_chm_canonico
    if not caminho_chm.exists():
        raise FileNotFoundError(caminho_chm)

    with rasterio.open(caminho_chm) as src:
        chm = src.read(1, masked=True).filled(np.nan).astype(np.float32)

    passo = max(1, int(max(chm.shape) / 1200))
    mapa_dec = chm[::passo, ::passo]
    valores = mapa_dec[np.isfinite(mapa_dec)]
    if not valores.size:
        raise RuntimeError("The canonical CHM contains no finite values.")

    out = cfg.dir_lidar_processado
    vmin, vmax = np.percentile(valores, [1, 99])
    caminhos = []

    fig, ax = plt.subplots(figsize=(10, 7))
    imagem = ax.imshow(mapa_dec, vmin=vmin, vmax=vmax)
    fig.colorbar(imagem, ax=ax, label="CHM (m)")
    ax.set_title("Reconstructed CHM: DSM − DTM")
    ax.set_xlabel("Decimated column")
    ax.set_ylabel("Decimated row")
    fig.tight_layout()
    caminho = out / "mapa_diagnostico_chm.png"
    fig.savefig(caminho, dpi=220, bbox_inches="tight")
    plt.close(fig)
    caminhos.append(caminho)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valores, bins=150)
    ax.set_xlabel("CHM (m)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of the reconstructed CHM")
    fig.tight_layout()
    caminho = out / "histograma_chm.png"
    fig.savefig(caminho, dpi=220, bbox_inches="tight")
    plt.close(fig)
    caminhos.append(caminho)
    return caminhos


def regenerar_figura_landsat_ingles(
    cfg: ConfigDados = CFG.dados,
) -> Path:
    """Regenerates the Landsat source-selection map from the existing cache."""
    mosaico = preparar_mosaico_landsat_qualidade(cfg, sobrescrever=False)
    source_index = np.asarray(mosaico["source_index"])
    out = cfg.dir_landsat_processado

    fig, ax = plt.subplots(figsize=(10, 7))
    imagem = ax.imshow(
        source_index,
        vmin=-1,
        vmax=max(1, len(mosaico.get("scenes", [])) - 1),
    )
    fig.colorbar(imagem, ax=ax, label="Selected scene index")
    ax.set_title("Landsat-5 source selected for each 30 m cell")
    ax.set_xlabel("Landsat column")
    ax.set_ylabel("Landsat row")
    fig.tight_layout()
    caminho = out / "mapa_fonte_cena_landsat.png"
    fig.savefig(caminho, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return caminho


def diagnosticar_finetuning_pre_pos(
    device=DEVICE,
    pasta_protocolo: Optional[Path | str] = None,
) -> dict:
    """Compares frozen pre/post fine-tuning checkpoints on identical partitions.

    No training, checkpoint modification, or hyperparameter selection is
    performed. Model selection is based exclusively on local-validation RMSE;
    the final-test metrics are only reported after that rule is applied.
    """
    pasta_protocolo = Path(
        pasta_protocolo
        or (CFG.dados.dir_resultados_v2 / "protocolo_finetuning")
    )
    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)

    def carregar_indices(nome_particao: str) -> np.ndarray:
        caminho = pasta_protocolo / f"manifesto_{nome_particao}.csv"
        if not caminho.exists():
            raise FileNotFoundError(caminho)
        tabela = pd.read_csv(caminho)
        if "dataset_index" not in tabela.columns:
            raise KeyError(f"dataset_index is absent from {caminho}")
        return tabela["dataset_index"].to_numpy(dtype=int)

    idx_ajuste = carregar_indices("local_ajuste")
    idx_validacao = carregar_indices("local_validacao")
    idx_teste = carregar_indices("teste_final")

    def criar_loader(indices: Sequence[int]) -> DataLoader:
        return DataLoader(
            Subset(dataset, list(map(int, indices))),
            batch_size=CFG.treino.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=PIN_MEMORY,
        )

    loader_ajuste = criar_loader(idx_ajuste)
    loader_validacao = criar_loader(idx_validacao)
    loader_teste = criar_loader(idx_teste)

    checkpoint_global = (
        pasta_protocolo
        / "modelo_global"
        / "tcct"
        / "completo"
        / "melhor_modelo_final.pth"
    )
    checkpoint_finetune = (
        pasta_protocolo
        / "fine_tuning_local"
        / "melhor_modelo_finetune.pth"
    )

    def carregar_modelo(caminho_checkpoint: Path):
        if not caminho_checkpoint.exists():
            raise FileNotFoundError(caminho_checkpoint)
        checkpoint = torch.load(
            caminho_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        stats = StatsTreino(**checkpoint["stats"])
        modelo = TCCTv2(CFG.modelo, stats).to(device)
        modelo.load_state_dict(checkpoint["model_state_dict"])
        modelo.eval()
        return modelo, stats

    modelo_global, stats_global = carregar_modelo(checkpoint_global)
    modelo_finetune, stats_finetune = carregar_modelo(checkpoint_finetune)

    def avaliar(modelo, stats, loader):
        metricas_patch, metricas_pixel, _, y_true, y_pred = avaliar_modelo(
            modelo,
            loader,
            stats,
            device,
            ablation="completo",
            xai=False,
        )
        return (
            metricas_patch,
            metricas_pixel,
            np.asarray(y_true),
            np.asarray(y_pred),
        )

    met_global_teste, _, y_teste, pred_global_teste = avaliar(
        modelo_global, stats_global, loader_teste
    )
    met_global_val, _, y_val, pred_global_val = avaliar(
        modelo_global, stats_global, loader_validacao
    )
    met_finetune_teste, _, y_teste_ft, pred_finetune_teste = avaliar(
        modelo_finetune, stats_finetune, loader_teste
    )
    met_finetune_val, _, y_val_ft, pred_finetune_val = avaliar(
        modelo_finetune, stats_finetune, loader_validacao
    )

    np.testing.assert_allclose(y_teste, y_teste_ft, rtol=0, atol=1e-6)
    np.testing.assert_allclose(y_val, y_val_ft, rtol=0, atol=1e-6)

    _, _, y_ajuste, pred_global_ajuste = avaliar(
        modelo_global, stats_global, loader_ajuste
    )
    deslocamento_local = float(np.mean(y_ajuste - pred_global_ajuste))
    pred_global_deslocada = pred_global_teste + deslocamento_local
    met_global_deslocado = metricas_np(y_teste, pred_global_deslocada)

    pred_media_global = np.full_like(y_teste, stats_global.target_mean)
    pred_media_local = np.full_like(y_teste, np.mean(y_ajuste))
    met_media_global = metricas_np(y_teste, pred_media_global)
    met_media_local = metricas_np(y_teste, pred_media_local)

    def preparar_linha(nome: str, metricas: dict) -> dict:
        linha = {"Model": nome, **metricas}
        std_true = float(metricas.get("std_true", np.nan))
        std_pred = float(metricas.get("std_pred", np.nan))
        linha["std_ratio"] = (
            std_pred / std_true
            if np.isfinite(std_true) and std_true > 0
            else np.nan
        )
        return linha

    comparacao_teste = pd.DataFrame([
        preparar_linha(
            "Global checkpoint before fine-tuning", met_global_teste
        ),
        preparar_linha(
            "Global checkpoint + local offset", met_global_deslocado
        ),
        preparar_linha(
            "Model after local fine-tuning", met_finetune_teste
        ),
        preparar_linha(
            "Constant global-training mean", met_media_global
        ),
        preparar_linha(
            "Constant local-adaptation mean", met_media_local
        ),
    ])
    comparacao_validacao = pd.DataFrame([
        preparar_linha(
            "Global checkpoint before fine-tuning", met_global_val
        ),
        preparar_linha(
            "Model after local fine-tuning", met_finetune_val
        ),
    ])

    selecionar_finetune = (
        float(met_finetune_val["rmse"]) < float(met_global_val["rmse"])
    )
    modelo_selecionado = (
        "Model after local fine-tuning"
        if selecionar_finetune
        else "Global checkpoint before fine-tuning"
    )
    metricas_teste_selecionado = (
        met_finetune_teste if selecionar_finetune else met_global_teste
    )

    pasta_saida = pasta_protocolo / "diagnostico_pre_pos"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    comparacao_teste.to_csv(
        pasta_saida / "comparacao_teste_final_pre_pos.csv", index=False
    )
    comparacao_validacao.to_csv(
        pasta_saida / "comparacao_validacao_local_pre_pos.csv", index=False
    )
    np.savez(
        pasta_saida / "predicoes_teste_pre_pos.npz",
        y_true=y_teste,
        pred_global=pred_global_teste,
        pred_global_offset=pred_global_deslocada,
        pred_finetune=pred_finetune_teste,
    )

    decisao = {
        "selection_rule": "lowest local-validation RMSE, including epoch 0",
        "test_used_for_selection": False,
        "selected_model": modelo_selecionado,
        "global_local_validation_rmse": float(met_global_val["rmse"]),
        "fine_tuned_local_validation_rmse": float(met_finetune_val["rmse"]),
        "selected_final_test_metrics": metricas_teste_selecionado,
        "local_offset_estimated_from_adaptation_set_m": deslocamento_local,
    }
    (pasta_saida / "decisao_selecao_validacao_local.json").write_text(
        json.dumps(decisao, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # English diagnostic scatter plot. It is descriptive; selection remains
    # based only on local validation.
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        y_teste,
        pred_global_teste,
        s=13,
        alpha=0.18,
        rasterized=True,
        label="Global checkpoint",
    )
    ax.scatter(
        y_teste,
        pred_finetune_teste,
        s=13,
        alpha=0.18,
        rasterized=True,
        label="After local fine-tuning",
    )
    limite_min = float(
        np.nanmin([y_teste.min(), pred_global_teste.min(), pred_finetune_teste.min()])
    )
    limite_max = float(
        np.nanmax([y_teste.max(), pred_global_teste.max(), pred_finetune_teste.max()])
    )
    ax.plot(
        [limite_min, limite_max],
        [limite_min, limite_max],
        linestyle="--",
        linewidth=1.2,
        label="1:1 line",
    )
    ax.set_xlabel("Observed LiDAR-derived canopy height (m)")
    ax.set_ylabel("Predicted canopy height (m)")
    ax.set_title("Safe Spatial Fine-Tuning Diagnostic — Final Test")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        pasta_saida / "fine_tuning_pre_post_final_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nCOMPARISON ON THE SAME FINAL TEST")
    print(
        comparacao_teste.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.6f}",
        )
    )
    print("\nCOMPARISON ON THE SAME LOCAL VALIDATION SET")
    print(
        comparacao_validacao.to_string(
            index=False,
            float_format=lambda valor: f"{valor:.6f}",
        )
    )
    print("\nVALIDATION-ONLY MODEL-SELECTION DECISION")
    print(json.dumps(decisao, ensure_ascii=False, indent=2))

    return {
        "test_comparison": comparacao_teste,
        "validation_comparison": comparacao_validacao,
        "selection": decisao,
        "output_dir": pasta_saida,
    }


def regenerar_figuras_revisao_ingles(
    pasta_resultados: Optional[Path | str] = None,
    incluir_chm_landsat: bool = True,
) -> list[Path]:
    """Regenerates all available figures in English without retraining models."""
    raiz = Path(pasta_resultados or CFG.dados.dir_resultados_v2)
    caminhos_gerados: list[Path] = []

    # Training-history figures from saved CSV files.
    for historico_path in sorted(raiz.rglob("historico.csv")):
        try:
            historico = pd.read_csv(historico_path)
            destino = historico_path.parent / "metricas.png"
            plot_historico(historico, destino)
            caminhos_gerados.append(destino)
        except Exception as exc:
            warnings.warn(
                f"Could not regenerate {historico_path}: {exc}"
            )

    dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)

    # Spatial-fold maps; no model is trained.
    pasta_auditoria = raiz / "auditoria_espacial"
    pasta_auditoria.mkdir(parents=True, exist_ok=True)
    for buffer in CFG.dados.buffers_a_auditar:
        cfg_tmp = copy.deepcopy(CFG.dados)
        cfg_tmp.buffer_blocos = int(buffer)
        folds = criar_folds_espaciais(dataset, cfg_tmp, CFG.treino)
        destino = pasta_auditoria / f"mapa_folds_buffer_{buffer}.png"
        plotar_mapa_folds(
            dataset,
            folds,
            destino,
            f"Spatial folds — buffer of {buffer} block(s)",
        )
        caminhos_gerados.append(destino)

    # Recreate existing spatial protocol maps without training.
    if (raiz / "holdout_espacial_70_20_10").exists():
        protocolo_holdout = criar_holdout_espacial_70_20_10(dataset)
        destino = Path(protocolo_holdout["pasta"]) / "mapa_holdout_70_20_10.png"
        caminhos_gerados.append(destino)

    if (raiz / "protocolo_finetuning").exists():
        protocolo_ft = criar_protocolo_finetuning_espacial(dataset)
        destino = Path(protocolo_ft["pasta"]) / "mapa_protocolo_finetuning.png"
        caminhos_gerados.append(destino)

    # Modality-gating/XAI point clouds from saved fold/seed CSV files.
    try:
        consolidar_gating_modalidade(raiz)
        pasta_xai = raiz / "xai_revisao"
        caminhos_gerados.extend(sorted(pasta_xai.glob("gating_weights*.png")))
        caminho_tcct = pasta_xai / "xai_modality_gating_cloud_complete_tcct.png"
        if caminho_tcct.exists():
            caminhos_gerados.append(caminho_tcct)
    except FileNotFoundError as exc:
        warnings.warn(str(exc))

    if incluir_chm_landsat:
        try:
            caminhos_gerados.extend(regenerar_diagnosticos_chm_ingles())
        except Exception as exc:
            warnings.warn(f"CHM diagnostic figures were not regenerated: {exc}")
        try:
            caminhos_gerados.append(regenerar_figura_landsat_ingles())
        except Exception as exc:
            warnings.warn(f"Landsat source figure was not regenerated: {exc}")

    # Deduplicate while preserving order.
    unicos = []
    vistos = set()
    for caminho in caminhos_gerados:
        caminho = Path(caminho)
        chave = str(caminho.resolve(strict=False))
        if chave not in vistos and caminho.exists():
            vistos.add(chave)
            unicos.append(caminho)

    manifesto = pd.DataFrame({"figure": [str(p) for p in unicos]})
    pasta_manifesto = raiz / "figuras_revisao_ingles"
    pasta_manifesto.mkdir(parents=True, exist_ok=True)
    manifesto.to_csv(
        pasta_manifesto / "manifesto_figuras_regeneradas.csv",
        index=False,
    )

    print(f"\nRegenerated {len(unicos)} English-language figure(s):")
    for caminho in unicos:
        print(" -", caminho)
    return unicos


# ================================================================
# INTERFACE DE LINHA DE COMANDO — SEGURA PARA JUPYTER
# ================================================================
def executando_em_notebook() -> bool:
    """Retorna True em kernels Jupyter/IPython.

    O ipykernel injeta argumentos como `-f kernel-....json` em sys.argv.
    Por isso o CLI nunca é iniciado automaticamente dentro do notebook.
    """
    return "ipykernel" in sys.modules or "IPython" in sys.modules


def main_cli(argv: Optional[Sequence[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="TCCT v3.2.1 — portátil, retomável e científico por grades ONERA .grille 3D",
        allow_abbrev=False,
    )
    parser.add_argument(
        "etapa",
        choices=[
            "status",
            "preparar-raiz",
            "inventario-completo",
            "auditoria",
            "preparar-chm",
            "preparar-landsat",
            "limpar-obsoletos",
            "decidir-datum-dtm",
            "auditar-layouts",
            "backgeo",
            "auditar-backgeo",
            "geometria-sar-comum",
            "coerencia",
            "geometria-rvog",
            "dataset",
            "auditar-dataset",
            "treino",
            "5fold",
            "5fold-multiseed",
            "status-multiseed",
            "holdout-70-20-10",
            "auditoria-espacial",
            "smoke-test",
            "finetuning-seguro",
            "tabela-revisao",
            "xai-comparativa",
            "figuras-ingles",
            "diagnostico-finetuning",
            "comparacoes-oof-phaseA",
            "rf",
            "rf-multiseed",
        ],
        help="Etapa a executar",
    )
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--modelo", choices=["tcct", "tcct_sem_coatencao", "cnn", "vit"], default="tcct")
    parser.add_argument("--ablation", default="completo")
    parser.add_argument("--seeds", default="42,314,2026")
    parser.add_argument("--confirmar-limpeza", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.base_dir:
        CFG.dados.base_dir = args.base_dir

    # Toda execução de terminal valida/cria a raiz isolada antes de continuar.
    preparar_estrutura_projeto(CFG.dados, criar=True)

    if args.etapa == "status":
        print(
            f"Pipeline={VERSAO_PIPELINE} Python={sys.executable} "
            f"PyTorch={torch.__version__} Device={DEVICE} "
            f"Base={CFG.dados.raiz}",
            flush=True,
        )
    elif args.etapa == "preparar-raiz":
        imprimir_estrutura_projeto(CFG.dados)
    elif args.etapa == "inventario-completo":
        inventariar_dados_brutos_completo(CFG.dados)
    elif args.etapa == "auditoria":
        executar_etapa("auditoria", auditar_ambiente, CFG.dados, True)
    elif args.etapa == "preparar-chm":
        preparar_chm_lidar_reproduzivel(CFG.dados)
    elif args.etapa == "preparar-landsat":
        preparar_mosaico_landsat_qualidade(CFG.dados)
    elif args.etapa == "limpar-obsoletos":
        remover_produtos_obsoletos_pre_grille(
            CFG.dados, confirmar=args.confirmar_limpeza
        )
    elif args.etapa == "decidir-datum-dtm":
        preparar_dtm_para_grille(CFG.dados)
    elif args.etapa == "auditar-layouts":
        auditar_layouts_slc(CFG.dados)
        auditar_layout_ha_p(CFG.dados)
    elif args.etapa == "backgeo":
        executar_etapa(
            "back-geocoding .grille 3D",
            executar_ortorretificacao_global,
            CFG.dados,
        )
    elif args.etapa == "auditar-backgeo":
        executar_etapa(
            "auditoria do back-geocoding .grille 3D",
            auditar_saidas_backgeocoding,
            CFG.dados,
        )
    elif args.etapa == "geometria-sar-comum":
        preparar_geometria_sar_comum(CFG.dados)
    elif args.etapa == "coerencia":
        preparar_mapas_coerencia(CFG.dados)
    elif args.etapa == "geometria-rvog":
        dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
        folds = criar_folds_espaciais(dataset)
        preparar_geometria_rvog(
            dataset, folds[0][0], folds[0][1],
            CFG.dados.dir_resultados_v2 / "preflight_fold_0",
        )
    elif args.etapa == "dataset":
        executar_etapa(
            "geração do dataset v6 .grille 3D",
            executar_geracao_dataset_v2,
            CFG.dados,
        )
    elif args.etapa == "auditar-dataset":
        dataset = executar_etapa(
            "abertura do dataset",
            TropiSARDatasetV2,
            CFG.dados.dir_dataset_v2,
        )
        executar_etapa("auditoria do dataset", auditar_dataset_v2, dataset, 1500)
    elif args.etapa == "treino":
        executar_etapa(
            f"treino em duas fases — fold {args.fold}",
            treinar_holdout,
            args.modelo,
            args.ablation,
            args.fold,
        )
    elif args.etapa == "5fold":
        executar_etapa(
            "validação 5-fold",
            executar_5fold,
            args.modelo,
            args.ablation,
        )
    elif args.etapa == "5fold-multiseed":
        seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
        executar_etapa(
            "5-fold multiseed retomável",
            executar_5fold_multiseed,
            args.modelo,
            args.ablation,
            seeds,
        )
    elif args.etapa == "status-multiseed":
        seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
        status_5fold_multiseed(args.modelo, args.ablation, seeds)
    elif args.etapa == "holdout-70-20-10":
        executar_etapa(
            "holdout espacial 70/20/10",
            executar_holdout_espacial_70_20_10,
            int(args.seeds.split(",")[0].strip()) if args.seeds else None,
            True,
        )
    elif args.etapa == "auditoria-espacial":
        dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
        executar_etapa("auditoria espacial", auditar_sensibilidade_buffer, dataset)
    elif args.etapa == "smoke-test":
        executar_etapa("smoke test da revisão", executar_smoke_test_revisao, args.fold)
    elif args.etapa == "finetuning-seguro":
        executar_etapa("fine-tuning espacial seguro", executar_finetuning_espacial_seguro)
    elif args.etapa == "tabela-revisao":
        executar_etapa("consolidação da Table I", consolidar_tabela_revisao)
    elif args.etapa == "xai-comparativa":
        executar_etapa("comparative modality gating", consolidar_gating_modalidade)
    elif args.etapa == "figuras-ingles":
        executar_etapa(
            "regeneration of English-language figures",
            regenerar_figuras_revisao_ingles,
        )
    elif args.etapa == "diagnostico-finetuning":
        executar_etapa(
            "safe pre/post fine-tuning diagnostic",
            diagnosticar_finetuning_pre_pos,
        )
    elif args.etapa == "comparacoes-oof-phaseA":
        executar_etapa(
            "strict Phase-A OOF pairwise comparisons",
            gerar_comparacoes_pareadas_oof_faseA,
        )
    elif args.etapa == "rf-multiseed":
        seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
        executar_etapa("RF 5-fold multiseed", executar_rf_5fold_multiseed, seeds)
    elif args.etapa == "rf":
        dataset = TropiSARDatasetV2(CFG.dados.dir_dataset_v2)
        folds = criar_folds_espaciais(dataset)
        salvar_manifestos_folds(
            dataset,
            folds,
            CFG.dados.dir_resultados_v2 / "manifestos_folds",
        )
        train_idx, val_idx = folds[args.fold]
        executar_etapa(
            f"Random Forest — fold {args.fold}",
            baseline_random_forest,
            dataset,
            train_idx,
            val_idx,
            400,
            CFG.dados.dir_resultados_v2 / f"fold_{args.fold}" / "random_forest",
        )


# Execução automática apenas em terminal. Dentro do notebook, esta célula
# somente define funções e termina sem tentar interpretar os argumentos do kernel.
if __name__ == "__main__" and not executando_em_notebook():
    main_cli()
