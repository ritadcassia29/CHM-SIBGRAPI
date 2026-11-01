[README_Coleta_Reprodutibilidade_TCCT.md](https://github.com/user-attachments/files/34246141/README_Coleta_Reprodutibilidade_TCCT.md)
# Coleta de artefatos de reprodutibilidade do TCCT

O script `Preparar_Pacote_Reprodutibilidade_TCCT.py` faz a busca e a auditoria
dos artefatos já existentes no projeto. Ele não refaz treinamentos.

## Execução recomendada no Windows

Coloque o `.py` e o `.bat` em `<PROJECT_ROOT>\Codigo` e execute:

```bat
Executar_Coleta_Reprodutibilidade_TCCT.bat
```

O modo padrão do `.bat` é `evidence`: copia todos os logs, CSVs, JSONs, NPZs,
manifests, tabelas, figuras, código e captura do ambiente, mas apenas inventaria
os checkpoints e modelos.

Para gerar o pacote incluindo pesos e modelos:

```bat
<LOCAL_PATH> Preparar_Pacote_Reprodutibilidade_TCCT.py ^
  --project-root "<PROJECT_ROOT>" ^
  --copy-mode full ^
  --max-file-mb 4096 ^
  --zip ^
  --strict
```

## Arquivo decisivo

Abra:

```text
Auditoria\Reprodutibilidade_TCCT_<data_hora>\ausencias_criticas.csv
```

Ele deve ficar vazio. O arquivo `resumo_auditoria.json` informa a quantidade de
combinações completas. O desenho esperado é:

- 75 execuções neurais: 5 modelos/configurações × 5 folds × 3 seeds;
- 15 execuções Random Forest: 5 folds × 3 seeds;
- 90 combinações no total.

## Observação sobre os pesos

Para o repositório público, não é obrigatório duplicar todos os checkpoints se
o tamanho inviabilizar o envio. O inventário SHA-256 pode documentar todos os
pesos localmente, enquanto o repositório publica:

- código e ambiente;
- manifests e hashes;
- logs e tabelas por seed/fold;
- predições OOF;
- checkpoints finais permitidos ou uma seleção claramente documentada.

Não publique dados brutos sujeitos às condições originais de distribuição.
