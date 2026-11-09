# TCCT v3.2.1 — instalação e retomada

## Arquivos principais

Copie para `SIBGRAPI_REV/Codigo/`:

- `Pipeline_TCCT_v3_2_1_SIBGRAPI_REV_GRILLE3D_RETOMAVEL.py`
- `Pipeline_TCCT_v3_2_1_SIBGRAPI_REV_GRILLE3D_PORTATIL_RETOMAVEL.ipynb`
- `GUIA_CIENTIFICO_TCCT_v3_2_1.ipynb`

O novo script usa a mesma pasta de resultados:

`Resultados/TCCT_v3_2_SIBGRAPI`

Assim, combinações do Passo 20 já concluídas são reutilizadas.

## Retomar o Passo 20 em outro computador

1. Abra somente o notebook v3.2.1 portátil/retomável.
2. Execute os Passos 00 e 01.
3. Vá ao Passo 20. A célula imprime o status das 15 combinações.
4. Altere `EXECUTAR_PASSO_20 = True`.
5. Execute a célula.

Não repita limpeza, back-geocoding, dataset, Passos 03–19 ou Fold 0.

## O que acontece com a execução atual da versão anterior

- combinações com checkpoint final e resumo válidos serão puladas;
- a combinação interrompida pela versão 3.2.0 pode reiniciar uma única vez;
- depois que uma combinação começar na v3.2.1, a retomada passa a ocorrer por época;
- se a interrupção ocorrer no meio de uma época, apenas essa época é refeita.

## Como interromper

É preferível interromper logo depois de `🏆 MODELO FINAL`, mas a v3.2.1 permite interrupção durante uma fase. Use o botão de interrupção do kernel antes de desligar o computador.

## Holdout 70/20/10

O Passo 29 é adicional e deve ser executado somente após congelar os hiperparâmetros e concluir os experimentos principais.

Ele usa aproximadamente:

- 70% treino;
- 20% validação;
- 10% teste final intocado.

A divisão é feita por faixas de blocos espaciais, não por patches aleatórios. A Fase A permite até 150 épocas com paciência 20; a Fase B, até 50 épocas com paciência 15.

Esse experimento não substitui o Passo 20, que fornece a evidência principal de generalização espacial e variabilidade entre seeds.

## Arquivos que não devem ser apagados

Não apague:

- `Resultados/TCCT_v3_2_SIBGRAPI/multiseed`;
- `Dataset/TCCT_dataset_v6_grille3d`;
- `Dados_Processados/BackGeocoding_Grille3D`;
- `Dados_Processados/Geometria_RVoG_Grille3D`;
- `Auditoria`;
- `Dados_Brutos`.

Os notebooks e scripts v3.2.0 podem ser movidos para uma pasta de histórico, mas não precisam ser excluídos.
