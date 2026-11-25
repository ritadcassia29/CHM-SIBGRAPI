# Guia de leitura do projeto CHM-SIBGRAPI

Este documento explica, em linguagem introdutória, o que o projeto faz, como os dados viram previsões e para que servem os principais scripts e pastas. Ele foi escrito a partir do código, dos READMEs, da configuração do pipeline e dos metadados do conjunto de dados que estão neste repositório.

> **Em uma frase:** o projeto combina imagens de radar e imagens ópticas de satélite para estimar a altura da vegetação em uma floresta da Guiana Francesa, usando medições LiDAR como referência para ensinar e avaliar os modelos.

## 1. O problema que o projeto tenta resolver

Uma floresta tem propriedades que não são fáceis de medir em grandes áreas. Aqui, o alvo principal é o **CHM** (*Canopy Height Model*): um mapa de altura do topo da vegetação, em metros. O CHM de referência é construído a partir de dados LiDAR: em termos simples, subtrai-se o modelo do terreno (DTM) do modelo da superfície (DSM). O LiDAR fornece a resposta conhecida durante o treinamento; o modelo tenta aprender a estimá-la usando sensores que cobrem áreas maiores.

O projeto também transforma altura em **AGB** (*Above-Ground Biomass*, biomassa acima do solo). Essa conversão usa uma relação alométrica, isto é, uma equação empírica que relaciona altura da vegetação a biomassa. AGB é, portanto, uma estimativa derivada do CHM e da equação adotada; não é uma medição direta feita pela rede neural.

## 2. Sensoriamento remoto: os dados de entrada

### Radar SAR

**SAR** significa radar de abertura sintética. O sensor envia micro-ondas e registra o eco que retorna. Diferentemente de uma câmera comum, o radar mede uma resposta eletromagnética que depende da estrutura, umidade, orientação e composição dos alvos. Ele pode observar através de nuvens, uma vantagem importante em regiões tropicais.

O projeto usa dados **PolInSAR** em duas bandas, **L** e **P**, para duas aquisições (05/06 e 06/03, identificadas no código como `tropi0506` e `tropi0603`). Quatro polarizações aparecem como `HH`, `HV`, `VH` e `VV`: as letras indicam a orientação da onda transmitida e recebida. O processamento forma uma **coerência complexa normalizada** para cada polarização. Pense nela como uma medida de semelhança entre as respostas do radar nas duas aquisições; por ser complexa, ela contém parte real e imaginária, que carregam informação de amplitude e fase.

O radar não vira simplesmente quatro números. O modelo recebe pequenas matrizes espaciais (*patches*), preservando padrões ao redor de cada local. A geometria dessas imagens é especialmente importante: o **back-geocoding** usa grades de localização chamadas `.grille` para trazer as imagens SAR para uma grade comum com as demais fontes. A política atual exige interseção geométrica válida das quatro grades, evitando usar pixels extrapolados ou fora das imagens.

### Landsat (óptico)

Landsat é um satélite que mede luz refletida em diferentes faixas do espectro. O conjunto usa seis bandas de reflectância e dois índices calculados a partir delas: **NDVI** e **EVI**, indicadores relacionados à vegetação. Nuvens, saturação e pixels sem dados podem invalidar uma observação, então o pipeline usa as máscaras de qualidade (`QA_PIXEL` e, quando disponível, `QA_RADSAT`).

O mosaico atual seleciona, para cada célula de 30 m, uma cena considerada válida e de melhor qualidade. As bandas de uma célula vêm da mesma cena, para não misturar datas dentro daquele pixel. O mosaico continua sendo **multitemporal**: diferentes células podem vir de datas diferentes. O arquivo `metadata.json` registra as cenas e a data selecionada. Essas imagens não são necessariamente simultâneas às aquisições TropiSAR.

### LiDAR e CHM

LiDAR mede distâncias usando pulsos de laser e permite construir superfícies de terreno e de copa com bastante detalhe. O projeto valida sistema de coordenadas, resolução, valores inválidos e alturas extremas antes de usar esses dados como referência. O CHM é o **alvo** que o modelo tenta prever; não deve ser confundido com uma entrada óptica ou de radar.

## 3. O que é uma rede neural, sem pressupor conhecimento prévio

Uma rede neural é uma função com muitos parâmetros ajustáveis. Ela recebe números (aqui, matrizes de bandas e canais de sensores), produz uma previsão e compara essa previsão com o valor de referência. No treinamento, um algoritmo ajusta os parâmetros para diminuir os erros. Repetir esse processo em muitos exemplos permite que o modelo aprenda relações úteis nos dados.

Uma **época** é uma passagem pelo conjunto de treinamento. Um **batch** é um pequeno grupo de exemplos processado de uma vez. A **loss** (função de perda) resume o erro que o treinamento tenta reduzir. O conjunto de **treino** ajusta os parâmetros; o de **validação** ajuda a escolher quando parar e qual versão do modelo guardar. Ele não deve ser confundido com um teste final independente.

O projeto padroniza as entradas usando estatísticas do conjunto de treino. Isso coloca canais com escalas diferentes em faixas mais comparáveis. A padronização do alvo CHM também é usada internamente, mas as previsões são reconvertidas para metros.

### Arquitetura principal: TCCT

O modelo principal é identificado como **TCCT**. É uma arquitetura de fusão: cada tipo de sensor tem um caminho de processamento, e depois as informações são combinadas para prever o CHM.

- **CVNN / convoluções complexas:** o caminho do radar trata números complexos (partes real e imaginária) em vez de descartar diretamente a fase. Convoluções procuram padrões locais, como formas e texturas, em cada patch.
- **Encoder óptico:** extrai padrões das bandas Landsat e dos índices de vegetação.
- **Coatenção trimodal:** mecanismo que permite aos caminhos de radar L, radar P e óptico ponderar informações uns dos outros. A intenção é aprender quais fontes ajudam mais em cada exemplo.
- **Cabeça global/local:** combina um resumo do patch com detalhes espaciais para produzir um mapa CHM previsto dentro do patch.
- **Gating modal:** pesos/diagnósticos associados à contribuição das modalidades são exportados para inspeção. Eles ajudam a entender o comportamento do modelo, mas não provam causalidade nem explicam sozinhos a decisão.

O código contém variantes para comparação: TCCT sem coatenção, uma CNN com fusão antecipada e um ViT denso (*Vision Transformer*). Também existe uma baseline **Random Forest**, que é um conjunto de árvores de decisão e não uma rede neural.

### Função de perda e duas fases

A loss do TCCT combina mais de um critério. Ela considera erro nos pixels de CHM, erro no resumo por patch, diferenças de gradiente/estrutura e concordância estatística entre valores previstos e observados. Em áreas de vegetação alta, o código pode dar peso adicional aos exemplos acima do limiar configurado. Isso tenta equilibrar diferentes aspectos do problema, mas não significa que o modelo esteja correto só porque a loss caiu.

O treinamento do TCCT pode ter duas fases:

1. **Fase A supervisionada:** aprende a prever o CHM usando as referências LiDAR e a loss de imagem.
2. **Fase B com termo RVoG:** continua o ajuste acrescentando uma restrição inspirada na física do radar, quando há dados geométricos válidos.

**RVoG** (*Random Volume over Ground*) é um modelo físico simplificado para descrever como o sinal de radar interage com vegetação e solo. Nesta implementação, geometria como `kz` (número de onda vertical) e ângulo de incidência ajuda a calcular um termo de consistência. Esse termo é adicionado gradualmente na Fase B. A ideia é combinar aprendizado a partir de exemplos com uma restrição física plausível; não é uma medição direta da altura.

## 4. Fluxo do pipeline principal

A ordem conceitual do arquivo `Pipeline_TCCT...RETOMAVEL.py` é:

1. **Configuração e ambiente:** escolhe diretórios, bibliotecas, dispositivo disponível (GPU CUDA, Apple MPS ou CPU), parâmetros, sementes e caminhos do projeto.
2. **Auditoria dos arquivos:** inventaria dados e metadados, valida formatos, sistema de coordenadas e layouts binários. O objetivo é encontrar incompatibilidades antes das etapas pesadas.
3. **Preparação LiDAR/CHM:** valida o DTM e DSM, produz o CHM canônico e relatórios/figuras diagnósticas.
4. **Back-geocoding SAR:** mapeia as imagens SAR para a grade geográfica comum usando as `.grille`; registra mapas de linha/coluna, máscaras e auditoria geométrica.
5. **Landsat e coerência:** cria o mosaico óptico controlado por qualidade e calcula mapas de coerência normalizada do radar.
6. **Dataset:** junta entradas SAR, Landsat, CHM e metadados em patches. Cada exemplo contém matrizes de entrada, CHM de referência, máscara de pixels válidos e informações de qualidade/geometria. No dataset v6 atual há **10.500 amostras**.
7. **Geometria RVoG:** prepara ou audita parâmetros físicos para a banda P. A configuração atual indica `ha_p_real`, não permite calibrar RVoG livremente a partir do conjunto de treino, mas permite calibrar atenuação/signo com `kz` real.
8. **Folds espaciais:** divide os exemplos por blocos/regiões, e não por patches sorteados sem considerar localização. Isso reduz a chance de treino e validação conterem vizinhos quase idênticos. O projeto usa 5 folds e audita também o efeito de um buffer espacial.
9. **Treino e avaliação:** calcula estatísticas somente no treino, cria loaders, treina os modelos, salva checkpoints e avalia previsões.
10. **Experimentos multiseed:** repete as combinações com sementes 42, 314 e 2026. Isso mostra quanto o resultado pode variar por inicialização aleatória. Cinco folds × três seeds dão 15 execuções por configuração.
11. **Comparações e produtos:** reúne resultados, executa baselines, salva previsões fora da amostra, gera tabelas, gráficos e mosaicos CHM/AGB.
12. **Retomada:** checkpoints e estados por época permitem retomar treinamentos interrompidos. Os resumos e arquivos `estado_execucao*.json` mostram o andamento.

A configuração salva em `Resultados/TCCT_v3_2_SIBGRAPI/config_pipeline.json` é uma fotografia dos parâmetros utilizados. Entre eles: patches de 30 m, janela de coerência 5, batch size 24, 5 folds, três seeds e limites de épocas para as duas fases. Os valores são escolhas experimentais, não leis gerais.

## 5. Validação espacial: por que importa

Em imagens geográficas, dois patches vizinhos costumam se parecer muito. Se um vai para treino e outro para validação, o modelo pode parecer bom por ter visto praticamente o mesmo local. Isso é **vazamento espacial**: a avaliação fica otimista e não representa bem a previsão em uma região nova.

Por isso, o pipeline agrupa por blocos espaciais e faz validação cruzada em cinco folds. Em cada fold, uma região fica para validação e as demais para treino. O treinamento multiseed repete o processo com diferentes sementes. Os manifests `manifesto_fold_*.csv` registram quais amostras pertencem a cada divisão; seus hashes ajudam a detectar mudanças.

O guia de retomada também menciona um experimento **holdout 70/20/10**: aproximadamente 70% treino, 20% validação e 10% teste final intocado, por faixas de blocos. É uma análise adicional, recomendada depois de congelar os hiperparâmetros. Não substitui os folds espaciais principais.

## 6. Como ler as métricas

As principais métricas de regressão aparecem nos resumos:

- **MAE:** erro absoluto médio, em metros para o CHM. Menor é melhor e costuma ser fácil de interpretar.
- **RMSE:** raiz do erro quadrático médio, também em metros. Penaliza mais erros grandes.
- **R²:** mede quanto da variação dos valores é explicada em relação a uma referência média. Quanto mais perto de 1, melhor; pode ser negativo se o modelo for pior que prever a média.
- **CCC:** coeficiente de concordância, que considera correlação e também diferenças de média/escala. Mais perto de 1 indica melhor concordância.
- **Média ± desvio padrão entre folds/seeds:** resume desempenho e variabilidade. Uma média boa com grande desvio sugere instabilidade entre regiões ou inicializações.

Não escolha um modelo olhando apenas uma métrica. Compare desempenho, variabilidade, distribuição dos erros e a forma como a validação foi feita. Uma métrica alta não elimina problemas de qualidade dos dados ou de representatividade geográfica.

**OOF** (*out-of-fold*) significa que cada previsão é produzida por um modelo que não treinou naquela amostra/região. Essas previsões podem ser reunidas em mosaicos e comparadas de forma mais justa. As figuras pareadas e os scripts de mosaico usam predições salvas; eles não precisam treinar o modelo novamente.

## 7. Mapa rápido dos arquivos de código

Pasta [Codigo](Codigo):

- `Pipeline_TCCT_v3_2_1_SIBGRAPI_REV_GRILLE3D_RETOMAVEL.py`: núcleo do projeto. Faz auditorias e processamento, monta dataset, define modelos, loss, treinamento, folds, avaliação e produtos derivados.
- `Pipeline_TCCT_v3_2_1_SIBGRAPI_REV_GRILLE3D_PORTATIL_RETOMAVEL.ipynb`: interface de notebook para executar etapas do pipeline. O notebook portátil e o `.py` correspondem ao mesmo fluxo geral; para repetir a execução documentada, siga o README de retomada.
- `GUIA_CIENTIFICO_TCCT_v3_2_1.ipynb`: material científico de apoio sobre método e decisões experimentais.
- `Calcular_AGB_Todos_Modelos_Predicoes_Salvas.py`: carrega predições já salvas, alinha amostras de validação com células do inventário florestal, calcula métricas CHM/AGB e gera tabelas/figuras. Não é o treinamento central.
- `TCCT_AGB_Allometry_SupportMatched.py`: calibra/avalia uma relação alométrica com suporte espacial compatível entre CHM e células do inventário. A preocupação com suporte evita comparar uma média de área com uma medida de área diferente sem ajuste.
- `TCCT_Gerar_Mosaico_OOF_CHM_e_AGB.py`: reconstrói mosaicos de previsões OOF a partir dos checkpoints/folds e pode avaliar biomassa no suporte do inventário.
- `Gerar_Figuras_Pareadas_Predicoes_Salvas.py`: valida o alinhamento das predições por amostra e gera gráficos comparativos usando resultados existentes, sem retreinar.
- `Coletar_Artefatos_Multiseed_TCCT.py`: coleta e audita resultados de múltiplas seeds/folds, com hashes e opção de pacote ZIP.
- `Preparar_Pacote_Reprodutibilidade_TCCT_macOS_corrigido.py`: inventaria arquivos, confere a completude das execuções e prepara artefatos de reprodutibilidade. O README correspondente explica modos de cópia; não é código de treinamento.
- `CHANGELOG_TCCT_v3_2_1.md`: alterações da versão.
- `MANIFESTO_SHA256_TCCT_v3_2_1.txt`: hashes de arquivos para conferir integridade.

Os notebooks podem conter cópias ou chamadas das etapas do pipeline. O arquivo `.py` é a referência mais fácil para localizar definições e funções; os notebooks são uma maneira interativa de executar e documentar o processo. O README de retomada descreve o uso operacional do Passo 20 e do holdout.

## 8. Como interpretar as pastas de dados e resultados

- `Dados_Brutos/`: fontes originais. São entradas, não resultados do modelo.
- `Dados_Processados/`: CHM, mosaicos, mapas de coerência, back-geocoding e geometria derivados.
- `Dataset/TCCT_dataset_v6_grille3d/`: índice e metadados do dataset usado. `metadata.json` documenta a versão, número de amostras e decisões de qualidade; `index.csv` cataloga patches.
- `Auditoria/`: relatórios sobre arquivos, geometrias, filtros, coordenadas e reprodutibilidade.
- `Resultados/Metricas/`: resultados agregados e detalhados por modelo/fold, estados de execução e resumos.
- `Resultados/TCCT_v3_2_SIBGRAPI/`: checkpoints, históricos, configurações e resultados do pipeline. `multiseed/` organiza execuções por seed/fold.
- `Resultados/Figuras/`: diagnósticos, comparações e visualizações geradas.
- Arquivos `.pth`: pesos/checkpoints de redes neurais. Eles guardam parâmetros aprendidos; não são código-fonte nem imagens legíveis diretamente.
- Arquivos `.csv`: tabelas; `.json`: configurações e relatórios; `.npy`/`.npz`: matrizes numéricas; `.tif`: raster geográfico.

Os arquivos existentes são resultados de execuções específicas; o fato de estarem na pasta não garante que todas as combinações planejadas terminaram. Consulte os resumos e estados de execução antes de afirmar que o experimento está completo.

## 9. Um glossário curto

| Termo | Em palavras simples |
|---|---|
| Patch | Pequeno recorte de imagem usado como um exemplo do modelo. |
| Banda | Faixa do espectro medida por um sensor. |
| Polarização | Orientação da onda do radar ao transmitir/receber. |
| Coerência | Semelhança complexa entre respostas de radar de aquisições relacionadas. |
| Back-geocoding | Mapeamento dos pixels SAR para suas posições geográficas. |
| Tensor | Matriz numérica que o modelo processa; pode ter canais, linhas e colunas. |
| Parâmetro/peso | Número ajustável dentro da rede, aprendido durante o treinamento. |
| Seed/semente | Valor que controla aleatoriedade para tornar uma execução repetível. |
| Fold | Uma das divisões da validação cruzada. |
| Checkpoint | Arquivo com estado/pesos do modelo para avaliar ou retomar treinamento. |
| Ablation/ablação | Experimento que remove uma parte do modelo para medir sua contribuição. |
| Baseline | Método de referência simples ou alternativo para comparação. |
| Vazamento | Informação de validação/teste que influencia o treinamento ou deixa a avaliação artificialmente fácil. |
| CRS | Sistema de referência de coordenadas que informa onde os pixels ficam na Terra. |
| Raster | Imagem em grade geográfica, com um valor por pixel. |

## 10. Um roteiro amigável para explorar o código

1. Leia este guia e o [README de instalação e retomada](README_TCCT_v3_2_1_RETOMADA.md).
2. Abra `Resultados/TCCT_v3_2_SIBGRAPI/config_pipeline.json` para ver os parâmetros e `Dataset/TCCT_dataset_v6_grille3d/metadata.json` para entender de onde vieram as 10.500 amostras.
3. No pipeline `.py`, localize os títulos de seção começando por `0.` até `9.`. Leia primeiro as seções 4 a 7: dataset, folds, arquitetura, loss e treinamento.
4. Depois leia as etapas de preparação (1 a 3B) para acompanhar como as fontes são alinhadas.
5. Por fim, examine um `resumo_final_*.csv`, um `resumo_por_fold_*.csv` e os históricos. O primeiro resume; o segundo revela variação geográfica; o histórico mostra o treinamento época a época.
6. Ao comparar modelos, procure resultados por fold e seed, e confira se as previsões estão alinhadas às mesmas amostras.

### Limites desta explicação

Este guia explica a intenção e a estrutura do código, mas não substitui uma revisão científica dos pressupostos, das escolhas de hiperparâmetros ou da validade externa do estudo. A qualidade do modelo depende da qualidade/alinhamento dos dados e de quão bem as regiões de validação representam locais onde se deseja aplicar o método.
