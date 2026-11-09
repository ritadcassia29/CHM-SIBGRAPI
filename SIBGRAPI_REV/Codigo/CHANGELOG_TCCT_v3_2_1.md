# Changelog — TCCT v3.2.1

## Portabilidade

- Detecção da raiz `SIBGRAPI_REV` em macOS, Windows e Linux.
- Interrupção em caso de múltiplas cópias válidas.
- Notebook portátil carrega explicitamente o script v3.2.1.

## Correções preservadas

- classificação de arquivos `.grille`;
- estatísticas de `k_z` pela função existente;
- proteção contra `NaN` no ramo RVoG com `theta` ausente;
- médias mascaradas com `torch.where`;
- clipping com erro para gradientes não finitos.

## Retomada do treinamento

- combinação fold–seed concluída é detectada e pulada;
- progresso mostra seed, fold, posição, pendências e ETA;
- resumos multiseed são atualizados incrementalmente;
- checkpoint por época inclui modelo, loss, AdamW, scheduler, melhor estado,
  histórico, early stopping, DataLoader e RNG;
- retomada da última época concluída;
- estado parcial antigo é arquivado para evitar mistura de trajetórias.

## Experimento adicional

- holdout espacial aproximado 70/20/10;
- teste final intocado;
- paciência maior opcional;
- protocolo explicitamente complementar ao 5-fold multiseed.

## Documentação

- notebook executável com explicação de cada etapa;
- notebook científico somente com explicações;
- contexto mestre atualizado.
