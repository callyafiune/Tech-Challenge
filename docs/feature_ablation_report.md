# Ablacao de Features - Telco Churn

## Objetivo

Avaliar se atributos de baixo valor preditivo ou grupos redundantes podem ser removidos ou
sumarizados sem piorar o F1 do campeao tabular.

## Protocolo

- Modelo fixo: RandomForest com os hiperparametros do campeao `random_forest_005`.
- Validacao cruzada estratificada com 5 folds.
- Threshold de F1 escolhido em split interno de cada fold.
- Todos os experimentos foram registrados no MLflow em `telco-churn-feature-ablation`.
- Criterio estrito de nao piora: F1 medio em threshold 0,5 maior ou igual ao conjunto completo.

## Referencia

- Feature set: `full_current`.
- Features finais: `80`.
- F1 medio em threshold 0,5: `0.6395`.
- F1 medio com threshold interno: `0.6340`.
- PR-AUC media: `0.6568`.

## Melhor F1

- Feature set: `no_gender`.
- Features finais: `79`.
- F1 medio em threshold 0,5: `0.6402`.
- Delta contra referencia: `+0.0006`.
- F1 medio com threshold interno: `0.6344`.

## Melhor Reducao Sem Piora Estrita

`no_gender` preservou F1 com `79` features finais e F1 medio `0.6402`.

## Sumarizacoes Avaliadas

| Feature set | Ideia testada | Features finais | F1 medio | Delta F1 | Leitura |
|---|---|---:|---:|---:|---|
| `relationship_summarized` | Trocar `Partner` e `Dependents` por `has_family_context` | 79 | 0.6399 | +0.0004 | Boa candidata: simplifica e nao piora F1 estrito. |
| `service_counts_only` | Trocar servicos individuais por contagens e buckets | 76 | 0.6387 | -0.0008 | Quase empata; melhora F1 com threshold interno, mas nao passa no criterio estrito. |
| `streaming_summarized` | Trocar `StreamingTV` e `StreamingMovies` por `streaming_bundle` | 78 | 0.6378 | -0.0018 | Perda pequena; aceitavel apenas se simplicidade for prioridade. |
| `protection_count_only` | Trocar protecoes individuais por contagem | 65 | 0.6357 | -0.0038 | Remove muitas features, mas perde sinal demais para promover agora. |
| `compact_operational` | Remover demografia e servicos individuais, mantendo resumos | 62 | 0.6324 | -0.0072 | Compactacao agressiva demais para o objetivo de F1. |

## Recomendacao

Promover com seguranca metodologica apenas:

- Remover `gender`, pois melhorou levemente o F1 medio e reduz risco etico.
- Considerar substituir `Partner` e `Dependents` por `has_family_context`, pois manteve F1 em
  threshold 0,5 sem piora.

Nao promover por enquanto:

- Remocao agressiva dos servicos de protecao, porque a perda de PR-AUC e F1 indica que esses campos
  ainda carregam sinal util.
- Pipeline compacto operacional, porque a queda de F1 e maior que o ganho de simplicidade neste
  momento.

## Leitura Critica

Ablacao deve ser interpretada com cuidado porque as diferencas de F1 sao pequenas e podem estar
dentro da variancia entre folds. Quando duas versoes empatam em F1, a versao com menos features e
menor risco etico deve ser preferida, especialmente se remove `gender` ou resume atributos
demograficos.

Artefato comparativo: `reports/feature_ablation/comparison.csv`.
