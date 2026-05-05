# Recomendação Final - Telco Churn

## Decisão Recomendada

Para a versão final do projeto, a recomendação é manter a **MLP PyTorch refinada** como modelo
neural central, porque ela combina arquitetura aderente ao escopo técnico com desempenho
competitivo:

- AUC-ROC holdout: `0.8463`.
- PR-AUC holdout: `0.6580`.
- F1 otimizado holdout: `0.6396`.
- Threshold de F1: `0.51`.
- Threshold de negócio: `0.18`.

Para uso operacional futuro, o melhor candidato tabular é o **RandomForest sem `gender`**, pois teve
o melhor F1 médio na rodada de ablação, foi confirmado na rodada de refinamento sem vazamento e foi
promovido formalmente como challenger operacional:

- AUC-ROC CV: `0.8452`.
- PR-AUC CV: `0.6558`.
- F1 CV: `0.6402`.
- Ganho contra RandomForest completo: `+0.0006`.
- Redução de risco ético: remove `gender`.
- Promoção formal: `docs/promocao_challenger.md`.
- MLflow run de promoção: `e04eb2c506144cefb37bbd20aa17797f`.
- Registered model: `telco-churn-random-forest-challenger`, alias `challenger`.
- Refinamento adicional: `docs/f1_refinement_report.md`, sem novo ganho de F1 em threshold 0,5 e
  sem promoção adicional.
- Comparação estatística: `docs/model_comparison_statistical.md`, sem evidência suficiente para
  trocar a recomendação por pequenas diferenças de média em 5 folds.

## Feature Set Recomendado

Recomendação para a próxima versão do pipeline:

1. Remover `gender`.
2. Considerar substituir `Partner` e `Dependents` por `has_family_context`.
3. Manter os serviços de proteção individuais por enquanto.
4. Manter `StreamingTV` e `StreamingMovies` individuais, salvo se simplicidade operacional for mais importante que o máximo F1.

Justificativa:

- `gender` não trouxe ganho preditivo e sua remoção melhorou levemente o F1.
- `Partner` e `Dependents` podem ser resumidos sem perda relevante.
- Serviços de proteção ainda carregam sinal útil; removê-los derrubou F1 e PR-AUC.
- Compactação agressiva reduziu demais o desempenho.
- Interações adicionais de contrato, pagamento, cobrança e proteção aumentaram as features finais
  de `79` para até `172` e não superaram o F1 da referência.
- A comparação pareada por fold reforçou uma decisão conservadora: manter a recomendação atual e
  não promover os candidatos do refinamento.

## Recomendação de Negócio

O modelo deve ser usado para priorizar campanhas de retenção, não para tomar decisões automáticas
contra clientes. A operação deve ranquear clientes por probabilidade de churn e acionar uma fração
controlada da base, por exemplo os 20% clientes com maior risco.

A métrica de negócio deve continuar acompanhando:

- F1, para equilíbrio entre precisão e recall.
- PR-AUC, por causa do desbalanceamento da base.
- Lift@20%, por refletir a utilidade em campanhas priorizadas.
- Valor líquido estimado da campanha, com custo de contato e valor salvo.

## Escolha Final para Apresentação

Na apresentação, a narrativa recomendada é:

> A MLP PyTorch refinada é o modelo neural principal do projeto.
> Em paralelo, os experimentos Scikit-Learn e a ablação mostraram que modelos tabulares são muito
> competitivos neste dataset. O melhor challenger operacional foi um RandomForest sem `gender`,
> com F1 médio de `0.6402`, reforçando a decisão de remover atributos sensíveis quando eles não
> adicionam valor preditivo. Uma rodada final de refinamento sem vazamento testou novas interações
> e confirmou que elas não justificam aumentar a complexidade do pipeline.

## Próximos Passos

- Retreinar uma versão `v2` da MLP sem `gender`.
- Testar a MLP com `has_family_context` no lugar de `Partner` e `Dependents`.
- Validar o challenger não neural em shadow mode antes de qualquer troca do modelo principal.
- Não promover os candidatos da rodada de refinamento de F1.
- Repetir a comparação estatística por fold quando novos candidatos forem treinados.
- Toda promoção futura deve repetir o protocolo usado em `docs/promocao_challenger.md`.
- Monitorar drift de dados e queda de performance antes de usar em campanhas reais.
