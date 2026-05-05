# ML Canvas - Previsão de Churn Telco

## Problema de Negócio

Uma operadora de telecomunicações precisa priorizar clientes com maior risco de cancelamento para
ações de retenção. O modelo deve apoiar decisões operacionais antes que o cliente efetivamente
churne.

## Stakeholders

- Diretoria: acompanha impacto financeiro e redução de churn.
- Marketing e Retenção: usa a lista priorizada para campanhas.
- Atendimento ao Cliente: executa contatos e ofertas.
- Engenharia de Dados e ML: mantém pipeline, API e monitoramento.
- Clientes: recebem ofertas ou abordagens de retenção.
- Compliance e Governança: valida uso responsável dos dados.

## Decisão Apoiada

Para cada cliente ativo, produzir uma probabilidade de churn. A operação pode acionar campanhas nos
clientes com maior score, respeitando capacidade e orçamento.

## Dados

Dataset Telco Customer Churn com 7.043 clientes, 21 colunas e alvo binário `Churn`. As principais
famílias de atributos são perfil do cliente, serviços contratados, contrato, forma de pagamento,
tenure e cobranças.

## Features Iniciais

- `tenure`, `MonthlyCharges` e `TotalCharges`.
- Contrato, serviço de internet, método de pagamento e serviços adicionais.
- `avg_monthly_spend`, `charges_delta`, `tenure_bucket`, `num_services` e
  `has_protection_bundle`.

## Métricas Técnicas

- Primária: AUC-ROC.
- Complementares: PR-AUC, F1, precision, recall, balanced accuracy, log loss e Brier score.
- Comparação planejada: DummyClassifier, Regressão Logística, MLP PyTorch e challengers tabulares.

## Métrica de Negócio

A métrica de negócio é a economia incremental estimada da campanha de retenção:

```text
economia_incremental = valor_salvo_em_TP - custo_da_campanha_em_TP_e_FP
custo_total_erro = custo_dos_FP + custo_dos_FN
```

Premissas iniciais:

- Valor salvo por cliente retido: `MonthlyCharges * 12`.
- Custo da oferta: `MonthlyCharges * 1`.
- Falso positivo: cliente acionado sem churn real, gerando custo de oferta e contato.
- Falso negativo: cliente com churn real que não foi acionado, gerando perda potencial de receita.
- Razão inicial de custo FN/FP: aproximadamente `12x`, antes de ajustes por margem, LTV e aceite de
  oferta.
- Threshold ótimo: ponto de corte que maximiza a economia incremental ou minimiza o custo total de
  FP e FN no conjunto validado.
- Peso econômico no treino: usar apenas como experimento controlado; a decisão principal deve ser
  feita pelo threshold para preservar calibração das probabilidades.

## SLOs Iniciais

- Inferência futura via API abaixo de 300 ms por cliente em ambiente local.
- Pipeline reprodutível com seed fixa e validação cruzada estratificada.
- Registro completo de parâmetros, métricas e artefatos em MLflow.

## Riscos e Limitações

- Dataset histórico pode não representar sazonalidade ou mudanças comerciais recentes.
- Variáveis sensíveis indiretas podem gerar vieses de abordagem.
- Probabilidade precisa ser calibrada antes de uso financeiro intensivo.
- A estratégia de oferta ainda usa premissas sintéticas e deve ser calibrada com dados reais de
  margem, LTV e custo de campanha.
