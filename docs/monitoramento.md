# Plano de Monitoramento

## Objetivo

Garantir que a API de churn continue disponível, com dados de entrada válidos e desempenho
compatível com o observado em validação.

## Saúde da API

Métricas:

- Latência p50, p95 e p99.
- Taxa de erro 4xx e 5xx.
- Requisições por segundo.
- Uso de CPU e memória.

Alertas iniciais:

- p95 acima de 300 ms por 5 minutos.
- Erros 5xx acima de 1% por 5 minutos.
- `/health` retornando `degraded`.

## Qualidade dos Dados

Validações em tempo de request:

- Tipos e domínios via Pydantic.
- Campos extras rejeitados.
- Ranges mínimos para `tenure` e `MonthlyCharges`.

Monitoramento em lote:

- Distribuição semanal de `tenure`, `MonthlyCharges`, `Contract`, `InternetService` e
  `PaymentMethod`.
- PSI semanal contra a base de treino.
- Alerta se PSI > 0,2 em feature crítica.

## Qualidade do Modelo

Quando o rótulo real estiver disponível:

- AUC-ROC, PR-AUC, F1, recall e precision em janela móvel.
- Lift no top 20% de risco.
- Economia incremental estimada da campanha.

Alertas:

- Queda maior que 5 pontos percentuais em AUC ou F1 frente ao holdout.
- Recall abaixo do mínimo operacional definido pelo time de retenção.

## Governança

- Expor versão do modelo em `/health` e `/predict`.
- Registrar `request_id`, versão do modelo, probabilidade, threshold e latência.
- Não registrar payload bruto com dados sensíveis nos logs da aplicação.
- Reavaliar threshold quando custo de campanha ou valor de retenção mudar.

## Runbook

1. Confirmar se `/health` está `ok`.
2. Validar drift das features críticas.
3. Comparar métricas recentes com o holdout.
4. Se houver regressão, reverter para versão anterior ou promover baseline mais simples.
5. Abrir tarefa de retreino com nova amostra rotulada.
