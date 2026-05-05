# Métrica de Negócio - Custo de Churn Evitado

O objetivo operacional não é apenas classificar churn, mas priorizar clientes para uma campanha de
retenção financeiramente defensável.

## Premissas adotadas

- Cliente salvo gera valor aproximado de 12 meses de mensalidade.
- Custo de retenção equivale a 1 mensalidade oferecida como incentivo.
- A campanha é aplicada a clientes cujo score fica acima do threshold escolhido.

## Fórmulas

```text
valor_cliente = MonthlyCharges * meses_retidos
custo_oferta = MonthlyCharges * multiplicador_oferta
economia_incremental = soma(valor_cliente dos TP) - soma(custo_oferta dos clientes acionados)
valor_operacional_liquido = economia_incremental - soma(valor_cliente dos FN)
```

Também registramos:

- `lift_at_top_20pct`: concentração de churn real nos 20% clientes com maior score.
- `threshold_business_optimal`: threshold que maximiza `business_incremental_savings`.
- Sensibilidade futura: variar custo da oferta e meses retidos em novas rodadas de validação.
