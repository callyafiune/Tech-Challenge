# Métrica de Negócio - Custo de Churn Evitado

O objetivo operacional não é apenas classificar churn, mas priorizar clientes para uma campanha de
retenção financeiramente defensável.

## Premissas adotadas

- Cliente salvo gera valor aproximado de 12 meses de mensalidade.
- Custo de retenção equivale a 1 mensalidade oferecida como incentivo.
- A campanha é aplicada a clientes cujo score fica acima do threshold escolhido.
- Um falso negativo representa perda potencial do cliente que churnaria e não recebeu ação.
- Um falso positivo representa custo de abordagem/oferta para cliente que não churnaria.

## Fórmulas

```text
valor_cliente = MonthlyCharges * meses_retidos
custo_oferta = MonthlyCharges * multiplicador_oferta
economia_incremental = soma(valor_cliente dos TP) - soma(custo_oferta dos clientes acionados)
valor_operacional_liquido = economia_incremental - soma(valor_cliente dos FN)
custo_fp = soma(custo_oferta dos FP)
custo_fn = soma(valor_cliente dos FN)
custo_total_erro = custo_fp + custo_fn
razao_custo_fn_fp = meses_retidos / multiplicador_oferta
```

Com as premissas atuais, deixar um churner passar custa aproximadamente `12x` o custo unitário de
acionar um cliente indevidamente, pois `meses_retidos=12` e `multiplicador_oferta=1`. Essa razão é
uma aproximação inicial: em operação real, deve ser substituída por margem, LTV, custo de aquisição,
probabilidade real de aceite da oferta e orçamento de campanha.

Também registramos:

- `lift_at_top_20pct`: concentração de churn real nos 20% clientes com maior score.
- `threshold_business_optimal`: threshold que maximiza `business_incremental_savings`.
- `business_false_positive_cost`: custo de ofertas para falsos positivos.
- `business_false_negative_cost`: valor potencial perdido nos falsos negativos.
- `business_total_error_cost`: soma do custo de falsos positivos e falsos negativos.
- `business_fn_fp_unit_cost_ratio`: razão unitária entre custo de FN e custo de FP.

## Uso no Threshold e no Treinamento

A primeira alavanca recomendada é ajustar o threshold de decisão, não alterar diretamente o treino.
O modelo deve aprender probabilidades bem calibradas; depois, o ponto de corte pode ser escolhido
para maximizar economia incremental ou minimizar `business_total_error_cost`.

O uso de pesos no treinamento deve ser tratado como experimento separado. Na MLP, `pos_weight`
compensa principalmente o desbalanceamento da classe positiva. Usar um peso econômico muito alto
pode aumentar recall, mas também pode piorar calibração, precision e custo com falsos positivos.
Qualquer nova ponderação econômica deve ser registrada no MLflow e comparada por F1, PR-AUC,
Brier/log loss, lift e custo total.

## Sensibilidade Recomendada

Executar cenários com diferentes premissas:

- `meses_retidos`: 6, 12 e 18 meses.
- `multiplicador_oferta`: 0,5, 1,0 e 2,0 mensalidades.
- Razões aproximadas de custo FN/FP: 3x, 5x, 10x e 12x.

O objetivo é verificar se a recomendação de threshold permanece estável quando as premissas de
negócio mudam.
