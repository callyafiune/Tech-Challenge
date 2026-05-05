# Dicionário de Dados - Telco Customer Churn

Este documento descreve, em português, cada coluna do arquivo `Telco-Customer-Churn.csv`.
A base contém informações demográficas, serviços contratados, cobranças e o rótulo de churn de
clientes de telecomunicações.

| Coluna | Tipo na base | Valores observados | Descrição |
|---|---|---|---|
| `customerID` | Texto | Identificador único | Código único do cliente. É usado apenas para identificação e deve ser removido antes do treinamento do modelo, pois não representa uma característica preditiva generalizável. |
| `gender` | Categórica | `Female`, `Male` | Gênero informado do cliente. Pode ser usado para análise de perfil, mas exige cuidado em interpretações de viés e uso operacional. |
| `SeniorCitizen` | Numérica binária | `0`, `1` | Indica se o cliente é idoso. `1` representa cliente idoso e `0` representa cliente não idoso. |
| `Partner` | Categórica binária | `Yes`, `No` | Indica se o cliente possui parceiro ou parceira. |
| `Dependents` | Categórica binária | `Yes`, `No` | Indica se o cliente possui dependentes. |
| `tenure` | Numérica inteira | Meses de permanência | Quantidade de meses em que o cliente permaneceu ativo na empresa. É uma das variáveis mais relevantes para churn, pois clientes recentes tendem a ter comportamento diferente de clientes antigos. |
| `PhoneService` | Categórica binária | `Yes`, `No` | Indica se o cliente possui serviço de telefone contratado. |
| `MultipleLines` | Categórica | `Yes`, `No`, `No phone service` | Indica se o cliente possui múltiplas linhas telefônicas. O valor `No phone service` significa que o cliente não possui serviço de telefone. |
| `InternetService` | Categórica | `DSL`, `Fiber optic`, `No` | Tipo de serviço de internet contratado. `No` indica ausência de serviço de internet. |
| `OnlineSecurity` | Categórica | `Yes`, `No`, `No internet service` | Indica se o cliente possui serviço adicional de segurança online. `No internet service` significa que o cliente não possui internet contratada. |
| `OnlineBackup` | Categórica | `Yes`, `No`, `No internet service` | Indica se o cliente possui serviço de backup online. `No internet service` significa ausência de internet contratada. |
| `DeviceProtection` | Categórica | `Yes`, `No`, `No internet service` | Indica se o cliente possui proteção de dispositivo contratada. `No internet service` significa ausência de internet contratada. |
| `TechSupport` | Categórica | `Yes`, `No`, `No internet service` | Indica se o cliente possui suporte técnico contratado. `No internet service` significa ausência de internet contratada. |
| `StreamingTV` | Categórica | `Yes`, `No`, `No internet service` | Indica se o cliente possui serviço de streaming de TV. `No internet service` significa ausência de internet contratada. |
| `StreamingMovies` | Categórica | `Yes`, `No`, `No internet service` | Indica se o cliente possui serviço de streaming de filmes. `No internet service` significa ausência de internet contratada. |
| `Contract` | Categórica | `Month-to-month`, `One year`, `Two year` | Tipo de contrato do cliente. Contratos mensais normalmente estão associados a maior risco de churn do que contratos anuais ou de dois anos. |
| `PaperlessBilling` | Categórica binária | `Yes`, `No` | Indica se o cliente utiliza cobrança sem papel, ou seja, fatura digital. |
| `PaymentMethod` | Categórica | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Método de pagamento utilizado pelo cliente. Pode capturar diferenças operacionais e comportamentais associadas ao risco de churn. |
| `MonthlyCharges` | Numérica decimal | Valor monetário mensal | Valor cobrado mensalmente do cliente. Ajuda a representar o nível de gasto recorrente e o pacote contratado. |
| `TotalCharges` | Texto com conteúdo numérico | Valor monetário acumulado ou espaço em branco | Total acumulado cobrado do cliente ao longo do relacionamento. Na base original chega como texto e possui alguns valores em branco; no pipeline, esses valores são convertidos para número e tratados como `0` quando `tenure=0`. |
| `Churn` | Categórica alvo | `Yes`, `No` | Variável alvo do problema. `Yes` indica que o cliente cancelou o serviço; `No` indica que permaneceu ativo. No treinamento, é codificada como `1` para churn e `0` para não churn. |

## Observações de Modelagem

- `customerID` não deve ser usado como feature do modelo.
- `Churn` é a variável alvo e não deve entrar como variável explicativa.
- `TotalCharges` precisa de conversão numérica antes do treinamento.
- Categorias como `No internet service` e `No phone service` representam ausência do serviço base e
  podem ser normalizadas ou codificadas de forma explícita no pipeline.
- Variáveis de contrato, tempo de permanência, cobranças e serviços adicionais tendem a ser muito
  importantes para análise de churn.

## Tratamento das Colunas na Modelagem

O pipeline de modelagem usa `split_features_target` para separar o alvo e depois aplica
`build_feature_pipeline`. A normalização efetiva acontece em três etapas:

1. Limpeza semântica e engenharia de features.
2. Imputação de valores ausentes.
3. Escalonamento numérico e codificação one-hot de variáveis categóricas.

| Coluna original | Tratamento aplicado | Como chega ao modelo |
|---|---|---|
| `customerID` | Removida antes do treinamento. | Não entra como feature. |
| `gender` | Mantida como categórica. | Imputação por moda e `OneHotEncoder(drop="if_binary")`, gerando uma coluna binária. |
| `SeniorCitizen` | Mantida como variável binária numérica. | Imputação por mediana e `StandardScaler`. Também participa da feature `senior_month_to_month`. |
| `Partner` | Mantida como categórica binária. | Imputação por moda e one-hot binário. |
| `Dependents` | Mantida como categórica binária. | Imputação por moda e one-hot binário. |
| `tenure` | Mantida como numérica e usada para criar faixas. | Imputação por mediana e `StandardScaler`. Também gera `tenure_bucket`, `is_zero_tenure`, `month_to_month_low_tenure`, `contract_tenure_segment`, `avg_monthly_spend` e `total_to_monthly_ratio`. |
| `PhoneService` | Mantida como categórica binária. | Imputação por moda e one-hot binário. Também entra no cálculo de `num_services`. |
| `MultipleLines` | `No phone service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra no cálculo de `num_services`. |
| `InternetService` | Mantida como categórica com `DSL`, `Fiber optic` ou `No`. | Imputação por moda e one-hot. Também gera `has_internet_service`, `fiber_without_security` e `internet_security_profile`. |
| `OnlineSecurity` | `No internet service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra em `num_services`, `num_protection_services`, `has_protection_bundle`, `fiber_without_security` e `internet_security_profile`. |
| `OnlineBackup` | `No internet service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra em `num_services`, `num_protection_services` e `has_protection_bundle`. |
| `DeviceProtection` | `No internet service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra em `num_services`, `num_protection_services` e `has_protection_bundle`. |
| `TechSupport` | `No internet service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra em `num_services`, `num_protection_services`, `has_protection_bundle` e `internet_security_profile`. |
| `StreamingTV` | `No internet service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra em `num_services` e `streaming_bundle`. |
| `StreamingMovies` | `No internet service` é convertido para `No`. | Imputação por moda e one-hot binário. Também entra em `num_services` e `streaming_bundle`. |
| `Contract` | Mantida como categórica. | Imputação por moda e one-hot. Também gera `contract_tenure_segment`, `payment_contract_profile`, `electronic_check_month_to_month`, `month_to_month_low_tenure` e `senior_month_to_month`. |
| `PaperlessBilling` | Mantida como categórica binária. | Imputação por moda e one-hot binário. |
| `PaymentMethod` | Mantida como categórica. | Imputação por moda e one-hot. Também gera `payment_contract_profile` e `electronic_check_month_to_month`. |
| `MonthlyCharges` | Mantida como numérica. | Imputação por mediana e `StandardScaler`. Também gera `charges_delta` e `total_to_monthly_ratio`. |
| `TotalCharges` | Convertida de texto para número. Valores vazios viram `0.0` quando `tenure=0`; outros inválidos ficam como ausentes para imputação. | Imputação por mediana e `StandardScaler`. Também gera `avg_monthly_spend` e `total_to_monthly_ratio`. |
| `Churn` | Separada como alvo. `Yes` vira `1`; `No` vira `0`. | Não entra nas features; é usada como variável resposta supervisionada. |

## Features Criadas no Pipeline

| Feature criada | Tipo | Regra |
|---|---|---|
| `avg_monthly_spend` | Numérica | `TotalCharges / tenure`, com proteção para `tenure=0`. |
| `charges_delta` | Numérica | `MonthlyCharges - avg_monthly_spend`. |
| `total_to_monthly_ratio` | Numérica | `TotalCharges / MonthlyCharges`, com proteção para divisão por zero. |
| `tenure_bucket` | Categórica | Faixas de permanência: `0-6`, `7-12`, `13-24`, `25-48`, `49+`. |
| `num_services` | Numérica | Soma dos serviços marcados como `Yes` em telefone, múltiplas linhas e serviços de internet. |
| `num_protection_services` | Numérica | Soma de `OnlineSecurity`, `OnlineBackup`, `DeviceProtection` e `TechSupport` marcados como `Yes`. |
| `has_protection_bundle` | Numérica binária | `1` quando todos os serviços de proteção estão ativos; caso contrário `0`. |
| `is_zero_tenure` | Numérica binária | `1` quando `tenure=0`; caso contrário `0`. |
| `has_internet_service` | Numérica binária | `1` quando `InternetService` é diferente de `No`; caso contrário `0`. |
| `fiber_without_security` | Numérica binária | `1` quando o cliente usa fibra e não possui segurança online. |
| `electronic_check_month_to_month` | Numérica binária | `1` quando o pagamento é `Electronic check` e o contrato é mensal. |
| `month_to_month_low_tenure` | Numérica binária | `1` quando o contrato é mensal e `tenure <= 12`. |
| `senior_month_to_month` | Numérica binária | `1` quando o cliente é idoso e tem contrato mensal. |
| `streaming_bundle` | Numérica binária | `1` quando `StreamingTV` e `StreamingMovies` estão ativos. |
| `contract_tenure_segment` | Categórica | Combinação de `Contract` com `tenure_bucket`. |
| `internet_security_profile` | Categórica | Combinação de `InternetService`, `OnlineSecurity` e `TechSupport`. |
| `payment_contract_profile` | Categórica | Combinação de `PaymentMethod` com `Contract`. |

## Transformação Final

Depois da engenharia de features:

- Features numéricas recebem `SimpleImputer(strategy="median")` e `StandardScaler`.
- Features categóricas recebem `SimpleImputer(strategy="most_frequent")` e `OneHotEncoder`.
- O `OneHotEncoder` usa `drop="if_binary"`, removendo uma categoria apenas quando a variável é
  binária.
- Categorias desconhecidas em produção são ignoradas com `handle_unknown="ignore"`.
- O pipeline final gera uma matriz numérica pronta para Scikit-Learn e para a MLP em PyTorch.

## Resultado da Ablação de Features

Foi executada uma rodada de ablação usando o RandomForest campeão como modelo fixo. O objetivo foi
verificar quais colunas poderiam ser removidas ou resumidas sem piorar F1.

Principais conclusões:

- `gender` pode ser removida: o F1 médio subiu de `0.6395` para `0.6402`, com 79 features finais.
- `Partner` e `Dependents` podem ser sumarizadas em `has_family_context`: o F1 médio ficou em
  `0.6399`, sem piora contra o conjunto completo.
- `StreamingTV` e `StreamingMovies` podem ser resumidas em `streaming_bundle`, mas houve pequena
  perda de F1 (`-0.0018`), então essa troca deve ser usada apenas se simplicidade for prioridade.
- Serviços de proteção individuais ainda carregam sinal útil; trocar tudo por contagem reduziu F1 e
  PR-AUC.

Relatório completo: `docs/feature_ablation_report.md`.
