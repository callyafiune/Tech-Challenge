# Especificação do Projeto

## Visão Geral

Este projeto constrói uma solução end-to-end para previsão de churn em telecomunicações. A entrega
inclui exploração de dados, baselines, MLP em PyTorch, comparação com modelos Scikit-Learn,
rastreamento com MLflow, API FastAPI, testes automatizados e documentação técnica.

## Objetivo de Negócio

Uma operadora de telecomunicações precisa priorizar clientes com maior risco de cancelamento para
ações de retenção. O modelo deve estimar a probabilidade de churn e apoiar decisões operacionais de
campanha, sem tomar decisões automáticas contra clientes.

## Dataset

Arquivo utilizado: `Telco-Customer-Churn.csv`.

Características esperadas:

- Dados tabulares de clientes de telecomunicações.
- Variáveis demográficas, contratuais, serviços contratados e cobranças.
- Alvo binário `Churn`, com `Yes` para cancelamento e `No` para permanência.

## Requisitos Técnicos

- Código organizado em `src/`.
- Testes automatizados em `tests/`.
- Notebooks de apoio em `notebooks/`.
- Documentação em `docs/`.
- Dependências e ferramentas centralizadas em `pyproject.toml`.
- Automação por `Makefile`.
- Lint com `ruff`.
- Testes com `pytest`.
- Validação de schema com `pandera`.
- Rastreamento de experimentos com `MLflow`.
- API de inferência com `FastAPI`.

## Modelagem

O modelo neural principal deve ser uma MLP em PyTorch com:

- Seeds fixadas para reprodutibilidade.
- Tratamento de desbalanceamento de classes.
- Early stopping.
- Comparação com baselines Scikit-Learn.
- Métricas técnicas: AUC-ROC, PR-AUC, F1, precision, recall e balanced accuracy.
- Métricas operacionais: lift@20%, threshold de negócio e economia incremental estimada.

Também são permitidos modelos Scikit-Learn como baselines, challengers ou referência operacional,
desde que todos os experimentos sejam rastreados no MLflow.

## API

A API deve expor:

- `GET /health`: status da aplicação e versão do modelo.
- `POST /predict`: probabilidade de churn, predição binária e threshold utilizado.

As entradas devem ser validadas com Pydantic e a aplicação deve usar logging estruturado.

## Documentação Esperada

- README com setup, execução, resultados e estrutura do projeto.
- Dicionário de dados.
- Relatório de qualidade dos dados.
- ML Canvas.
- Model Card.
- Plano de monitoramento.
- Arquitetura de deploy.
- Recomendação final.
- Checklist de entrega.

## Critérios de Aceite

- Projeto instala com `pip install -e .`.
- `ruff check .` passa sem erros.
- `pytest` passa sem erros.
- MLflow contém histórico dos experimentos relevantes.
- A MLP está treinada, documentada e comparada com baselines.
- A API responde em `/health` e `/predict`.
- Dados locais, ambientes virtuais, modelos treinados e runs do MLflow não são versionados no Git.
- Documentação final está em português e consistente com os artefatos gerados.

## Entrega

A entrega principal é o repositório organizado com código, testes, documentação e instruções de
execução. Quando o deploy em nuvem não for executado, a arquitetura recomendada deve estar
documentada como plano operacional.
