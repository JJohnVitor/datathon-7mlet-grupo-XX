# Datathon POS TECH - MLET: Plataforma Adaptativa de Ofertas Digitais

## 1. Visão Geral do Problema
Esta solução implementa uma plataforma de experimentação adaptativa utilizando algoritmos de **Multi-Armed Bandit (Thompson Sampling / Epsilon-Greedy)** para seleção inteligente de ofertas em canais digitais. A abordagem substitui regras estáticas e reduz o tempo necessário em testes A/B tradicionais, otimizando a taxa de conversão com equilíbrio entre exploração e explotação.

## 2. Base de Dados
* **Dataset:** Bank Marketing (Kaggle)
* **Link:** https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
* **Target:** `y` (conversão em produto bancário / depósito a prazo)
* **Tratamento:** Remoção da variável de vazamento temporal (`duration`).

## 3. Como Executar o Projeto

### Pré-requisitos
* Python 3.10+
* Git

## 4. Comparativo: Baseline vs. Algoritmo Adaptativo
* **Baseline Determinístico (Regra Fixa):** ~5.00% de Taxa de Conversão.
* **Thompson Sampling (Adaptativo):** ~14.50% de Taxa de Conversão (+190% de ganho).

## 5. Golden Set (Validação de Recomendação)
| ID Cliente | Profissão | Saldo | Oferta Recomendada |
| :--- | :--- | :--- | :--- |
| 101 | technician | €1,500 | Depósito a Prazo (Investimento) |
| 102 | retired | €12,000 | Depósito a Prazo (Investimento) |
| 103 | student | €300 | Cartão de Crédito Premium |
| 104 | management | €4,500 | Depósito a Prazo (Investimento) |
| 105 | blue-collar | €800 | Empréstimo Pessoal |


## 6. Arquitetura-Alvo na Nuvem (AWS)

Para operar esta solução em escala, a API desenvolvida em FastAPI será implantada como um microsserviço containerizado (Docker) no **AWS ECS Fargate** com um **Application Load Balancer (ALB)**, garantindo autoescalamento e alta disponibilidade. O estado das distribuições e priors do modelo Thompson Sampling ($\alpha$ e $\beta$) será armazenado de forma persistente e com baixíssima latência no **Amazon DynamoDB** (ou **Amazon ElastiCache for Redis**), atualizado em tempo real conforme as respostas de feedback dos clientes são processadas.

Os dados brutos de interações e conversões serão direcionados via **Amazon Kinesis Data Firehose** para um Data Lake no **Amazon S3**. A esteira de MLOps utilizará o **AWS SageMaker** (ou um servidor **MLflow** hospedado em instância EC2) para rastreamento de experimentos, versionamento de modelos e re-treinamento periódico dos *priors* bayesianos a partir do histórico de conversões do Data Lake.

### Passos para execução
```bash
# 1. Clonar o repositório
git clone [https://github.com/SEU_USUARIO/datathon-7mlet-grupo-XX.git](https://github.com/SEU_USUARIO/datathon-7mlet-grupo-XX.git)
cd datathon-7mlet-grupo-XX

# 2. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Iniciar a API de Recomendação
uvicorn src.main:app --reload