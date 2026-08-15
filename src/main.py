from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import os
import sys

# Garante a importação dos módulos locais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import ThompsonSamplingBandit

app = FastAPI(
    title="Plataforma Adaptativa de Ofertas Digitais",
    description="API para recomendação de ofertas em tempo real utilizando Thompson Sampling",
    version="1.0.0"
)

# Nomes das ofertas disponíveis no sistema
OFFERS = {
    0: "Depósito a Prazo (Investimento)",
    1: "Cartão de Crédito Premium",
    2: "Empréstimo Pessoal",
    3: "Seguro de Vida"
}

# Inicializa o modelo de bandit adaptativo
bandit = ThompsonSamplingBandit(n_arms=len(OFFERS))

# Define os Priors aprendidos na simulação/histórico (ajuste conforme necessário)
bandit.alpha = np.array([120.0, 35.0, 50.0, 20.0])
bandit.beta = np.array([550.0, 650.0, 580.0, 680.0])


class ClientPayload(BaseModel):
    cliente_id: int
    age: int
    job: str
    balance: float
    housing: str
    loan: Optional[str] = "no"


class FeedbackPayload(BaseModel):
    cliente_id: int
    offered_arm: int
    converted: int  # 1 para conversão/sucesso, 0 para recusa


@app.get("/")
def health_check():
    """Endpoint de saúde da API."""
    return {
        "status": "online",
        "service": "Adaptive Offer Engine",
        "algorithm": "Thompson Sampling (Bayesian Multi-Armed Bandit)"
    }


@app.post("/recommend")
def recommend_offer(client: ClientPayload):
    """
    Recebe os dados do cliente e seleciona a oferta com maior propensão de conversão
    equilibrando exploração e explotação.
    """
    # Amostra a ação ideal
    recommended_arm = bandit.select_action()
    recommended_offer = OFFERS[recommended_arm]
    expected_rates = bandit.get_expected_rewards()

    return {
        "cliente_id": client.cliente_id,
        "recommended_arm": int(recommended_arm),
        "recommended_offer": recommended_offer,
        "expected_conversion_rate": round(float(expected_rates[recommended_arm]), 4),
        "status": "sucesso"
    }


@app.post("/feedback")
def update_feedback(feedback: FeedbackPayload):
    """
    Recebe a resposta da oferta enviada e atualiza os parâmetros do modelo (Feedback Loop).
    """
    if feedback.offered_arm not in OFFERS:
        raise HTTPException(status_code=400, detail="Braço/Oferta inválido.")

    # Atualiza as distribuições do Thompson Sampling
    bandit.update(action=feedback.offered_arm, reward=feedback.converted)

    return {
        "message": "Feedback registrado e priors do modelo atualizados com sucesso.",
        "updated_priors": {
            "alpha": bandit.alpha.tolist(),
            "beta": bandit.beta.tolist()
        }
    }