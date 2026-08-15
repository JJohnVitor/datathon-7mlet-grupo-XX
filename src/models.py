import numpy as np
import pandas as pd

class DeterministicBaseline:
    """Baseline estático/determinístico: Recomenda sempre a melhor oferta histórica fixa."""
    def __init__(self, fixed_action: int = 0):
        self.fixed_action = fixed_action

    def select_action(self, context=None) -> int:
        return self.fixed_action


class ThompsonSamplingBandit:
    """
    Algoritmo Adaptativo Thompson Sampling (Bayesian Bandit)
    Aplica prior Beta(alpha, beta) para cada braço/oferta.
    """
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Sucessos iniciais (prior uniforme)
        self.beta = np.ones(n_arms)   # Falhas iniciais (prior uniforme)

    def select_action(self) -> int:
        """Amostra da distribuição Beta de cada braço e escolhe o de maior valor."""
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        return int(np.argmax(samples))

    def update(self, action: int, reward: int):
        """Atualiza a distribuição do braço escolhido com base na resposta (0 ou 1)."""
        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1

    def get_expected_rewards(self) -> list:
        """Retorna a taxa média de conversão esperada para cada oferta."""
        return [self.alpha[i] / (self.alpha[i] + self.beta[i]) for i in range(self.n_arms)]