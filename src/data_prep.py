import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """Carrega a base bruta a partir do caminho indicado."""
    df = pd.read_csv(filepath, sep=';' if ';' in open(filepath).readline() else ',')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa a base de dados:
    - Remove a coluna 'duration' (Vazamento Temporal / Data Leakage)
    - Converte a variável alvo 'y' em binário (1 = sim/conversão, 0 = não)
    """
    df_clean = df.copy()
    
    # 1. Remoção OBRIGATÓRIA de vazamento temporal (Data Leakage)
    if 'duration' in df_clean.columns:
        df_clean = df_clean.drop(columns=['duration'])
        
    # 2. Tratamento da variável alvo 'y' ou 'deposit'
    target_col = 'y' if 'y' in df_clean.columns else 'deposit'
    if target_col in df_clean.columns:
        df_clean['target'] = df_clean[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1', 'success'] else 0)
        if target_col != 'target':
            df_clean = df_clean.drop(columns=[target_col])
            
    return df_clean

def preprocess_features(df: pd.DataFrame):
    """
    Prepara as features categóricas e numéricas para modelagem/contexto.
    """
    df_proc = df.copy()
    target = df_proc.pop('target')
    
    # Codificação de variáveis categóricas
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col])
        encoders[col] = le
        
    return df_proc, target, encoders

if __name__ == "__main__":
    # Teste de execução do pipeline
    raw_path = "data/raw/bank-marketing.csv"
    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)
    df_clean.to_csv("data/processed/bank_cleaned.csv", index=False)
    print("Dataset limpo salvo em data/processed/bank_cleaned.csv")