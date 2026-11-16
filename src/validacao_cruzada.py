"""
Etapa 4 do CRISP-DM - Modelagem: validação cruzada (K-Fold = 5) no conjunto de treino.

Compara os 4 modelos pelo R² em 5 folds antes de olhar para os dados de teste.
A regressão polinomial usa Ridge (regularização L2) para controlar a multicolinearidade
gerada pelas features ao quadrado e pelas interações.

Executar a partir da raiz do repositório:  python src/validacao_cruzada.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)

# 1. Carregamento dos dados
df = pd.read_csv('data/ltv_base_tratada.csv')

X = df.drop(columns=['LTV'])
y = df['LTV']

# 2. Divisão Treino/Teste (80/20 com seed 42) - a validação cruzada usa só o treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Preprocessador dentro do Pipeline (é ajustado de novo em cada fold, sem vazamento)
num_cols = ['valor_1_compra']
cat_cols = ['Produto Fonte', 'Fonte Campanha', 'Sexo', 'Formacao', 'dia_semana_compra', 'mes_compra']
pass_cols = ['recorrente_1_compra']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols),
        ('pass', 'passthrough', pass_cols)
    ]
)

# 4. Configuração da Validação Cruzada (K-Fold = 5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5. Definição dos 4 Pipelines
pipelines = {
    '1. DummyRegressor (Média)': Pipeline([
        ('prep', preprocessor),
        ('model', DummyRegressor(strategy='mean'))
    ]),
    '2. Regressão Linear': Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ]),
    '3. Regressão Polinomial (g=2) + Ridge': Pipeline([
        ('prep', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', Ridge(alpha=1.0, random_state=42))
    ]),
    '4. Random Forest': Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

# 6. Execução e Coleta dos Resultados de R²
results = {}
for name, pipe in pipelines.items():
    results[name] = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')

# 7. Tabela Resultante dos 5 Folds
df_results = pd.DataFrame(results, index=[f'Fold {i}' for i in range(1, 6)])

print("=== TABELA DE RESULTADOS POR FOLD (R²) ===")
print(df_results.round(6))

print("\n=== RESUMO (MÉDIA E DESVIO PADRÃO) ===")
summary = pd.DataFrame({
    'Média R²': df_results.mean(),
    'Desvio Padrão': df_results.std()
})
print(summary.round(6))
