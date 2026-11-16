import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Carregamento e divisão dos dados
df = pd.read_csv('data/ltv_base_tratada.csv')
X = df.drop(columns=['LTV'])
y = df['LTV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2. Definição do Preprocessador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['valor_1_compra']),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), 
         ['Produto Fonte', 'Fonte Campanha', 'Sexo', 'Formacao', 'dia_semana_compra', 'mes_compra']),
        ('pass', 'passthrough', ['recorrente_1_compra'])
    ]
)

# 3. Pipelines dos Modelos
pipelines = {
    'DummyRegressor (Média)': Pipeline([
        ('prep', preprocessor),
        ('model', DummyRegressor(strategy='mean'))
    ]),
    'Regressão Linear': Pipeline([
        ('prep', preprocessor),
        ('model', LinearRegression())
    ]),
    'Regressão Polinomial (g=2) + Ridge': Pipeline([
        ('prep', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', Ridge(alpha=1.0, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
}

# 4. Treinamento e Avaliação no Conjunto de Teste
test_results = []

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    test_results.append({
        'Modelo': name,
        'R²': r2_score(y_test, y_pred),
        'RMSE (R$)': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE (R$)': mean_absolute_error(y_test, y_pred)
    })

df_test = pd.DataFrame(test_results)
print(df_test.to_string(index=False))

# 5. Perpetuação (salvar) do modelo de Regressão Linear
import joblib

modelo_final = pipelines['Regressão Linear']
joblib.dump(modelo_final, 'models/modelo_regressao_linear.pkl')
print('\nModelo de Regressão Linear salvo em: models/modelo_regressao_linear.pkl')