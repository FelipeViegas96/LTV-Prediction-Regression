import pandas as pd
import joblib

# 1. Carregar o modelo treinado e os novos clientes
modelo = joblib.load('models/modelo_regressao_linear.pkl')
novos_clientes = pd.read_csv('data/novos_clientes.csv')

# 2. Fazer a previsão de LTV
novos_clientes['LTV_previsto'] = modelo.predict(novos_clientes)

# 3. Exibir o resultado
print(novos_clientes[['Produto Fonte', 'valor_1_compra', 'LTV_previsto']].to_string(index=False))
novos_clientes.to_csv("data/novos_clientes_com_ltv.csv", index=False)