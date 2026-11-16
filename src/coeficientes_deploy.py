"""
Etapa 6 do CRISP-DM - Deploy em planilha: coeficientes da Regressão Linear em reais.

O modelo foi treinado com valor_1_compra padronizado (z = (x - média) / desvio).
Para usar a fórmula numa planilha digitando o valor direto em R$, desfazemos o StandardScaler:

    LTV = b0 + b_valor * (x - mu) / sigma + soma(b_i * X_i)
        = [b0 - b_valor * mu / sigma] + [b_valor / sigma] * x + soma(b_i * X_i)
           \\___ intercepto ajustado __/   \\_ coef. em R$ _/

As categorias de referência (removidas pelo drop='first') têm coeficiente 0.

Pré-requisito: rodar antes  python src/modelagem.py  (gera o modelo salvo).
Executar a partir da raiz do repositório:  python src/coeficientes_deploy.py
"""
import joblib
import pandas as pd

modelo = joblib.load('models/modelo_regressao_linear.pkl')
prep = modelo.named_steps['prep']
reg = modelo.named_steps['model']

# Nomes das colunas após o pré-processamento, alinhados com os coeficientes
nomes = prep.get_feature_names_out()
coefs = pd.Series(reg.coef_, index=nomes)

# Parâmetros do StandardScaler aprendidos no treino
scaler = prep.named_transformers_['num']
mu, sigma = scaler.mean_[0], scaler.scale_[0]
b_valor = coefs['num__valor_1_compra']

intercepto_ajustado = reg.intercept_ - b_valor * mu / sigma
coef_valor_reais = b_valor / sigma

print('=== DESPADRONIZAÇÃO DO valor_1_compra ===')
print(f'Média (mu):            R$ {mu:,.2f}')
print(f'Desvio padrão (sigma): R$ {sigma:,.2f}')
print(f'Coeficiente em R$:     {coef_valor_reais:.6f}  (cada R$ 1 a mais na 1ª compra soma isso ao LTV)')
print(f'Intercepto ajustado:   R$ {intercepto_ajustado:,.2f}\n')

# Tabela de coeficientes pronta para a planilha (variável | categoria | coeficiente)
linhas = [('Intercepto', '-', intercepto_ajustado),
          ('valor_1_compra', 'por R$ 1', coef_valor_reais)]
for nome, valor in coefs.items():
    if nome == 'num__valor_1_compra':
        continue
    bloco, resto = nome.split('__', 1)
    if bloco == 'pass':
        linhas.append((resto, '= 1', valor))
    else:
        # OneHotEncoder gera "coluna_categoria"; a coluna é o prefixo conhecido
        for col in prep.transformers_[1][2]:
            if resto.startswith(col + '_'):
                linhas.append((col, resto[len(col) + 1:], valor))
                break

tabela = pd.DataFrame(linhas, columns=['variavel', 'categoria', 'coeficiente'])
print('=== TABELA DE COEFICIENTES (em reais) ===')
print(tabela.round(4).to_string(index=False))

# Categorias de referência (coeficiente 0) - o cliente "base" da fórmula
encoder = prep.named_transformers_['cat']
refs = {col: str(cats[0]) for col, cats in zip(prep.transformers_[1][2], encoder.categories_)}
print('\nCategorias de referência (coeficiente 0):', refs)

tabela.to_csv('models/coeficientes_ltv.csv', index=False)
print('\nTabela salva em: models/coeficientes_ltv.csv')
