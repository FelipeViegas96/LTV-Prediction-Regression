"""
Etapa 2 do CRISP-DM - Entendimento dos dados (análise exploratória).

Reproduz as análises pedidas nesta etapa do case:
  1. Amostra da base
  2. Tipo de dado de cada coluna
  3. Estatística descritiva
  4. Valores vazios, incluindo os "disfarçados" ('0', 'Prefiro não informar', 'Não informado')
  5. LTV médio por Produto Fonte, Fonte Campanha e recorrente_1_compra
  6. Assimetria e curtose do LTV
  7. Correlações (LTV x features numéricas e eta para as categóricas)

Executar a partir da raiz do repositório:  python src/analise_exploratoria.py
"""
import numpy as np
import pandas as pd

pd.set_option('display.width', 140)
pd.set_option('display.max_columns', 20)

# Carregar a base original e converter os campos numéricos que vêm como texto com vírgula
df = pd.read_csv('data/ltv_base.csv')
for col in ['LTV', 'valor_1_compra']:
    df[col] = df[col].str.replace(',', '.').astype(float)

# 1. Amostra
print('=== 1. AMOSTRA DA BASE ===')
print(df.head(), '\n')

# 2. Tipos de dados
print('=== 2. TIPOS DE DADOS ===')
print(df.dtypes.to_frame('tipo'), '\n')

# 3. Estatística descritiva (numéricas e categóricas separadas)
print('=== 3. ESTATÍSTICA DESCRITIVA ===')
print(df.describe().round(2), '\n')
print(df.describe(include='object'), '\n')

# 4. Vazios explícitos (NaN) e vazios "disfarçados" em forma de texto/código
print('=== 4. VALORES VAZIOS ===')
placeholders = ['0', 0, 'Prefiro não informar', 'Não informado']
vazios = pd.DataFrame({
    'nulos (NaN)': df.isna().sum(),
    'placeholders': df.isin(placeholders).sum(),
})
vazios['% total'] = ((vazios.sum(axis=1)) / len(df) * 100).round(2)
# recorrente_1_compra = 0 é informação válida (não recorrente), não é vazio
vazios.loc['recorrente_1_compra', ['placeholders', '% total']] = [0, 0]
print(vazios, '\n')

# 5. LTV médio por feature
print('=== 5. LTV MÉDIO POR FEATURE ===')
for col in ['Produto Fonte', 'Fonte Campanha', 'recorrente_1_compra']:
    tabela = (df.groupby(col)['LTV']
                .agg(clientes='count', ltv_medio='mean')
                .sort_values('ltv_medio', ascending=False)
                .round(2))
    print(f'--- {col} ---')
    print(tabela, '\n')

# 6. Assimetria do LTV
print('=== 6. ASSIMETRIA DO LTV ===')
print(f"Média:    R$ {df['LTV'].mean():,.2f}")
print(f"Mediana:  R$ {df['LTV'].median():,.2f}")
print(f"Skewness: {df['LTV'].skew():.2f}  (> 0 = cauda à direita)")
print(f"Curtose:  {df['LTV'].kurt():.2f}  (excesso em relação à normal)\n")

# 7. Correlações
print('=== 7. CORRELAÇÕES ===')
renda_num = pd.to_numeric(df['Renda'], errors='coerce').replace(0, np.nan)
num = df[['LTV', 'valor_1_compra', 'recorrente_1_compra']].assign(Renda=renda_num)
print('Pearson (numéricas):')
print(num.corr().round(2), '\n')


def eta(categoria, alvo):
    """Correlation ratio (eta): quanto da variância do alvo é explicada pela categoria (0 a 1)."""
    media_geral = alvo.mean()
    grupos = alvo.groupby(categoria)
    ss_entre = (grupos.count() * (grupos.mean() - media_geral) ** 2).sum()
    ss_total = ((alvo - media_geral) ** 2).sum()
    return np.sqrt(ss_entre / ss_total)


print('Eta (categóricas x LTV):')
cats = ['Produto Fonte', 'Fonte Campanha', 'Formacao', 'Sexo']
print(pd.Series({c: eta(df[c].astype(str), df['LTV']) for c in cats})
        .sort_values(ascending=False).round(2).to_frame('eta'))
