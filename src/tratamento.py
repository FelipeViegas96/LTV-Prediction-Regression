import pandas as pd

# 1. Carregar a base original
df = pd.read_csv('data/ltv_base.csv')

# 2. Remover duplicatas pelo ID
df.drop_duplicates(subset=['ID'], keep='first', inplace=True)

# 3. Remover colunas desnecessárias
df.drop(columns=['Renda', 'ID'], inplace=True, errors='ignore')

# 4. Tratar tipos numéricos
df['LTV'] = df['LTV'].str.replace(',', '.').astype(float)
df['valor_1_compra'] = df['valor_1_compra'].str.replace(',', '.').astype(float)

# 5. Tratar data de compra
df['data_compra'] = pd.to_numeric(df['data_compra'].str.replace(',', '.'), errors='coerce')
df['data_compra'] = pd.to_datetime(df['data_compra'], unit='D', origin='1899-12-30')

# 6. Redução de cardinalidade - Sexo
df['Sexo'] = df['Sexo'].apply(lambda val: val if val in ['Masculino', 'Feminino'] else 'Outros')

# 7. Redução de cardinalidade - Formação
def tratar_formacao(val):
    if pd.isna(val) or val in ['0', 'Prefiro não informar']:
        return 'Não informado'
    elif val in ['Doutorado', 'Mestrado', 'Pós-Graduação']:
        return 'Superior+'
    elif val in ['Superior Completo', 'Técnico Completo']:
        return 'Superior Completo'
    elif val in ['Superior Incompleto', 'Técnico Incompleto']:
        return 'Superior Incompleto'
    elif val in ['Médio Completo', 'Médio Incompleto']:
        return 'Médio'
    else:
        return 'Fundamental'

df['Formacao'] = df['Formacao'].apply(tratar_formacao)

# 8. Redução de cardinalidade - Produto Fonte
df['Produto Fonte'] = df['Produto Fonte'].replace({'PBI': 'Power BI'})
produtos_frequentes = df['Produto Fonte'].value_counts()[df['Produto Fonte'].value_counts() >= 400].index
df['Produto Fonte'] = df['Produto Fonte'].apply(lambda x: x if x in produtos_frequentes else 'Outros')

# Extrair o mes e o dia da semana
df['mes_compra'] = df['data_compra'].dt.month
df['dia_semana_compra'] = df['data_compra'].dt.day_name()

# Traduzir os nomes dos dias da semana
dias_pt = {
    'Monday': 'Segunda-feira',
    'Tuesday': 'Terça-feira',
    'Wednesday': 'Quarta-feira',
    'Thursday': 'Quinta-feira',
    'Friday': 'Sexta-feira',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
}
df['dia_semana_compra'] = df['dia_semana_compra'].map(dias_pt)

# Drop data_compra
df = df.drop(columns=['data_compra'])


# 9. Salvar arquivo final
df.to_csv('data/ltv_base_tratada.csv', index=False)