# Previsão de LTV (Lifetime Value) com Regressão

Projeto de Machine Learning supervisionado para estimar o **LTV** (valor que um cliente gera ao longo do tempo, em R$) de uma empresa de educação online, usando apenas informações disponíveis **no momento da primeira compra**.

> **Crédito:** este projeto é o case prático *Hashtag Lifetime Value*, do Módulo III (Data Analysis and Predictive Insights) da pós-graduação em **Data Analytics e IA Aplicada a Negócios (FNAT / Hashtag Treinamentos)**, com o Prof. João Lira. Os scripts de tratamento, validação cruzada, modelagem e uso prático seguem a solução de referência apresentada no curso. A análise exploratória e o cálculo dos coeficientes para a planilha foram escritos em Python a partir das análises feitas no case. Tudo está organizado aqui como estudo e portfólio.

---

## Problema de negócio

A empresa depende de mídia paga e campanhas digitais para adquirir clientes. Ao prever o LTV logo na primeira compra, é possível:

- priorizar **canais e produtos** que trazem clientes de maior retorno;
- otimizar o **investimento em aquisição** (quanto vale a pena pagar por um cliente de cada perfil).

**Tipo de tarefa:** regressão supervisionada. O alvo (`LTV`) é uma variável contínua em reais.

## Metodologia (CRISP-DM)

| Etapa | O que foi feito |
|---|---|
| 1. Entendimento do negócio | Definição do alvo (LTV), da unidade de análise (cliente) e das métricas de sucesso |
| 2. Entendimento dos dados | Tipos, estatística descritiva, nulos "disfarçados" (`0`, `Prefiro não informar`), assimetria do LTV e correlações (`src/analise_exploratoria.py`) |
| 3. Preparação dos dados | Limpeza, conversão de tipos, redução de cardinalidade e engenharia de atributos de data (`src/tratamento.py`) |
| 4. Modelagem | Validação cruzada (K-Fold = 5) de 4 modelos dentro de um `Pipeline` do scikit-learn (`src/validacao_cruzada.py`) |
| 5. Avaliação | R², RMSE e MAE no conjunto de teste (20% dos dados) (`src/modelagem.py`) |
| 6. Deploy | Modelo salvo com `joblib` e usado para prever novos clientes (`src/uso_pratico.py`); coeficientes em reais para uma calculadora em planilha (`src/coeficientes_deploy.py`) |

## Principais achados da análise exploratória

- **LTV assimétrico à direita** (skewness 1,02; média R$ 2.192 vs. mediana R$ 2.144).
- **`valor_1_compra` é a variável mais forte** (correlação de Pearson 0,91 com o LTV).
- **Recorrência na 1ª compra reduz o LTV** (correlação -0,73): quem entra por assinatura tem LTV médio de R$ 1.064, contra R$ 3.005 de quem compra à vista. Isso acontece porque a 1ª compra recorrente tem valor baixo.
- `valor_1_compra` e `recorrente_1_compra` são correlacionadas entre si (-0,77), o que gera multicolinearidade. Ela foi mantida, como no curso, porque o objetivo é previsão.
- Entre as categóricas, **Produto Fonte** é a que mais explica o LTV (eta 0,48). `Renda` praticamente não tem relação com o alvo (0,01) e foi descartada.

## Dados

A base tem **38.753 clientes** e 10 colunas originais, com compras de jan/2023 a mai/2024:

`ID`, `LTV` (alvo), `data_compra`, `valor_1_compra`, `recorrente_1_compra`, `Produto Fonte`, `Fonte Campanha`, `Sexo`, `Formacao`, `Renda`

> **Os dados não estão neste repositório**, pois são material do curso. Para rodar o projeto, coloque o arquivo `ltv_base.csv` na pasta `data/`. O formato esperado está descrito em [`data/README.md`](data/README.md).

### Principais tratamentos (`src/tratamento.py`)

- Remoção de duplicatas por `ID` e das colunas `ID` e `Renda` (sem poder preditivo relevante).
- Conversão de `LTV` e `valor_1_compra` (texto com vírgula) para número, e de `data_compra` (número serial do Excel) para data.
- **Redução de cardinalidade:**
  - `Sexo` → Masculino / Feminino / Outros
  - `Formacao` → Superior+ / Superior Completo / Superior Incompleto / Médio / Fundamental / Não informado
  - `Produto Fonte` → unificação de "PBI" com "Power BI"; produtos com menos de 400 clientes viram "Outros"
- Extração de `mes_compra` e `dia_semana_compra` a partir da data, que depois é descartada.

## Modelagem

Pré-processamento com `ColumnTransformer`:

- `StandardScaler` em `valor_1_compra`;
- `OneHotEncoder` (drop first) nas categóricas, incluindo `mes_compra` tratado como categoria;
- `recorrente_1_compra` mantida como está (já é 0/1).

O split treino/teste (80/20, `random_state=42`) é feito **antes** de ajustar o pipeline, para evitar vazamento de dados (*data leakage*).

### Validação cruzada (K-Fold = 5, no treino)

| Modelo | R² médio | Desvio padrão |
|---|---:|---:|
| DummyRegressor (média) | -0,001 | 0,001 |
| **Regressão Linear** | **0,850** | 0,005 |
| Regressão Polinomial (grau 2) + Ridge | 0,847 | 0,005 |
| Random Forest | 0,825 | 0,008 |

A regressão polinomial pura (sem regularização) ficou instável por causa da multicolinearidade criada pelas features ao quadrado e pelas interações. Por isso foi usada com **Ridge** (regularização L2).

### Resultados no conjunto de teste

| Modelo | R² | RMSE (R$) | MAE (R$) |
|---|---:|---:|---:|
| DummyRegressor (média, baseline) | 0,000 | 1.301,31 | 1.024,58 |
| **Regressão Linear** | **0,848** | **507,74** | **404,03** |
| Regressão Polinomial (grau 2) + Ridge | 0,846 | 511,23 | 406,70 |
| Random Forest | 0,823 | 547,51 | 435,15 |

**Conclusão:** a Regressão Linear teve o melhor desempenho e é também o modelo mais simples e interpretável. Ela explica cerca de 85% da variação do LTV e erra, em média, ~R$ 404 por cliente, contra ~R$ 1.025 do baseline. Por isso foi o modelo escolhido para o deploy.

### Deploy em planilha (`src/coeficientes_deploy.py`)

Para a diretoria testar o modelo numa planilha, digitando o valor da 1ª compra direto em reais, o `StandardScaler` foi "desfeito" nos coeficientes:

$$LTV = \underbrace{\left(\beta_0 - \frac{\beta_{valor}\,\mu}{\sigma}\right)}_{\text{intercepto ajustado}} + \underbrace{\frac{\beta_{valor}}{\sigma}}_{\text{coef. em R\$}} \cdot x + \sum \beta_i X_i$$

- Média (μ) da 1ª compra no treino: R$ 715,11. Desvio padrão (σ): R$ 665,91.
- **Cada R$ 1,00 a mais na 1ª compra soma R$ 1,52 ao LTV previsto.**
- Intercepto ajustado: **R$ 1.505,89**, que é o LTV do cliente "base" (todas as categorias de referência e valor zero), a partir do qual os coeficientes de cada categoria somam ou subtraem.

O script gera a tabela completa em `models/coeficientes_ltv.csv`.

### Exemplo de uso (`src/uso_pratico.py`)

| Produto Fonte | 1ª compra (R$) | LTV previsto (R$) |
|---|---:|---:|
| Python | 97,14 | 1.079,86 |
| Ciência de Dados | 1.165,68 | 3.324,03 |
| Power BI | 997,00 | 3.017,50 |
| Excel | 297,00 | 939,69 |

## Como executar

```bash
pip install -r requirements.txt

# coloque ltv_base.csv e novos_clientes.csv em data/
python src/analise_exploratoria.py  # etapa 2: análise exploratória
python src/tratamento.py            # etapa 3: gera data/ltv_base_tratada.csv
python src/validacao_cruzada.py     # etapa 4: K-Fold dos 4 modelos
python src/modelagem.py             # etapa 5: avaliação no teste e salva o modelo
python src/coeficientes_deploy.py   # etapa 6: coeficientes em R$ para a planilha
python src/uso_pratico.py           # etapa 6: prevê o LTV dos novos clientes
```

Os comandos devem ser rodados a partir da raiz do repositório.

## Estrutura

```
LTV-Prediction-Regression/
├── data/            # dados (não versionados), ver data/README.md
├── models/          # modelo treinado (gerado ao rodar a modelagem)
├── src/
│   ├── analise_exploratoria.py
│   ├── tratamento.py
│   ├── validacao_cruzada.py
│   ├── modelagem.py
│   ├── coeficientes_deploy.py
│   └── uso_pratico.py
├── requirements.txt
└── README.md
```

## Tecnologias

Python · pandas · NumPy · scikit-learn · joblib

---

**Autor:** Felipe Viegas Rodrigues
