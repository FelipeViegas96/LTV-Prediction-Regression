# Dados

Os arquivos de dados não são versionados, porque são material do curso. Para rodar o projeto, coloque aqui:

## `ltv_base.csv` (base de treino)

| Coluna | Descrição | Exemplo |
|---|---|---|
| `ID` | Identificador do cliente | `CL0001` |
| `LTV` | Valor total gerado pelo cliente (alvo), texto com vírgula decimal | `"1049,97"` |
| `data_compra` | Data da 1ª compra em número serial do Excel | `"44927,41"` |
| `valor_1_compra` | Valor da 1ª compra, texto com vírgula decimal | `"97,14"` |
| `recorrente_1_compra` | 1 se a 1ª compra foi recorrente (assinatura), 0 se não | `1` |
| `Produto Fonte` | Produto da 1ª compra | `SQL` |
| `Fonte Campanha` | Canal/campanha de aquisição | `Lista Espera` |
| `Sexo` | Sexo informado (`0` = não informado) | `Feminino` |
| `Formacao` | Escolaridade informada | `Superior Completo` |
| `Renda` | Faixa de renda (descartada no tratamento) | `0` |

## `novos_clientes.csv` (para `uso_pratico.py`)

Mesmas colunas da base **já tratada**, sem `LTV`:

`valor_1_compra, recorrente_1_compra, Produto Fonte, Fonte Campanha, Sexo, Formacao, mes_compra, dia_semana_compra`
