import pandas as pd

# Carrega o CSV (supondo que ele esteja na mesma pasta ou especificando o caminho completo)
df = pd.read_csv("data/EV-charging.csv")

# Número total de linhas
n = len(df)

# Define os índices para os cortes
train_end = int(n * 0.7)
test_end = train_end + int(n * 0.2)  # 20% dos dados

# Cria os dataframes de treinamento e teste
df_train = df.iloc[:train_end]
df_test = df.iloc[train_end:test_end]

# Salva os dataframes em arquivos CSV
df_train.to_csv("data/EV-charging_train.csv", index=False)
df_test.to_csv("data/EV-charging_test.csv", index=False)

print("Divisão realizada: treinamento com", len(df_train), "linhas e teste com", len(df_test), "linhas.")
