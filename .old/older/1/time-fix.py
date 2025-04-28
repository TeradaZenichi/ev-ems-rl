import pandas as pd

# Carregar o CSV
df = pd.read_csv('data/EV-charging.csv', sep=',')

# Converter a coluna 'timestamp' para datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Adicionar 1 minuto a cada valor da coluna 'timestamp'
df['timestamp'] = df['timestamp'] + pd.Timedelta(minutes=1)

# Salvar o CSV modificado
df.to_csv('data/EV-charging_modified.csv', index=False)

print("Arquivo modificado salvo como 'EV-charging_modified.csv'")
