import pandas as pd

# Carregar o arquivo CSV
file_path = "charging-ev/Electric_Vehicle_Charging_Station_Data_-8671638762898357044.csv.xls"
df = pd.read_csv(file_path, sep=",", encoding="utf-8")

# Converter as colunas de datas para datetime
df["Start_Date___Time"] = pd.to_datetime(df["Start_Date___Time"], errors="coerce")
df["End_Date___Time"] = pd.to_datetime(df["End_Date___Time"], errors="coerce")

# Remover registros com datas inválidas, se houver
df = df.dropna(subset=["Start_Date___Time", "End_Date___Time"])

# Identificar o menor e o maior horário na série
start_time = min(df["Start_Date___Time"].min(), df["End_Date___Time"].min())
end_time   = max(df["Start_Date___Time"].max(), df["End_Date___Time"].max())

print("Menor horário:", start_time)
print("Maior horário:", end_time)

# Criar uma linha do tempo de 5 em 5 minutos
timeline = pd.date_range(start=start_time, end=end_time, freq="5T")
timeline_df = pd.DataFrame({"timestamp": timeline})
timeline_df.set_index("timestamp", inplace=True)

# Identificar todas as estações de carga únicas
stations = df["Station_Name"].unique()

# Para cada estação, vamos criar uma coluna preenchida inicialmente com 0
for station in stations:
    timeline_df[station] = 0

# Para cada registro de carregamento, preencher com 1 o intervalo correspondente
for idx, row in df.iterrows():
    station = row["Station_Name"]
    start_charging = row["Start_Date___Time"]
    end_charging = row["End_Date___Time"]
    
    # Seleciona os timestamps entre o início e fim do carregamento
    mask = (timeline_df.index >= start_charging) & (timeline_df.index <= end_charging)
    
    # Atribuir 1 para o período em que o carregamento ocorreu
    timeline_df.loc[mask, station] = 1

# Resetar o índice para que a coluna timestamp fique no dataframe final
timeline_df = timeline_df.reset_index()

# Exibir as primeiras linhas do dataframe resultante
print(timeline_df.head(10))

# Salvar o dataframe resultante em CSV separado por vírgula
output_file_path = "data/processed_charging_timeline.csv"
timeline_df.to_csv(output_file_path, index=False, sep=",", encoding="utf-8")

print("Arquivo salvo em:", output_file_path)
