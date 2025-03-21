import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
# from tslearn.clustering import TimeSeriesKMeans, KShape
# from tslearn.metrics import cdist_dtw
# from sklearn.metrics import silhouette_score
from datetime import datetime
import seaborn as sns
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*force_all_finite.*"  # Filtra por el texto del warning
)

# Load the data
df = pd.read_csv('data/Electric_Vehicle_Charging_Station_Data_-8671638762898357044.csv')

station = 'BOULDER / N BOULDER REC 1'

# Filter the data
df_station = df[df['Station_Name'] == station]

features = ['Start_Date___Time', 'Start_Time_Zone', 'End_Date___Time',
            'End_Time_Zone', 'Total_Duration__hh_mm_ss_', 'Charging_Time__hh_mm_ss_',
            'Energy__kWh_', 'Port_Type']

df_station = df_station[features]
df_station['Start_Date___Time'] = pd.to_datetime(df_station['Start_Date___Time'], format='mixed', errors='coerce')
df_station['End_Date___Time'] = pd.to_datetime(df_station['End_Date___Time'], format='mixed', errors='coerce')
df_station['Total_Duration_Minutes'] = pd.to_timedelta(df_station['Total_Duration__hh_mm_ss_']).dt.total_seconds() / 60
df_station['Total_Charging_Minutes'] = pd.to_timedelta(df_station ['Charging_Time__hh_mm_ss_']).dt.total_seconds() / 60
df_station = df_station.drop(columns=['Total_Duration__hh_mm_ss_','Charging_Time__hh_mm_ss_'])
#df_station = df_station[df_station['Start_Date___Time'] < '2023-06-02'] # filter start time until may 2023 6508 samples

# sort by start time
df_station = df_station.sort_values(by='Start_Date___Time')

df_station = df_station.drop_duplicates(keep='first')
df_station = df_station[df_station['Total_Charging_Minutes']>0]
df_station = df_station[df_station['Energy__kWh_']>0]
df_station = df_station.reset_index(drop=True) # restart index

# Convertir a datetime y redondear (floor) a minutos completos,
# ya que la base de datos original solo tiene horas y minutos.

df_station['Start_Date___Time'] = pd.to_datetime(df_station['Start_Date___Time'], errors='coerce')
df_station['End_Date___Time'] = pd.to_datetime(df_station['End_Date___Time'], errors='coerce')

# Redondeamos los tiempos a minutos completos
df_station['Start_Date___Time'] = df_station['Start_Date___Time'].dt.floor('min')
df_station['End_Date___Time'] = df_station['End_Date___Time'].dt.floor('min')

# -----------------------------
# Expansión de sesiones a nivel de minuto: Charging Window
# -----------------------------
# Para cada sesión, usamos np.ceil para obtener el número de minutos completos en que se
# realiza la carga y distribuimos la energía uniformemente en esos intervalos.

charging_expanded = []
for idx, row in df_station.iterrows():
    start_time = row['Start_Date___Time']
    charging_minutes = row['Total_Charging_Minutes']
    total_kwh = row['Energy__kWh_']
    
    # Número de minutos completos necesarios (si es fraccional, se redondea hacia arriba)
    n_minutes = int(np.ceil(charging_minutes))
    
    # Generar un rango de tiempo con intervalos de 1 minuto, empezando en start_time
    rng = pd.date_range(start=start_time, periods=n_minutes, freq='min')
    
    if n_minutes == 0:
        continue
        
    # Distribuir la energía de forma uniforme entre los intervalos completos
    energy_per_min = total_kwh / n_minutes
    
    df_tmp = pd.DataFrame({
        'energy_kWh': energy_per_min,
        'session_flag': 1  # Indica que en ese minuto la sesión está activa
    }, index=rng)
    df_tmp['session_id'] = idx
    charging_expanded.append(df_tmp)

# Concatenar todas las sesiones expandidas
charging_expanded_df = pd.concat(charging_expanded)

# Crear la columna 'date' usando la fecha del índice (que ya está en minutos completos)
charging_expanded_df['date'] = charging_expanded_df.index.date

# Como los timestamps ya son de minuto completo, asignamos 'minute_floor' igual al índice
charging_expanded_df['minute_floor'] = charging_expanded_df.index

# -----------------------------
# Expansión de sesiones a nivel de minuto: Duración Total
# -----------------------------
# Se expande la ventana total de la sesión (desde el inicio hasta que termina la sesión)
# usando el mismo criterio de intervalos completos.

duration_expanded = []
for idx, row in df_station.iterrows():
    start_time = row['Start_Date___Time']
    duration_minutes = row['Total_Duration_Minutes']
    
    n_minutes = int(np.ceil(duration_minutes))
    rng = pd.date_range(start=start_time, periods=n_minutes, freq='min')
    
    if n_minutes == 0:
        continue
        
    df_tmp = pd.DataFrame({
        'duration_active': 1
    }, index=rng)
    df_tmp['session_id'] = idx
    duration_expanded.append(df_tmp)
    
duration_expanded_df = pd.concat(duration_expanded)
duration_expanded_df['date'] = duration_expanded_df.index.date

# -----------------------------
# Cálculo de métricas diarias (features)
# -----------------------------

# 1. Volume: Suma total de energía (kWh) cargada en el día
daily_volume = charging_expanded_df.groupby('date')['energy_kWh'].sum().rename('Volume')

# 2. Total_Charging_Minutes: Suma de minutos de carga (cada minuto cuenta)
daily_total_charging = charging_expanded_df.groupby('date').size().rename('Total_Charging_Minutes')

# 3. Total_Duration_Minutes: Suma de minutos de la duración total de las sesiones
daily_total_duration = duration_expanded_df.groupby('date').size().rename('Total_Duration_Minutes')

# 4. Occupancy: Fracción de minutos en el día en que hubo al menos una carga activa.
# Se usa la columna 'minute_floor' para evitar duplicados por segundos.
unique_charging_minutes = charging_expanded_df.groupby('date')['minute_floor'].nunique()
daily_occupancy = (unique_charging_minutes / 1440).rename('Occupancy')  # 1440 minutos en un día

# 5. Max_Sessions: Número máximo de sesiones simultáneas en cualquier minuto del día
sessions_per_min = charging_expanded_df.groupby('minute_floor').size()
max_sessions = sessions_per_min.groupby(sessions_per_min.index.date).max()
max_sessions = pd.Series(max_sessions, name='Max_Sessions')

# 6. Máximo Volumen Instantáneo, Hora de inicio del pico y Duración del pico
# Se agrupa por minuto sumando las energías (para capturar solapamientos)
vol_per_min = charging_expanded_df.groupby('minute_floor')['energy_kWh'].sum()
minute_agg = pd.DataFrame({
    'vol_per_min': vol_per_min,
    'sessions': charging_expanded_df.groupby('minute_floor').size()
})
minute_agg['date'] = minute_agg.index.date

# Función auxiliar para calcular la duración (en minutos) de la secuencia consecutiva
# en que se mantiene el valor máximo a partir de la primera ocurrencia.
def compute_run_length(series, target):
    run = 0
    for value in series:
        if np.isclose(value, target):
            run += 1
        else:
            break
    return run

daily_max_vol_dict = {}
daily_time_max_vol_dict = {}
daily_duration_vol_max_dict = {}

for day, group in minute_agg.groupby('date'):
    max_vol = group['vol_per_min'].max()
    daily_max_vol_dict[day] = max_vol
    
    # Se obtiene el primer minuto en que ocurre este valor máximo
    max_times = group[group['vol_per_min'] == max_vol]
    first_max_time = max_times.index[0]
    
    # Convertir la hora a formato decimal (hora + minuto/60)
    dt = first_max_time
    time_decimal = dt.hour + dt.minute/60 + dt.second/3600
    daily_time_max_vol_dict[day] = time_decimal
    
    # Calcular la duración (en minutos) de la secuencia consecutiva con el valor máximo
    group_sorted = group.sort_index()
    series_from_max = group_sorted.loc[first_max_time:]['vol_per_min']
    run_length = compute_run_length(series_from_max, max_vol)
    daily_duration_vol_max_dict[day] = run_length

daily_max_vol = pd.Series(daily_max_vol_dict, name='Max_Volumen')
daily_time_max_vol = pd.Series(daily_time_max_vol_dict, name='Time_Max_Vol')
daily_duration_vol_max = pd.Series(daily_duration_vol_max_dict, name='Duration_Vol_Max')

# 7. Day_Type: Clasifica cada día como día laborable (WD) o no (NWD)
# Se definen feriados fijos (por ejemplo: Año Nuevo, 25 de marzo, Navidad)
holidays = [(1, 1), (3, 25), (12, 25)]
def get_day_type(date_obj):
    if date_obj.weekday() >= 5 or (date_obj.month, date_obj.day) in holidays:
        return 'NWD'
    else:
        return 'WD'
day_type = pd.Series({d: get_day_type(d) for d in daily_volume.index}, name='Day_Type')

# 8. Sessions_Count: Número de sesiones que iniciaron en cada día
df_station['date'] = df_station['Start_Date___Time'].dt.date
daily_session_count = df_station.groupby('date').size().rename('Sessions_Count')

# -----------------------------
# Compilación de todas las métricas diarias en un solo DataFrame
# -----------------------------
daily_features = pd.concat([
    daily_volume,
    daily_total_charging,
    daily_total_duration,
    daily_occupancy,
    max_sessions,
    daily_max_vol,
    daily_time_max_vol,
    daily_duration_vol_max,
    day_type,
    daily_session_count
], axis=1).sort_index()

print(daily_features.describe())
# save the data
daily_features.to_csv('data/daily_features.csv')

# 1. Convertir Day_Type a variable numérica: WD -> 0, NWD -> 1
daily_features_numeric = daily_features.copy()
daily_features_numeric['Day_Type_Num'] = daily_features_numeric['Day_Type'].map({'WD': 0, 'NWD': 1})

# 2. Seleccionar solo las columnas numéricas para el análisis de correlación
numeric_cols = daily_features_numeric.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

# Imprimir la matriz de correlaciones
print("Matriz de correlaciones:")
print(corr_matrix)

# 3. Graficar el heatmap de correlaciones con Seaborn
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Heatmap de correlaciones de las features diarias")
plt.show()


# Ensure the index is a DatetimeIndex
daily_features.index = pd.to_datetime(daily_features.index)

# Now create a column for the day of the week
daily_features['Day_of_Week'] = daily_features.index.day_name()

# Define order for days so that the plots display them in natural order
order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily_features['Day_of_Week'] = pd.Categorical(daily_features['Day_of_Week'], categories=order_days, ordered=True)

# Create a column for Year-Month (e.g., "2018-01", "2019-01")
daily_features['Year_Month'] = daily_features.index.to_period('M').astype(str)

###########################################
# Plot 1: Time Series of Volume and Occupancy
###########################################
fig, ax1 = plt.subplots(figsize=(14, 6))
# Plot Volume over time on primary y-axis
ax1.plot(daily_features.index, daily_features['Volume'], label='Volume', color='blue', linewidth=2)
ax1.set_xlabel("Date")
ax1.set_ylabel("Volume (kWh)", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a twin y-axis for Occupancy
ax2 = ax1.twinx()
ax2.plot(daily_features.index, daily_features['Occupancy'], label='Occupancy', color='red', linewidth=2)
ax2.set_ylabel("Occupancy (fraction)", color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Volume and Occupancy Over Time")
# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
plt.show()

###########################################
# Plot 2: Boxplots by Day of the Week
###########################################
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Boxplot for Volume by day of the week
sns.boxplot(x='Day_of_Week', y='Volume', data=daily_features, ax=axs[0], order=order_days)
axs[0].set_title("Volume Distribution by Day of the Week")
axs[0].set_xlabel("")
axs[0].set_ylabel("Volume (kWh)")

# Boxplot for Occupancy by day of the week
sns.boxplot(x='Day_of_Week', y='Occupancy', data=daily_features, ax=axs[1], order=order_days)
axs[1].set_title("Occupancy Distribution by Day of the Week")
axs[1].set_xlabel("Day of the Week")
axs[1].set_ylabel("Occupancy (fraction)")

plt.tight_layout()
plt.show()

###########################################
# Plot 3: Boxplots by Year-Month
###########################################
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Boxplot for Volume by Year-Month
sns.boxplot(x='Year_Month', y='Volume', data=daily_features, ax=axs[0])
axs[0].set_title("Volume Distribution by Month-Year")
axs[0].set_xlabel("")
axs[0].set_ylabel("Volume (kWh)")
axs[0].tick_params(axis='x', rotation=90)

# Boxplot for Occupancy by Year-Month
sns.boxplot(x='Year_Month', y='Occupancy', data=daily_features, ax=axs[1])
axs[1].set_title("Occupancy Distribution by Month-Year")
axs[1].set_xlabel("Month-Year")
axs[1].set_ylabel("Occupancy (fraction)")
axs[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# Ensure the index is a DatetimeIndex
daily_features.index = pd.to_datetime(daily_features.index)

# Create a complete date range from the earliest to the latest date in the data
full_range = pd.date_range(start=daily_features.index.min(), end=daily_features.index.max(), freq='D')

# Identify which dates in the full range are missing from your DataFrame's index
missing_dates = full_range.difference(daily_features.index)

if missing_dates.empty:
    print("All days are present in the data.")
else:
    print("Missing dates:")
    print(missing_dates)
    print(f"Total missing dates: {len(missing_dates)}")
