import pandas as pd
import numpy as np



df_day = pd.read_csv('/content/daily_btc_obs_2014_2023.csv')
print(df_day.head())
print(df_day.shape)

# Converti la colonna 'Date' in datetime
df_day['Date'] = pd.to_datetime(df_day['Date'])

# Imposta la colonna 'Date' come indice
df_day.set_index('Date', inplace=True)

# Resample i dati a livello settimanale
df_week = df_day.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

print(df_week)

# Salva il DataFrame settimanale risultante in un nuovo file CSV
df_week.to_csv('weekly_observations.csv')

"""Codice strategia trading momentum

Andiamo a creare tutte le variabili che ci serviranno, saranno 18:
"""

# qua proviamo invece a isolare solo quegli ET points in cui, a seguito di un cambio di trend
# da negativo a positivo, il momentum giornaliero e  settimanale sono concordi.
import pandas as pd
import numpy as np

# Funzione per calcolare il momentum
def calculate_momentum(data, window=14):
    momentum = data['Close'].diff(window) # per ogni close calcoliamo la differenza con 14 momenti precedenti
    return momentum

# Funzione per identificare l'ET basato sul cambiamento di trend
def identify_et(data, window=14):
    data['Momentum'] = calculate_momentum(data, window)
    data['ET'] = np.where((data['Momentum'] > 0) & (data['Momentum'].shift(1) <= 0) |
                          (data['Momentum'] < 0) & (data['Momentum'].shift(1) >= 0), 1, 0)
    return data

# Funzione per generare segnali di trading basati sul momentum e sull'ET
def generate_signals(data, window=14):
    data = identify_et(data, window)
    data['Signal'] = 0  # Inizializza tutti i segnali a 0
    data['Signal'][window:] = np.where(data['Momentum'][window:] > 0, 1, -1)
    data['Position'] = data['Signal'].shift()  # Segnale di trading per il periodo successivo
    return data

# Funzione per identificare solo gli ET points
def get_et_points(data):
    et_points = data[data['ET'] == 1]
    return et_points

# Funzione per unire i dati giornalieri e settimanali e filtrare gli ET points con momentum concorde
def filter_et_points_by_momentum(et_points_daily, et_points_weekly):
    # Allineiamo i dati settimanali ai dati giornalieri
    et_points_weekly = et_points_weekly.reindex(et_points_daily.index, method='ffill')

    # Uniamo i dati giornalieri e settimanali sui punti ET
    merged_data = et_points_daily.join(et_points_weekly[['Momentum']], rsuffix='_weekly')

    # Per ora filtriamo solo i punti ET con momentum giornaliero e settimanale positivo
    filtered_et_points = merged_data[(merged_data['Momentum'] > 0) & (merged_data['Momentum_weekly'] > 0) |
                                     (merged_data['Momentum'] < 0) & (merged_data['Momentum_weekly'] < 0)]
    # Aggiungiamo una colonna che mostra 1 se i momentum sono entrambi positivi e 0 se sono entrambi negativi
    filtered_et_points['Momentum_Status'] = np.where((filtered_et_points['Momentum'] > 0) &
                                                     (filtered_et_points['Momentum_weekly'] > 0), 1, 0)
    return filtered_et_points

# Funzione principale per applicare la strategia di momentum
def apply_momentum_strategy(data_daily, data_weekly, window=14):
    # Genera segnali sui dati giornalieri
    signals_daily = generate_signals(data_daily, window)

    # Genera segnali sui dati settimanali
    signals_weekly = generate_signals(data_weekly, window)

    # Identifica gli ET points per entrambi i set di dati
    et_points_daily = get_et_points(signals_daily)
    et_points_weekly = get_et_points(signals_weekly)

    # Filtra gli ET points con momentum positivo
    filtered_et_points = filter_et_points_by_momentum(et_points_daily, et_points_weekly)

    return filtered_et_points

# Utilizzo fjunzione
# Carica i dati giornalieri e settimanali nei DataFrame
data_daily = df_day
data_weekly = df_week

# Applica la strategia di momentum con una finestra di 14 periodi
filtered_et_points = apply_momentum_strategy(data_daily, data_weekly, window=14)

# Visualizza gli ET points filtrati
print("Filtered ET Points with Positive Daily and Weekly Momentum:")
print(filtered_et_points[['Close', 'Momentum', 'Momentum_weekly', 'Momentum_Status']])
print(len(filtered_et_points))

print(filtered_et_points)
filtered_et_points.to_csv('filtered_et_points.csv')

daily_data = df_day
weekly_data = df_week
et_points = filtered_et_points
# Controlla i nomi delle colonne
print("Daily Data Columns: ", daily_data.columns)
print("Weekly Data Columns: ", weekly_data.columns)
print("ET Points Columns: ", et_points.columns)

# Carica i dataset
daily_data = pd.read_csv('/content/daily_btc_obs_2014_2023.csv')
weekly_data = pd.read_csv('/content/weekly_observations.csv')
et_points = pd.read_csv('/content/filtered_et_points.csv')

# Converti le colonne Date in formato datetime
daily_data['Date'] = pd.to_datetime(daily_data['Date'])
weekly_data['Date'] = pd.to_datetime(weekly_data['Date'])
et_points['Date'] = pd.to_datetime(et_points['Date'])

# Funzione per calcolare l'RSI
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calcolo delle variabili per ogni punto ET
for idx, row in et_points.iterrows():
    date = row['Date']
    daily_window = daily_data[daily_data['Date'] <= date].tail(18)

    # Verifica che ci siano almeno 18 periodi disponibili
    if len(daily_window) < 18:
        continue

    # Calcolo delle variabili di price action
    returns = daily_window['Close'].pct_change().dropna()
    et_points.at[idx, 's.VPT'] = ((daily_window['Close'] - daily_window['Close'].mean()) ** 2).sum() / (len(daily_window) - 1)
    et_points.at[idx, 'x.V*PPT'] = returns.mean()
    et_points.at[idx, 's.CP.PT'] = returns.std()
    et_points.at[idx, 'Sp.Vr.CP.PT'] = np.polyfit(range(len(daily_window)), daily_window['Close'], 1)[0]
    et_points.at[idx, 'Sp.Vr.HL.PT'] = np.polyfit(range(len(daily_window)), daily_window['High'] - daily_window['Low'], 1)[0]
    et_points.at[idx, 'Sp.Vr.CO.PT'] = np.polyfit(range(len(daily_window)), daily_window['Close'] - daily_window['Open'], 1)[0]

    rsi = calculate_rsi(daily_window['Close'], 14)
    et_points.at[idx, 's.Vr.RSI.PT'] = ((rsi - rsi.mean()) ** 2).sum() / (len(rsi) - 1)
    et_points.at[idx, 'Sp.Vr.RSI.PT'] = np.polyfit(range(len(rsi.dropna())), rsi.dropna(), 1)[0]

    # Calcolo delle variabili di volume
    et_points.at[idx, 's.Vr.Vo.PT'] = ((daily_window['Volume'] - daily_window['Volume'].mean()) ** 2).sum() / (len(daily_window) - 1)
    et_points.at[idx, 'x.Vr.Vo.PT'] = daily_window['Volume'].pct_change().mean()
    et_points.at[idx, 'Sp.Vr.Vo.PT'] = np.polyfit(range(len(daily_window)), daily_window['Volume'], 1)[0]
    et_points.at[idx, 'Sp.Vo.PT'] = np.polyfit(range(len(daily_window)), daily_window['Volume'], 1)[0]

    # Calcolo delle divergenze
    et_points.at[idx, 'Dv.CP-RSLIPT'] = ((daily_window['Close'] - daily_window['Close'].mean()) * (rsi - rsi.mean())).sum() / ((len(daily_window) - 1) * daily_window['Close'].std() * rsi.std())
    et_points.at[idx, 'Dv.Vo-CP.PPT'] = ((daily_window['Volume'] - daily_window['Volume'].mean()) * (daily_window['Close'] - daily_window['Close'].mean())).sum() / ((len(daily_window) - 1) * daily_window['Volume'].std() * daily_window['Close'].std())
    et_points.at[idx, 'Dv.Vo-RSLIPT'] = ((daily_window['Volume'] - daily_window['Volume'].mean()) * (rsi - rsi.mean())).sum() / ((len(daily_window) - 1) * daily_window['Volume'].std() * rsi.std())

    # Altre metriche after holding time
    et_points.at[idx, 'GPL_$'] = row['Close'] - row['Open'] if row['Position'] == 'long' else row['Open'] - row['Close']
    et_points.at[idx, 'RP% M[1]'] = (row['High'] - row['Low']) / row['Open'] * 100 if row['Position'] == 'long' else (row['Low'] - row['High']) / row['Open'] * 100
    et_points.at[idx, 'Sp.XP-EP'] = (row['Close'] - daily_window['Close'].iloc[0]) / 18

# Calcolo delle metriche di trend a lungo termine
for idx, row in et_points.iterrows():
    date = row['Date']
    weekly_window = weekly_data[weekly_data['Date'] <= date].tail(18)

    if len(weekly_window) < 18:
        continue

    et_points.at[idx, 'M.Sp.PET'] = np.polyfit(range(len(weekly_window)), weekly_window['Close'], 1)[0]
    et_points.at[idx, 'LTT'] = weekly_window['Close'].mean()
    et_points.at[idx, 'M.Sp'] = np.polyfit(range(len(weekly_window)), weekly_window['Close'], 1)[0]
# calcolo trend
et_points['Trend'] = np.where(
    (et_points['GPL_$'] >= 0) & (et_points['Sp.XP-EP'] > 0), 0,
    np.where((et_points['GPL_$'] > 0) & (et_points['Sp.XP-EP'] < 0), 1, np.nan)
)

# Salva il nuovo dataset con le variabili aggiunte
et_points.to_csv('filtered_et_points_with_variables.csv', index=False)

# Visualizza i primi risultati per verifica
et_points.head()

#le mie variabili sono tutt'altro che normalmente distribuite e non seguono le classiche assunzioni statistiche di base, vorrei fare un analisi  multidimensionale per provare a creare nuove variabili che spieghino la variabile trend giornaliera

print(et_points.shape)
print(et_points.columns.tolist())

"""Proviamo ad aggiungere una variabile dicotomica che sia uguale a 1 se la posizione long è profittevole su base giornaliera, 0 altrimenti"""

# Ordina i dati giornalieri per data
daily_data = daily_data.sort_values('Date')

# Merge del dataset et_points con daily_data per ottenere il prezzo di apertura
et_points = et_points.merge(daily_data[['Date', 'Open']], on='Date', how='left')

# Funzione per calcolare se una posizione sarebbe stata profittevole
def calculate_profitability(row):
    if pd.isna(row['Open']):
        return np.nan
    return 1 if row['Open'] < row['Close'] else 0

# Applica la funzione per calcolare la variabile dicotomica
et_points['Profitable'] = et_points.apply(calculate_profitability, axis=1)

# Filtra i risultati per rimuovere valori NaN
et_points = et_points.dropna(subset=['Profitable'])

# Salva il nuovo dataset con la variabile aggiunta
et_points.to_csv('filtered_et_points_with_profitability.csv', index=False)

# Stampa delle dimensioni e delle colonne del dataframe aggiornato
print("Dimensioni del DataFrame aggiornato:", et_points.shape)
print("Colonne del DataFrame aggiornato:", et_points.columns.tolist())

# Visualizza i primi risultati per verifica
print(et_points.head())

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Seleziona le caratteristiche e l'etichetta
columns_to_drop = ['Date', 'Open', 'High', 'Low', 'Close', 'Trend']
for col in columns_to_drop:
    if col not in et_points.columns:
        print(f"Colonna '{col}' non trovata nel DataFrame e non sarà rimossa.")
        columns_to_drop.remove(col)

features = et_points.drop(columns=columns_to_drop)
target = et_points['Profitable']

# Standardizzazione dei dati
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applicazione della PCA
pca = PCA()
features_pca = pca.fit_transform(features_scaled)

# Calcolo della varianza spiegata cumulativa
explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("Varianza spiegata cumulativa:", explained_variance)

# Calcolo della varianza residua
residual_variance = 1 - explained_variance
print("Varianza residua cumulativa:", residual_variance)

# Determina il numero di componenti principali che spiegano il 95% della varianza
num_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Numero di componenti principali selezionate per spiegare il 95% della varianza: {num_components}")

# Riduci il dataset alle componenti principali selezionate
features_pca_selected = features_pca[:, :num_components]

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(features_pca_selected, target, test_size=0.2, random_state=42)

# Inizializza e addestra il modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prevedi sui dati di test
y_pred = model.predict(X_test)

# Valuta il modello
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Stampa le componenti principali selezionate
print("Componenti principali selezionate:", num_components)
print("Varianza spiegata da ciascuna componente:", pca.explained_variance_ratio_[:num_components])

# Creazione del grafico a gomito
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(residual_variance) + 1), residual_variance, marker='o', linestyle='--')
plt.xlabel('Numero di componenti principali')
plt.ylabel('Varianza residua cumulativa')
plt.title('Grafico a gomito della varianza residua')
plt.grid(True)
plt.show()

"""Andiamo adesso a salvare le prime sei componenti"""

# Calcolo della varianza cumulativa spiegata dalle prime sei componenti
explained_variance_six_components = np.sum(pca.explained_variance_ratio_[:10])
print(f"Varianza cumulativa spiegata dalle prime sei componenti: {explained_variance_six_components}")

# Salva le prime 6 componenti principali in un nuovo DataFrame
components_to_save = 10
features_pca_6 = features_pca[:, :components_to_save]

# Crea un nuovo DataFrame con le prime 6 componenti principali
pca_df = pd.DataFrame(features_pca_6, columns=[f'PC{i+1}' for i in range(components_to_save)])
pca_df['Profitable'] = target.values

# Visualizza il DataFrame per verifica
print(pca_df.head(10))

print(pca_df.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix




# Seleziona le caratteristiche e l'etichetta
X = pca_df.drop(columns=['Profitable'])
y = pca_df['Profitable']

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Inizializza i modelli
models = {
    'Logistic Regression': LogisticRegression(random_state=43),
    'Support Vector Machine': SVC(random_state=43),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=43)
}

# Addestra e valuta i modelli
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Results for {name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n" + "="*60 + "\n")

# Visualizza i risultati con un grafico a barre
accuracy_scores = [accuracy_score(y_test, models[name].predict(X_test)) for name in models]
model_names = list(models.keys())

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'orange'])
plt.xlabel('Modello')
plt.ylabel('Accuratezza')
plt.title('Confronto dei modelli di ML supervisionato')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical



# Rete Neurale
# Prepara i dati per la rete neurale
y_train_nn = to_categorical(y_train, num_classes=2)
y_test_nn = to_categorical(y_test, num_classes=2)

# Definisci la rete neurale
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(2, activation='softmax'))

# Compila il modello
nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Addestra la rete neurale
nn_model.fit(X_train, y_train_nn, epochs=50, batch_size=10, verbose=1)

# Valuta la rete neurale
y_pred_nn = nn_model.predict(X_test)
y_pred_nn_class = np.argmax(y_pred_nn, axis=1)

print("Results for Neural Network:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn_class))
print(classification_report(y_test, y_pred_nn_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn_class))
print("\n" + "="*60 + "\n")

# Visualizza i risultati con un grafico a barre
accuracy_scores = [accuracy_score(y_test, models[name].predict(X_test)) for name in models]
accuracy_scores.append(accuracy_score(y_test, y_pred_nn_class))
model_names = list(models.keys()) + ['Neural Network']

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=['#003f5c', '#2f4b7c', '#665191', '#a05195'])
plt.xlabel('Modello')
plt.ylabel('Accuratezza')
plt.title('Confronto dei modelli di ML supervisionato')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Visualizza i risultati con un grafico a barre
accuracy_scores = [accuracy_score(y_test, models[name].predict(X_test)) for name in models]
accuracy_scores.append(accuracy_score(y_test, y_pred_nn_class))
model_names = list(models.keys()) + ['Neural Network']

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=['#003f5c', '#2f4b7c', '#665191', '#a05195'])
plt.xlabel('Modello')
plt.ylabel('Accuratezza')
plt.title('Confronto dei modelli di ML supervisionato')
plt.ylim(0, 1)
plt.grid(True)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Dizionario per memorizzare le matrici di confusione
confusion_matrices = {}

# Calcola le matrici di confusione per i modelli esistenti
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

# Calcola la matrice di confusione per la rete neurale
y_pred_nn_class = np.argmax(nn_model.predict(X_test), axis=1)
cm_nn = confusion_matrix(y_test, y_pred_nn_class)
confusion_matrices['Neural Network'] = cm_nn

# Funzione per tracciare una heatmap della matrice di confusione
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

# Traccia le matrici di confusione per tutti i modelli
for name, cm in confusion_matrices.items():
    plot_confusion_matrix(cm, f'Matrix di Confusione - {name}')