import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt





df_day = pd.read_csv('C:/Users/aless/OneDrive/Desktop/tesi/daily_btc_obs_2014_2023.csv')
print(df_day.head())
print(df_day.shape)

# Convert 'Date' column into datetime format
df_day['Date'] = pd.to_datetime(df_day['Date'])

# Set 'Date' as index
df_day.set_index('Date', inplace=True)

# Create the weekly dataset
df_week = df_day.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Store weekly dataset
df_week.to_csv('weekly_observations.csv')

# Here we try instead to isolate only those ET points where, following a trend change, 
# the daily and weekly momentum are aligned.

def calculate_momentum(data, window=14): # calculate the momentum over 14 periods
    momentum = data['Close'].diff(window) # difference between each closing price and the one 14 periods before
    return momentum

# Identifies the entry point 
def identify_et(data, window=14): 
    data['Momentum'] = calculate_momentum(data, window)
    data['ET'] = np.where((data['Momentum'] > 0) & (data['Momentum'].shift(1) <= 0) |
                          (data['Momentum'] < 0) & (data['Momentum'].shift(1) >= 0), 1, 0)
    return data

# Generate trading signal based on et and momentum
def generate_signals(data, window=14): 
    data = identify_et(data, window)
    data['Signal'] = 0  # Inizializza tutti i segnali a 0
    data['Signal'][window:] = np.where(data['Momentum'][window:] > 0, 1, -1)
    data['Position'] = data['Signal'].shift()  # Segnale di trading per il periodo successivo
    return data

# Get the table with only et points
def get_et_points(data):
    et_points = data[data['ET'] == 1]
    return et_points

#Join daily and weekly data and filter ET points with aligned momentum 
def filter_et_points_by_momentum(et_points_daily, et_points_weekly):
    # Align daily and weekly data
    et_points_weekly = et_points_weekly.reindex(et_points_daily.index, method='ffill')

    # Join daily and weekly data on ET
    merged_data = et_points_daily.join(et_points_weekly[['Momentum']], rsuffix='_weekly')

    # Filter et points:
    filtered_et_points = merged_data[(merged_data['Momentum'] > 0) & (merged_data['Momentum_weekly'] > 0) |
                                     (merged_data['Momentum'] < 0) & (merged_data['Momentum_weekly'] < 0)]
    # Add a colummn equal to 0 if both momentum are negative and equal to one if both are positive
    filtered_et_points['Momentum_Status'] = np.where((filtered_et_points['Momentum'] > 0) &
                                                     (filtered_et_points['Momentum_weekly'] > 0), 1, 0)
    return filtered_et_points

# Main function to apply momentum strategy, it applies all the functions defined before
def apply_momentum_strategy(data_daily, data_weekly, window=14):
    # Gets daily signals
    signals_daily = generate_signals(data_daily, window)

    # Gets weekly signals
    signals_weekly = generate_signals(data_weekly, window)

    # Gets ET for both timeframes
    et_points_daily = get_et_points(signals_daily)
    et_points_weekly = get_et_points(signals_weekly)

    # Filters et points with aligned momentum
    filtered_et_points = filter_et_points_by_momentum(et_points_daily, et_points_weekly)

    return filtered_et_points


# Apply the momentum strategy function
filtered_et_points = apply_momentum_strategy(df_day, df_week, window=14)



filtered_et_points.to_csv('filtered_et_points.csv') # save it in the current directory

et_points = filtered_et_points

# Upload necessary datasets
df_day = pd.read_csv('C:/Users/aless/OneDrive/Desktop/tesi/daily_btc_obs_2014_2023.csv')
df_week = pd.read_csv("C:/Users/aless/OneDrive/Desktop/tesi/weekly_btc_obs_2014_2023.csv")
et_points = pd.read_csv('C:/Users/aless/OneDrive/Desktop/tesi/filtered_et_points.csv')


# Converts 'Date' into datetime format
df_day['Date'] = pd.to_datetime(df_day['Date'])
df_week['Date'] = pd.to_datetime(df_week['Date'])
et_points['Date'] = pd.to_datetime(et_points['Date'])

# Calculate the RSI function
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate 18 trading indicators for every point of the datasets
for idx, row in et_points.iterrows():
    date = row['Date']
    daily_window = df_day[df_day['Date'] <= date].tail(18)

    # Verify whether there are more than 18 periods left in the data
    if len(daily_window) < 18:
        continue

    # Price action variables
    # day to day closing price percentage change over daily_window
    returns = daily_window['Close'].pct_change().dropna() 
    # variance of closing price
    et_points.at[idx, 's.VPT'] = ((daily_window['Close'] - daily_window['Close'].mean()) ** 2).sum() / (len(daily_window) - 1)
    # average percent change in returns over daily_window
    et_points.at[idx, 'x.V*PPT'] = returns.mean()
    # standard deviation of returns over daily window
    et_points.at[idx, 's.CP.PT'] = returns.std()
    """Polyfit calculates the coefficients of a polynomial that best fits the data, minimizing the squared differences between
      the actual data points and the values predicted by the polynomial. """
    # Slope of the closing price trend, the [0] is used to get the first coefficient (the slope)
    et_points.at[idx, 'Sp.Vr.CP.PT'] = np.polyfit(range(len(daily_window)), daily_window['Close'], 1)[0]
    # Computes the slope of the difference between daily high and low prices
    et_points.at[idx, 'Sp.Vr.HL.PT'] = np.polyfit(range(len(daily_window)), daily_window['High'] - daily_window['Low'], 1)[0]
    # Computes the slope of the difference between closing and opening prices
    et_points.at[idx, 'Sp.Vr.CO.PT'] = np.polyfit(range(len(daily_window)), daily_window['Close'] - daily_window['Open'], 1)[0]
    # Relative Strength Index 
    rsi = calculate_rsi(daily_window['Close'], 14)
    # Relative Strength Index Variance
    et_points.at[idx, 's.Vr.RSI.PT'] = ((rsi - rsi.mean()) ** 2).sum() / (len(rsi) - 1)
    # Relative Strength Index Slope: a rising slope suggest strenghening momentum and viceversa
    et_points.at[idx, 'Sp.Vr.RSI.PT'] = np.polyfit(range(len(rsi.dropna())), rsi.dropna(), 1)[0]

    # Volume variables
    # Volume variance
    et_points.at[idx, 's.Vr.Vo.PT'] = ((daily_window['Volume'] - daily_window['Volume'].mean()) ** 2).sum() / (len(daily_window) - 1)
    # Average volume percentage change
    et_points.at[idx, 'x.Vr.Vo.PT'] = daily_window['Volume'].pct_change().mean()
    # Volume slope
    et_points.at[idx, 'Sp.Vr.Vo.PT'] = np.polyfit(range(len(daily_window)), daily_window['Volume'], 1)[0]
   
    # Divergence variables
    # Close price and RSI divergence: computes the correlation between price movement and RSI movement
    et_points.at[idx, 'Dv.CP-RSLIPT'] = ((daily_window['Close'] - daily_window['Close'].mean()) * (rsi - rsi.mean())).sum() / ((len(daily_window) - 1) * daily_window['Close'].std() * rsi.std())
    # Volume and close price divergence: correlation between volume and price movements
    et_points.at[idx, 'Dv.Vo-CP.PPT'] = ((daily_window['Volume'] - daily_window['Volume'].mean()) * (daily_window['Close'] - daily_window['Close'].mean())).sum() / ((len(daily_window) - 1) * daily_window['Volume'].std() * daily_window['Close'].std())
    # Volume and RSI divergence: correlation between volume and rsi
    et_points.at[idx, 'Dv.Vo-RSLIPT'] = ((daily_window['Volume'] - daily_window['Volume'].mean()) * (rsi - rsi.mean())).sum() / ((len(daily_window) - 1) * daily_window['Volume'].std() * rsi.std())

    # After holding time variables
    # Gross/profit loss for the trade depending on whether the position was long or short
    et_points.at[idx, 'GPL_$'] = row['Close'] - row['Open'] if row['Position'] == 'long' else row['Open'] - row['Close']
    # Percentage price range between high and low prices relative to the opening price
    et_points.at[idx, 'RP% M[1]'] = (row['High'] - row['Low']) / row['Open'] * 100 if row['Position'] == 'long' else (row['Low'] - row['High']) / row['Open'] * 100
    # Price movement: measures price change from the start to the end of the period
    et_points.at[idx, 'Sp.XP-EP'] = (row['Close'] - daily_window['Close'].iloc[0]) / 18

# Long term trend variables
for idx, row in et_points.iterrows():
    date = row['Date']
    weekly_window = df_week[df_week['Date'] <= date].tail(18)

    if len(weekly_window) < 18:
        continue

# Calculates the slope of the linear regression line of the weekly closing prices over the last 18 periods.
# 18 weekly period trend, basically: positive for incrementing trend and viceversa
    et_points.at[idx, 'M.Sp.PET'] = np.polyfit(range(len(weekly_window)), weekly_window['Close'], 1)[0]
# Average closing price over the last 18 periods
    et_points.at[idx, 'LTT'] = weekly_window['Close'].mean()


"""Proviamo ad aggiungere una variabile dicotomica che sia uguale a 1 se la posizione long è profittevole su base giornaliera, 0 altrimenti"""

# Order daily data by date 
df_day = df_day.sort_values('Date')

# Merge et_points and daily dataset to have the open price
et_points = et_points.merge(df_day[['Date', 'Open']], on='Date', how='left')

# Calculate profitability, this is going to be the target of our ml models
def calculate_profitability(row):
    if pd.isna(row['Open_x']) or pd.isna(row['Close']):
        return np.nan
    if row['Position'] == 1:
        return 1 if row['Close'] > row['Open_x'] else 0
    elif row['Position'] == -1:
        return 1 if row['Close'] < row['Open_x'] else 0
    else:
        return np.nan

# Apply calculate profitability on the dataset, creates the target column
et_points['Profitable'] = et_points.apply(calculate_profitability, axis=1)

# Shift the Profitable column up by 1 day to make it the next day's profitability
et_points['Next_Day_Profitable'] = et_points['Profitable'].shift(-1)

# Drop any rows where the target is NaN (because we can't predict beyond the dataset)
et_points.dropna(subset=['Next_Day_Profitable'], inplace=True)


# Rimuovere le colonne 'Date', 'Profitable', 'Open_y'
columns_to_remove = ['Date', 'Profitable', 'Open_y']  # Elenca tutte le colonne che vuoi rimuovere

# Rimuovi le colonne dal DataFrame
et_points_cleaned = et_points.drop(columns=columns_to_remove)

# Select attributes and target
# Definisce il target
target = et_points['Next_Day_Profitable']

# Remove the target column from the features before scaling and PCA
et_points_cleaned_no_target = et_points_cleaned.drop(columns=['Next_Day_Profitable'])

# Divide data into training and test set
X_train, X_test, y_train, y_test = train_test_split(et_points_cleaned_no_target, target, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Calculate cumulative variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)
print("Varianza spiegata cumulativa:", explained_variance)

# Determine number of PCA explaining 95% of the variance
num_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Numero di componenti principali selezionate per spiegare il 95% della varianza: {num_components}")

# Shrink the dataset to the selected dimension
X_train_pca_selected = X_train_pca[:, :num_components]
X_test_pca_selected = X_test_pca[:, :num_components]


# Assicurati che le dimensioni di X_train e y_train siano coerenti
assert X_train_pca_selected.shape[0] == y_train.shape[0], "Mismatch tra X_train_pca_selected e y_train"
assert X_test_pca_selected.shape[0] == y_test.shape[0], "Mismatch tra X_test_pca_selected e y_test"

# Inizializza e allena il modello
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca_selected, y_train)

# Predici sui dati di test
y_pred = model.predict(X_test_pca_selected)

# Valuta il modello
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Inizializza altri modelli
models = {
    'Logistic Regression': LogisticRegression(random_state=43),
    'Support Vector Machine': SVC(random_state=43),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=43)
}

# Train and evaluate the models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_pca_selected, y_train)
    y_pred = model.predict(X_test_pca_selected)

    print(f"Results for {name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n" + "="*60 + "\n")

# Visualizza i risultati con un barplot
accuracy_scores = [accuracy_score(y_test, models[name].predict(X_test_pca_selected)) for name in models]
model_names = list(models.keys())

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=['blue', 'green', 'orange'])
plt.xlabel('Modello')
plt.ylabel('Accuratezza')
plt.title('Confronto dei modelli di ML supervisionato')
plt.ylim(0, 1)
plt.grid(True)
plt.show()


# Prepare data for a neural network
y_train_nn = to_categorical(y_train, num_classes=2)
y_test_nn = to_categorical(y_test, num_classes=2)

# Define the neural network
nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train_pca_selected.shape[1], activation='relu'))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(2, activation='softmax'))


nn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train it
nn_model.fit(X_train_pca_selected, y_train_nn, epochs=50, batch_size=10, verbose=1)

# Evaluate it 
y_pred_nn = nn_model.predict(X_test_pca_selected)
y_pred_nn_class = np.argmax(y_pred_nn, axis=1)

print("Results for Neural Network:")
print("Accuracy:", accuracy_score(y_test, y_pred_nn_class))
print(classification_report(y_test, y_pred_nn_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn_class))
print("\n" + "="*60 + "\n")

# Barplot 
accuracy_scores = [accuracy_score(y_test, models[name].predict(X_test_pca_selected)) for name in models]
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

# Barplot
accuracy_scores = [accuracy_score(y_test, models[name].predict(X_test_pca_selected)) for name in models]
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


# Dict to memorize confusion matrixes
confusion_matrices = {}

# Confusion matrix for the trained models
for name, model in models.items():
    y_pred = model.predict(X_test_pca_selected)
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm

# For the neural network
y_pred_nn_class = np.argmax(nn_model.predict(X_test_pca_selected), axis=1)
cm_nn = confusion_matrix(y_test, y_pred_nn_class)
confusion_matrices['Neural Network'] = cm_nn

# Heatmap function for the confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

for name, cm in confusion_matrices.items():
    plot_confusion_matrix(cm, f'Matrix di Confusione - {name}')