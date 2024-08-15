import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import pickle

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Load the data
data_path = 'C:/Users/MSI CARBON/Desktop/latest/product1.csv'
df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

inventory_levels = df['inventory_levels']

# Apply STL decomposition to obtain seasonal, trend, and residuals components
stl = sm.tsa.seasonal_decompose(inventory_levels, period=30, model='multiplicative')
seasonal = stl.seasonal
trend = stl.trend
residuals = stl.resid

# Split the data into train and test sets
residuals = int(len(df) * 0.85)
train, test = df.iloc[:residuals], df.iloc[residuals:]

# Perform preprocessing
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train[['inventory_levels']])
test_scaled = scaler.transform(test[['inventory_levels']])

# Create dataset for training and testing
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 7
X_train, y_train = create_dataset(train_scaled, train_scaled, TIME_STEPS)
X_test, y_test = create_dataset(test_scaled, test_scaled, TIME_STEPS)

# Build the model
model = keras.Sequential([
    keras.layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])),
    keras.layers.Dropout(rate=0.2),
    keras.layers.RepeatVector(n=X_train.shape[1]),
    keras.layers.LSTM(units=64, return_sequences=True),
    keras.layers.Dropout(rate=0.2),
    keras.layers.LSTM(units=64, return_sequences=True),
    keras.layers.Dropout(rate=0.2),
    keras.layers.LSTM(units=128, return_sequences=True),  
    keras.layers.Dropout(rate=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2]))
])


model.compile(loss='mae', optimizer='adam')

# Add early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    shuffle=False,
    callbacks=[early_stopping]
)

# Perform anomaly detection
X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
train_rmse_loss = np.sqrt(np.mean(np.square(X_train_pred - X_train), axis=1))

threshold = np.max(train_mae_loss)

print("threshold = ", threshold)

X_test_pred = model.predict(X_test)
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
print("test_mae_loss: ",test_mae_loss )
test_rmse_loss = np.sqrt(np.mean(np.square(X_test_pred - X_test), axis=1))

# Calculate MAE and RMSE for training and testing datasets
train_mae = np.mean(train_mae_loss)
train_rmse = np.mean(train_rmse_loss)
test_mae = np.mean(test_mae_loss)
test_rmse = np.mean(test_rmse_loss)

print("Train MAE: {:.4f}".format(train_mae))
print("Train RMSE: {:.4f}".format(train_rmse))
print("Test MAE: {:.4f}".format(test_mae))
print("Test RMSE: {:.4f}".format(test_rmse))

# Plot the training and testing losses
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.show()

# Inverse transform the scaled data to get original values
X_train_original = scaler.inverse_transform(X_train.reshape(-1, 1))
X_train_pred_original = scaler.inverse_transform(X_train_pred.reshape(-1, 1))

# Reshape the arrays for plotting
X_train_original = X_train_original.reshape(-1, X_train.shape[1])
X_train_pred_original = X_train_pred_original.reshape(-1, X_train.shape[1])

# Plot the original training data and predicted values
plt.figure(figsize=(10, 6))
plt.plot(X_train_original, label='Original Data', color='blue')
plt.plot(X_train_pred_original, label='Predicted Data', color='red', linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Inventory Levels')
plt.title('Original vs. Predicted Training Data')
plt.legend()
plt.show()

#Save the trained model as a pickle file
model_path = 'model1.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)