This project investigates using Long Short-Term Memory (LSTM) with Seasonal-Trend Decomposition (STL) to detect anomalies in Bengco's Agricultural Supply inventory data.

Problem: Bengco relies on a manual inventory system, leading to potential inaccuracies and inefficiencies.

Solution: Develop a web application that automatically detects anomalies in inventory levels using a machine learning model.


Methodology:

-Data Collection: Inventory data from the past 4 years is collected.


Data Preprocessing:

-Relevant data like inventory levels and dates are extracted.

-Anomalies caused by external events (disasters, pests, theft) are filtered.

-Train-test split is performed (85% training, 15% testing/validation).


Model Development:

-STL decomposition separates the data into trend, seasonality, and residual components.

-An LSTM autoencoder model is trained on the residuals to learn normal patterns.

-The model reconstructs the data, and anomalies are identified by comparing the original data to the reconstruction with high error (Mean Squared Error).


Web Application Development:

-A web application is built using Python and Flask framework.

-Users can upload CSV files containing inventory data.

-The application processes the data and detects anomalies using the trained LSTM model.

Example outputs:

index.html
![Upload Page](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/032ad514-0ecd-4b2a-b8e6-59f21d8bd52e)

Result.html
![Detected Anomalies Page 1](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/47ed878a-986b-4801-9247-471ca986f469)

![Detected Anomalies Page 2](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/4c80534f-455e-467f-89a5-7f81a9672fa8)

no_anomaly.html
![no anomalies page](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/9eca1a32-8a56-48fa-bb25-d24260397692)

trend_seasonality.html
![Trend and Seasonality Page](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/fd8239ec-1c23-44b7-9a3f-06800f67f5b9)

null_values.html
![Null_values](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/1d42645b-0777-4639-b089-6635a578793a)
