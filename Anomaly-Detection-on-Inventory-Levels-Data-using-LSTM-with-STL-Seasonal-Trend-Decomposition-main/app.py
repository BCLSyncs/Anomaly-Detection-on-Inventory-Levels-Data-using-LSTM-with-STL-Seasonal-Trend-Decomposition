# Import necessary libraries
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory, session
from flask_session import Session
from sklearn.preprocessing import MinMaxScaler
from datetime import date, datetime, timedelta
from shutil import copyfile
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle
import json
import os

date_backup = date.today()

# Configuration settings
app = Flask(__name__)  # Create a Flask web application
app.secret_key = 'your_secret_key_here'  # Replace with your actual secret key
app.config['SESSION_TYPE'] = 'filesystem'  # Use filesystem storage for session data
Session(app)

# Load the trained model from the pickle file
with open('C:/Users/MSI CARBON/Desktop/anomaly design latest/models/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to create input sequences for the model
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Define the home page route
@app.route('/')
def home():
    if date_backup.strftime("%A") == "Friday":

        model_date_backup = str(date_backup).replace('-','.')
        
        path_input = r'C:\Users\MSI CARBON\Desktop\anomaly design latest\models\model.pkl'
        path_output = r'G:\Backup_Model' + '\\' + model_date_backup + ' - model.pkl'

        date_remove =date.today() - timedelta(days=21)

        model_date_remove = str(date_remove).replace('-', '.')
        file_to_remove = rf'G:\Backup_Model\{model_date_remove} - model.pkl'

        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)

        copyfile(path_input,path_output)
    return render_template('index.html')


# Route for detecting anomalies
@app.route('/detect_anomalies', methods=['GET','POST'])
def detect_anomalies():

    CSV_date = datetime.now().strftime('%Y-%m-%d') 
    
    # Get the uploaded CSV file from the user
    uploaded_file = request.files['file']
    file_name = os.path.splitext(uploaded_file.filename)[0]
    data = pd.read_csv(uploaded_file)
    

    if data.isnull().values.any():
        null_rows = data[data.isnull().any(axis=1)]
        null_data = null_rows.to_html()

        return render_template('null_values.html', table=null_data)
    
    else:
        # Extract relevant columns
        inventory_levels = data['inventory_levels']
        dates = data['Date']

        # Extract all of the anomaly data.
        columns_to_filter = data.columns.tolist()[3:]

        # Filter the DataFrame to include only the specified columns
        filtered_df = data[columns_to_filter]

        # Filter rows where any of the specified columns has a value greater than 1
        filtered_df = filtered_df[(filtered_df > 0).any(axis=1)]

        # Convert the 'Date' column to a pandas datetime object with the correct format
        data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

        # Add a new column 'Year' and 'Month' to the DataFrame to extract the year and month from the 'Date' column
        filtered_df['Year'] = data['Date'].dt.year
        filtered_df['Month'] = data['Date'].dt.month

        # Group the DataFrame by 'Year' and 'Month' columns and sum the values for each month and year
        grouped_df = filtered_df.groupby(['Year', 'Month']).sum().reset_index()

        # Find the column with the highest sum per year and month
        max_column_per_month = grouped_df[columns_to_filter].idxmax(axis=1)

        # Add the 'Prominent anomaly' column to the grouped DataFrame to show which column had the highest sum
        grouped_df['Prominent Anomaly'] = max_column_per_month

        # Generate a list of available years from your data
        available_years = list(filtered_df['Year'].unique())

        session['grouped_df'] = grouped_df.to_json(orient='split')

        # Convert the anomaly data to HTML table format for each year separately
        years = grouped_df['Year'].unique()
        tables = {}

        for year in years:
            year_data = grouped_df[grouped_df['Year'] == year]
            year_table = year_data.to_html(index=False)
            tables[f'Table_{year}'] = year_table

        # Find the column with the highest sum
        most_sum_column = filtered_df[columns_to_filter].sum().idxmax()

        # Calculate the sum of the most prominent anomaly for the entire dataset
        sum_of_most_prominent_anomaly = filtered_df[most_sum_column].sum()

        # Generate a list of available years from your data
        available_years = list(filtered_df['Year'].unique())

        selected_year = request.args.get('year')

        if selected_year is None:
            # Default to the first available year
            selected_year = available_years[0]

        # Generate a table for the selected year
        selected_data = grouped_df[grouped_df['Year'] == int(selected_year)]
        selected_table = selected_data.to_html(index=False)

        # Clear previous session data
        session.pop('trend_data', None)
        session.pop('seasonality_data', None)
        session.pop('trend_dates', None)

        # Perform seasonal decomposition on inventory levels
        stl = sm.tsa.seasonal_decompose(inventory_levels, period=30, model='additive')
        seasonal = stl.seasonal
        trend = stl.trend
        residuals = stl.resid

        # Scale the residuals
        scaler = MinMaxScaler()
        scaler.fit(residuals.values.reshape(-1, 1))
        scaled_residuals = scaler.transform(residuals.values.reshape(-1, 1))

        # Create input sequences for prediction
        sequences, _ = create_dataset(pd.DataFrame(scaled_residuals), scaled_residuals, 6)
        predictions = model.predict(sequences)

        # Calculate Mean Absolute Error (MAE) loss and set a threshold for anomaly detection
        mae_loss = np.mean(np.abs(predictions - sequences), axis=1)
        threshold = 0.4
        anomaly_mask = mae_loss > threshold

        # If anomalies are detected
        
        if np.any(anomaly_mask):
            anomaly_indices = np.where(anomaly_mask)[0]
            anomaly_indices = 6 + anomaly_indices
            anomaly_data = data.iloc[anomaly_indices].copy()
            anomaly_data['anomaly'] = False
            anomaly_data.loc[anomaly_indices, 'anomaly'] = True
            anomaly_table = anomaly_data.to_html(index=False)
            


            anomaly_data.to_csv(f'G:/Anomaly_Tables/{file_name}_Anomaly_Data({CSV_date}).csv', index=False)

            # Store trend, seasonality, and date data in the session
            session['trend_data'] = trend.values.tolist()
            session['seasonality_data'] = seasonal.values.tolist()
            session['trend_dates'] = dates.tolist()

            # Render the results page with anomaly and filtered data
            return render_template('result.html', table=anomaly_table, available_years=available_years, selected_year=selected_year, selected_table=selected_table, data=inventory_levels.tolist(), anomalies=anomaly_indices.tolist(), dates=dates.tolist(), prominent=most_sum_column, sum_prominent=sum_of_most_prominent_anomaly)
        else:
            # Redirect to a page indicating no anomalies were found
            return redirect(url_for('no_anomaly'))
    
@app.route('/getdata', methods=['GET'])
def get_data():
    # Retrieve grouped_df from the session
    selected_year = request.args.get('year')
    grouped_df_json = session.get('grouped_df', None)
    
    if grouped_df_json is not None and selected_year is not None:
        grouped_df = pd.read_json(grouped_df_json, orient='split')
        selected_data = grouped_df[grouped_df['Year'] == int(selected_year)]
        selected_table = selected_data.to_html(index=False)
        return selected_table
    else:
        return "Selected year is None or data is not available in the session."
# Route for displaying a page when no anomalies are found
@app.route('/no_anomaly')
def no_anomaly():
    return render_template('no_anomaly.html')

# Route to download a CSV template
@app.route('/download_template')
def download_template():
    template_filename = 'Template.csv'
    return send_from_directory('templates', template_filename, as_attachment=True)

# Route to view trend and seasonality data
@app.route('/view_trend')
def view_trend():
    trend_data = session.get('trend_data')
    seasonality_data = session.get('seasonality_data')
    trend_dates = session.get('trend_dates')

    if trend_data is None or seasonality_data is None or trend_dates is None:
        return "Error: Trend and Seasonality data not found in session."

    trend_data_cleaned = [value for value in trend_data if not np.isnan(value)]
    seasonality_data_cleaned = [value for value in seasonality_data if not np.isnan(value)]

    trend_data_json = json.dumps(trend_data_cleaned)
    seasonality_data_json = json.dumps(seasonality_data_cleaned)

    return render_template('trend_seasonality.html', trendData=trend_data_json, seasonalityData=seasonality_data_json, trendDates=json.dumps(trend_dates))

# Run the Flask app in debug mode
if __name__ == '__main__':
    app.run(debug=True)
