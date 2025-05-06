import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os

def convert_to_percentage(time_in_seconds, cycle_time):
    return (time_in_seconds / cycle_time) * 100

def preprocess_gait_data(input_path, output_path):
    # Load CSV file
    df = pd.read_csv(input_path)

    # Step 1: Compute gait cycle time
    df['Cycle_Time'] = (df['Lt_Stride_Time'] + df['Rt_Stride_Time']) / 2

    # Step 2: Convert time-based features to percentage
    time_features = ['Lt_Double_Support', 'Lt_Single_Support',
                     'Rt_Double_Support', 'Rt_Single_Support']

    for feature in time_features:
        df[feature] = df.apply(lambda row: convert_to_percentage(row[feature], row['Cycle_Time']), axis=1)

    df.drop(columns=['Cycle_Time'], inplace=True)

    # Step 3: Remove outliers using Isolation Forest
    feature_cols = df.drop(columns=['target']).columns  # Assuming 'target' is the label column
    iso = IsolationForest(contamination=0.01, random_state=42)
    outlier_preds = iso.fit_predict(df[feature_cols])
    df_cleaned = df[outlier_preds == 1].copy()

    # Step 4: Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = df_cleaned.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_cleaned[feature_cols])

    # Step 5: Save the preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_csv = "./data/your_dataset.csv"  # Modify as needed
    output_csv = "./data/preprocessed_dataset.csv"
    preprocess_gait_data(input_csv, output_csv)
