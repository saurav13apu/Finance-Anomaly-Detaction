import pandas as pd
import numpy as np

# Load the dataset
data = {
    'transaction_id': ['TRX001', 'TRX002', 'TRX003', 'TRX004', 'TRX005', 'TRX006', 'TRX007'],
    'date': ['2024-06-01', '2024-06-01', '2024-06-01', '2024-06-02', '2024-06-02', '2024-06-03', '2024-06-03'],
    'category': ['Food', 'Utilities', 'Entertainment', 'Food', 'Transport', 'Utilities', 'Food'],
    'amount': [25.00, 150.00, 200.00, 3000.00, 45.00, 135.00, 20.00]
}
df = pd.DataFrame(data)

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Data Preprocessing
def preprocess_data(df):
    # Handle missing data
    df = df.dropna()

    # Ensure amount is numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Drop rows with invalid data
    df = df.dropna(subset=['amount'])

    return df

df = preprocess_data(df)

# Calculate mean, median, and standard deviation for transaction amounts by categories
def calculate_statistics(df):
    stats_df = df.groupby('category')['amount'].agg(['mean', 'median', 'std', 'count']).reset_index()
    return stats_df

stats_df = calculate_statistics(df)
print("Statistics by category:")
print(stats_df)

# Anomaly Detection using Z-score
def z_score_outliers(df, stats_df, z_thresh=2):
    anomalies = []
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        if len(category_df) > 1:  # Ensure there are enough data points
            mean = stats_df[stats_df['category'] == category]['mean'].values[0]
            std = stats_df[stats_df['category'] == category]['std'].values[0]
            if std == 0 or np.isnan(std):
                continue  # Skip if standard deviation is zero or NaN
            category_df = category_df.copy()  # Create a copy to avoid SettingWithCopyWarning
            category_df['z_score'] = (category_df['amount'] - mean) / std
            print(f"Category: {category}, Mean: {mean}, Std: {std}")
            print(category_df[['transaction_id', 'amount', 'z_score']])
            outliers = category_df[np.abs(category_df['z_score']) > z_thresh]
            for index, row in outliers.iterrows():
                reason = f"Z-score outlier (Z-score = {row['z_score']:.2f})"
                anomalies.append((row['transaction_id'], row['date'], row['category'], row['amount'], reason))

    return anomalies

# Anomaly Detection using IQR
def iqr_outliers(df):
    anomalies = []
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        if len(category_df) > 1:  # Ensure there are enough data points
            Q1 = category_df['amount'].quantile(0.25)
            Q3 = category_df['amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            print(f"Category: {category}, Q1: {Q1}, Q3: {Q3}, IQR: {IQR}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            outliers = category_df[(category_df['amount'] < lower_bound) | (category_df['amount'] > upper_bound)]
            for index, row in outliers.iterrows():
                reason = f"IQR outlier (amount not in [{lower_bound:.2f}, {upper_bound:.2f}])"
                anomalies.append((row['transaction_id'], row['date'], row['category'], row['amount'], reason))

    return anomalies

# Anomaly Detection using MAD
def mad_outliers(df, mad_thresh=3):
    anomalies = []
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        if len(category_df) > 1:  # Ensure there are enough data points
            median = category_df['amount'].median()
            mad = np.median(np.abs(category_df['amount'] - median))
            if mad == 0:
                mad = np.std(category_df['amount'])  # Fallback to std if MAD is zero
            lower_bound = median - mad_thresh * mad
            upper_bound = median + mad_thresh * mad
            print(f"Category: {category}, Median: {median}, MAD: {mad}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")
            outliers = category_df[(category_df['amount'] < lower_bound) | (category_df['amount'] > upper_bound)]
            for index, row in outliers.iterrows():
                reason = f"MAD outlier (amount not in [{lower_bound:.2f}, {upper_bound:.2f}])"
                anomalies.append((row['transaction_id'], row['date'], row['category'], row['amount'], reason))

    return anomalies

# Detect anomalies using Z-score
anomalies_z = z_score_outliers(df, stats_df)

# Detect anomalies using IQR
anomalies_iqr = iqr_outliers(df)

# Detect anomalies using MAD
anomalies_mad = mad_outliers(df)

# Combine all anomalies
all_anomalies = anomalies_z + anomalies_iqr + anomalies_mad

# Remove duplicates
all_anomalies = list(dict.fromkeys(all_anomalies))

# Reporting
def generate_report(anomalies):
    report_df = pd.DataFrame(anomalies, columns=['transaction_id', 'date', 'category', 'amount', 'reason_for_anomaly'])
    return report_df

report_df = generate_report(all_anomalies)
print("\nAnomalies detected:")
print(report_df)