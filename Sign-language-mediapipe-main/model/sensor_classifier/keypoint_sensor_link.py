import pandas as pd

# Paths to the CSV files (update these paths as necessary for your environment)
keypoint_classifier_label_path = 'C:/Users/tripa/Desktop/hand-gesture-recognition-mediapipe-main/model/keypoint_classifier/keypoint_classifier_label.csv'
sensor_data_path = 'sensor_data.csv'
updated_sensor_data_path = 'Linked_key_sensor.csv'  # Path for the output file

# Reading the CSV files
keypoint_classifier_df = pd.read_csv(keypoint_classifier_label_path)
sensor_data_df = pd.read_csv(sensor_data_path)

# Creating a set of labels from both DataFrames to understand what needs to be mapped
keypoint_labels = set(keypoint_classifier_df.iloc[:, 0].str.capitalize())  # Capitalize to match cases like 'You'
sensor_labels = set(sensor_data_df['label'].str.capitalize())  # Apply the same capitalization here

# Create the mapping dictionary
# Since the keypoint labels are gestures and some might not directly match the sensor labels,
# we initialize the mapping with direct matches and then check for any special cases
label_mapping = {label: label for label in sensor_labels if label in keypoint_labels}

# Special case handling for 'you' vs. 'You'
# Assuming 'you' in sensor_data should map to 'You' in keypoint labels if present
if 'You' in keypoint_labels and 'you' in sensor_labels:
    label_mapping['you'] = 'You'

# Apply the mapping to sensor_data_df, leaving unmapped labels as empty
sensor_data_df['Mapped Label'] = sensor_data_df['label'].str.capitalize().map(label_mapping).fillna('')

# Saving the updated sensor_data_df with the 'Mapped Label' column to a new CSV file
sensor_data_df.to_csv(updated_sensor_data_path, index=False)

# Print the path to the updated CSV file
print(f"Updated sensor data saved to: {updated_sensor_data_path}")
