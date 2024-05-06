import os
import pandas as pd
import zipfile


extracted_folder_path = 'E:/Workspace/raspi_dataset'


# Listing the contents of the extracted folder to get to the dataset directory
dataset_directory = os.path.join(extracted_folder_path, 'ASL-Sensor-Dataglove-Dataset')

# List of directory names corresponding to each label, as assumed from your description
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'you', 'Z'
]

# Columns to be extracted from each CSV file
columns_to_extract = ['timestamp','flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5', 'ACCx', 'ACCy', 'ACCz', 'GYRx', 'GYRy', 'GYRz']

# Preparing an empty DataFrame to compile the dataset
compiled_data = pd.DataFrame(columns=['label'] + columns_to_extract)

# Function to compile data for each label
def compile_label_data(directory, label, compiled_data):
    try:
        # Load the CSV file for the label
        df = pd.read_csv(os.path.join(directory, label + '.csv'))
        # Extract the columns of interest
        label_data = df[columns_to_extract]
        # Add a 'label' column
        label_data.insert(0, 'label', label)
        # Append the label data to the compiled data
        compiled_data = pd.concat([compiled_data, label_data], ignore_index=True)
    except FileNotFoundError:
        print(f"File for label '{label}' not found in directory '{directory}'.")
    except Exception as e:
        print(f"An error occurred while compiling data for label '{label}': {e}.")
    return compiled_data

# Iterating over the directories and compiling the data for each label
for dir_number in sorted(os.listdir(dataset_directory)):
    current_dir = os.path.join(dataset_directory, dir_number)
    for label in labels:
        compiled_data = compile_label_data(current_dir, label, compiled_data)

# Save the compiled dataset to a CSV file
compiled_csv_file_path = 'sensor_data.csv'
compiled_data.to_csv(compiled_csv_file_path, index=False)

compiled_csv_file_path
