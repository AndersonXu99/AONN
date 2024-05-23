import csv
import os
import time

# deprecated code

# user input master file path
# csv_file_path = input('Enter the path to the CSV file: ')

csv_file_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\5X5 Mod Depth to Power updated.csv'
BG_folder_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 12 2024\\Test - Copy (2)\\BG'
EIT_folder_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 12 2024\\Test - Copy (2)\\EIT'

starting_time = time.time()

#Read CSV file
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip header row

    # Iterate through each row in the CSV file
    for row in csv_reader:
        original_name = row[0]
        desired_name = row[2]

        # Construct full paths for the original and desired files
        BG_original_path = os.path.join(BG_folder_path, original_name + '.TIFF')
        BG_desired_path = os.path.join(BG_folder_path, desired_name + '.TIFF')

        # Construct full paths for the original and desired files
        EIT_original_path = os.path.join(EIT_folder_path, original_name + '.TIFF')
        EIT_desired_path = os.path.join(EIT_folder_path, desired_name + '.TIFF')

        # Rename the file
        try:
            os.rename(BG_original_path, BG_desired_path)
            os.rename(EIT_original_path, EIT_desired_path)
        except FileNotFoundError:
            print(f'File {BG_original_path} not found.')

ending_time = time.time()
print(f'Total time: {ending_time - starting_time}')

print('Renaming process complete.')
