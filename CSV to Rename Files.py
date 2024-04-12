import csv
import os
# user input master file path
# csv_file_path = input('Enter the path to the CSV file: ')
csv_file_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\2X2 Mod Depth to Power updated.csv'
master_folder_path = r''.join(input('Enter the path to the master folder: '))


csv_file_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\2X2 Mod Depth to Power updated.csv'
BG_folder_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\138C\\2X2 Trans5 50mm cell 138C Onres 3036MHz\\BG'
EIT_folder_path = 'C:\\Users\\zxq220007\\Box\\Quantum Optics Lab\\TeTON OANN Testbed\\Data 2024\\Apr 11 2024\\138C\\2X2 Trans5 50mm cell 138C Onres 3036MHz\\EIT'


# this is a test
# Hello world


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

print('Renaming process complete.')