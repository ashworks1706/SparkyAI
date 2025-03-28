import os
import shutil

def convert_py_to_txt(source_folder, desktop_path):
    # Ensure the source folder path ends with a separator
    source_folder = os.path.join(source_folder, '')

    # Get all files in the specified folder (not in subfolders)
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    for file in files:
        if file.endswith('.py'):
            source_path = os.path.join(source_folder, file)
            dest_path = os.path.join(desktop_path, file[:-3] + '.txt')
            
            # Copy the file to desktop with .txt extension
            shutil.copy2(source_path, dest_path)

    print("Conversion completed. Text files saved to desktop.")

# Specify the source folder and desktop path
source_folder = input('Enter path : ')
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

# Run the conversion
convert_py_to_txt(source_folder, desktop_path)

