import os
import subprocess

# Define the base directory containing the list of folders
base_dataset_path = r"C:\Sambhav\Manipal\Year 2\3rd Semester\Glaucoma Detection\11INDI\COMBINED"

# List of folder names
folder_names = os.listdir(base_dataset_path)

# Iterate over each folder
for folder_name in folder_names:
    # Define the custom dataset path and filename
    custom_dataset_path = os.path.join(base_dataset_path, folder_name)
    filename = folder_name

    # Create a folder for saving results
    results_folder = os.path.join(r"C:\Sambhav\Manipal\Year 2\3rd Semester\Glaucoma Detection\CCT\AyushRLModel\Results", filename)
    os.makedirs(results_folder, exist_ok=True)

    # Command to run the CCTModelwRLCode.py script with arguments
    command = [
        "python", 
        "train2AyushModifiedModifief.py", 
        "--custom_dataset_path={}".format(custom_dataset_path), 
        "--filename={}".format(filename)
    ]

    # Execute the command
    subprocess.run(command)
