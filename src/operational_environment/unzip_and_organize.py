"""
This script is aimed at helping extract and organize the files from the Nordex/Bachmann Platform into a workable format
"""

import os
import shutil
import tarfile
import zipfile

# Define paths
unprocessed_dir = "/Users/guillermogildeavallebellido/Desktop/isweargodbethelast/different months/TURBINE1/unclassified"
raw_dentol_dir = "/Users/guillermogildeavallebellido/Desktop/isweargodbethelast/different months/TURBINE1"

# Define ontology: map aduchannel numbers to folder names
ontology = {
    "aduchannel1": "TURBINE1_PlanetaryStage1_AI2" ### Add as many "aduchannelN" as sensors to be evaluated
}

"""
    "aduchannel2": "TurbineN_MainBearing_AI1",
    "aduchannel3": "TurbineN_PlanetaryStage2_AI3",
    "aduchannel4": "TurbineN_HighSpeedShaft_AI4",
    "aduchannel5": "TurbineN_GeneratorDriveEnd_AI5",
    "aduchannel6": "TurbineN_GeneratorNonDriveEnd_AI6",
    "aduchannel7": "TurbineN_HighSpeedShaftAxial_AI7",

"""
# Function to extract compressed files
def extract_compressed_file(file_path, extract_to):
    if file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        print(f"Unsupported file type for {file_path}")

# Function to organize files
def organize_files(src_dir, dest_dir, ontology):
    for root, _, files in os.walk(src_dir):
        for file in files:
            for channel, folder_name in ontology.items():
                if file.startswith(channel):
                    destination_folder = os.path.join(dest_dir, folder_name)
                    os.makedirs(destination_folder, exist_ok=True)
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(destination_folder, file)

                    # Move file, overwrite if exists
                    shutil.move(src_file, dest_file)
                    print(f"Moved: {src_file} -> {dest_file}")
                    break  # Exit loop once a match is found

# Main script
def main():
    for compressed_file in os.listdir(unprocessed_dir):
        compressed_path = os.path.join(unprocessed_dir, compressed_file)

        # Extract each compressed file into a temporary directory
        temp_dir = os.path.join(unprocessed_dir, "temp_extracted")
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Extracting {compressed_path}...")
        extract_compressed_file(compressed_path, temp_dir)

        # Organize extracted files into the correct sensor folders
        print(f"Organizing files from {compressed_file}...")
        organize_files(temp_dir, raw_dentol_dir, ontology)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

    print("Processing complete!")

if __name__ == "__main__":
    main()
