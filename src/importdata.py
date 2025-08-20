import os
import zipfile
import kaggle

competition_name = 'aptos2019-blindness-detection'
raw_data_dir = '../data/raw' 

def handle_data_dir():
    """
    Checks if the raw data directory exists and contains the zip file.
    If not, it creates the directory and downloads the data.
    """
    zipped_data_path = os.path.join(raw_data_dir, f"{competition_name}.zip")

    if os.path.exists(zipped_data_path):
        print(f"Data zip file already found at {zipped_data_path}.")
        return

    print(f"Data not found. Creating directory at {raw_data_dir}...")
    os.makedirs(raw_data_dir, exist_ok=True) 

    print("Directory ready. Downloading data...")
    kaggle.api.competition_download_files(competition_name, path=raw_data_dir)

    print(f"Data downloaded to {raw_data_dir}.")


def handle_data_extraction():
    """
    Searches the raw data directory for the downloaded
    .zip file and extracts it, if needed.
    """
    print("Searching for zip file in raw data directory...")
    extracted_data_path = os.path.join(raw_data_dir, competition_name)
    zipped_data_path = os.path.join(raw_data_dir, f"{competition_name}.zip")

    if not os.path.exists(zipped_data_path):
        print("Zip file not found. Please run the download function first.")
        return

    print(f"File found at {zipped_data_path}.")

    if os.path.exists(extracted_data_path) and os.listdir(extracted_data_path):
        print(f"Unzipped files found at {extracted_data_path}. Skipping extraction...")
        return

    print("Extracting files...")
    os.makedirs(extracted_data_path, exist_ok=True)
    with zipfile.ZipFile(zipped_data_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_data_path)
    print("Extraction complete.")

def main():
    handle_data_dir()
    handle_data_extraction()


if __name__ == "__main__":
    main()