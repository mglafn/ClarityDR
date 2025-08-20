import os
import zipfile
import kaggle

competition_name = 'aptos2019-blindness-detection'
data_dir = '../data'

def handle_data_dir():
    """
    Checks if data directory exists in the local path.
    If it doesn't, create it and download the dataset.
    """
    if os.path.exists(data_dir):
        print("Data directory already exists.")
        return

    print("Data directory not found. Creating directory...")
    os.makedirs(data_dir)

    print("Directory created. Downloading data...")
    kaggle.api.competition_download_files(competition_name, path=data_dir)

    print(f"Data downloaded to {data_dir}.")


def handle_data_extraction():
    """
    Searches data directory for the downloaded
    .zip file and extracts, if needed.
    """
    print("Searching for zip file...")
    extracted_data_path = os.path.join(data_dir, competition_name) # Keep this line
    zipped_data_path = os.path.join(data_dir, f"{competition_name}.zip")

    if not os.path.exists(zipped_data_path):
        print("Zip file not found. Something went wrong.")
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
