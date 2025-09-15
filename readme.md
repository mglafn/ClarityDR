1. pip install -r requirements.txt
2. Place kaggle.json in .kaggle of home directory.


To start and stop
- docker-compose up --build | docker-compose up
- docker-compose down

To navigate back to a running Docker:
1. docker ps
2. docker exec -it <image> bash
3. ps aux | grep python
4. kill <id>

# ClarityDR: Diabetic Retinopathy Classification

This project implements and compares two deep learning models for classifying diabetic retinopathy from fundus images, based on the APTOS 2019 Blindness Detection dataset.

- **Model A:** Transfer Learning (frozen ResNet50 base)
- **Model B:** Fine-Tuning (unfrozen ResNet50 base)

The entire development environment is containerized using Docker to ensure perfect reproducibility and to handle a complex, unsupported hardware configuration.

## Development Environment Setup: Docker & ROCm on Navi 10

**⚠️ Important Note:** This project is configured to run in a very specific, unofficial environment: an AMD Navi 10 GPU (e.g., RX 5700 XT) on a modern Linux distribution (e.g., Nobara/Fedora) using ROCm. This hardware is not officially supported by modern ROCm releases. The setup below contains a series of community-sourced workarounds to make it functional.

Follow these instructions precisely.

### 1. Prerequisites: Docker Installation

This environment requires a working Docker installation. On Fedora-based systems like Nobara, the standard `docker-ce` package may not be available. Install Docker using the native system packages:

sudo dnf install moby-engine docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

ou must reboot your computer after running these commands for the user group changes to take effect. Verify your installation by running docker ps.
2. Building the Docker Image (One-Time Setup)
The Dockerfile in this repository defines the entire development environment. It starts from a stable, older ROCm 5.2.3 development image and manually installs a compatible version of PyTorch and all other necessary libraries.
To build the image, navigate to the project's root directory in your terminal and run:

docker-compose build --no-cache

This will be a very long process (20-40 minutes) as it downloads the multi-gigabyte base image and installs all dependencies. You only need to do this once.
3. Running the Jupyter Lab Environment
Once the image is built, you can start and stop the environment with simple commands.

To START your work session:

docker-compose up -d

This will start the container in the background.
To ACCESS your notebooks:
Open a web browser and navigate to http://localhost:8888. You will see the Jupyter Lab interface with the entire project directory ready to use. All work is saved directly to your project folder.
To STOP your work session:

docker-compose down