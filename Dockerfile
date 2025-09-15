# Start from the CORRECT and VERIFIED ROCm 5.2.3 development base image
FROM rocm/dev-ubuntu-20.04:5.2.3

# Set the working directory inside the container
WORKDIR /app

ENV OPENCV_OPENCL_RUNTIME=
ENV OPENCV_FOR_THREADS_NUM=1

# --- Environment Setup ---
# Upgrade pip first for better dependency resolution
RUN pip install --upgrade pip

# Install the correct PyTorch for ROCm 5.2 from its official repository
# Using the 1.13.1 version that the error log confirmed is available
RUN pip install --no-cache-dir \
    torch==1.13.1+rocm5.2 \
    torchvision==0.14.1+rocm5.2 \
    torchaudio==0.13.1+rocm5.2 \
    --extra-index-url https://download.pytorch.org/whl/rocm5.2

# Copy the requirements file
COPY requirements.txt .

# Install packages from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Install other major libraries separately for clarity and compatibility
RUN pip install --no-cache-dir \
    jupyterlab \
    "pytorch-lightning<2.1"

# Expose the port Jupyter will run on
EXPOSE 8888