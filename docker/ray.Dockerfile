FROM rayproject/ray-ml:2.35.0-py311

# Install additional dependencies for audio processing and transformers
RUN pip install --no-cache-dir \
    torch==2.3.* \
    torchaudio==2.3.* \
    transformers==4.43.* \
    librosa==0.10.* \
    scikit-learn==1.4.* \
    pandas==2.* \
    numpy==1.26.*

# Copy core library
COPY core/ /app/core/

# Install core library in editable mode
RUN pip install --no-cache-dir -e ./core

# Set working directory
WORKDIR /app

# Default command
CMD ["bash"]
