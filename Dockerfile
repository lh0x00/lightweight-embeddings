# Use Python 3.10.9 as the base image for consistent runtime environment
FROM python:3.10.9

# Add metadata labels
LABEL maintainer="lamhieu.vk@gmail.com"
LABEL description="Lightweight embeddings service using FastAPI and Hugging Face Transformers"
LABEL version="1.0"

# Setup non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
  PATH=/home/user/.local/bin:$PATH

# Set working directory for all subsequent commands
WORKDIR $HOME/app

# Copy application files
# Copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
COPY --chown=user . .

# Install Python dependencies
# --no-cache-dir reduces image size
# --upgrade ensures latest compatible versions
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Expose service port
EXPOSE 7860

# Launch FastAPI application using uvicorn server
# --host 0.0.0.0: Listen on all network interfaces
# --port 7860: Run on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
