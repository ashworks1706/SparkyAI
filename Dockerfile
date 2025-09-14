FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Prevents Chrome from crashing in a sandbox
    CHROMEDRIVER_PATH=/usr/local/bin/chromedriver \
    DISPLAY=:99

# Install system dependencies including Chrome and Chromedriver
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    unzip \
    xvfb \
    libxi6 \
    libxss1 \
    libnss3 \
    libatk1.0-0 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    curl \
    git \
    ca-certificates \
    fonts-liberation \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome - using updated key method since apt-key is deprecated
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub > /usr/share/keyrings/google-chrome.key \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.key] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install ChromeDriver with better error handling
RUN LATEST_CHROMEDRIVER=$(curl -sS https://chromedriver.storage.googleapis.com/LATEST_RELEASE) \
    && wget -q "https://chromedriver.storage.googleapis.com/${LATEST_CHROMEDRIVER}/chromedriver_linux64.zip" \
    && unzip chromedriver_linux64.zip -d /usr/local/bin \
    && chmod +x /usr/local/bin/chromedriver \
    && rm chromedriver_linux64.zip

# Create directories for logs and storage
RUN mkdir -p logs qdrant_storage

# Copy and install requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configure virtual display for headless Chrome (for WSL and headless environments)
RUN echo '#!/bin/bash\nXvfb :99 -screen 0 1280x1024x24 &\nexec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Copy project files
COPY . .

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]
