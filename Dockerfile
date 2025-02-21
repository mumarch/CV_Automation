FROM python:3.10-slim-bullseye # Or your preferred Python base image

# Install system dependencies (including a newer sqlite3)
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your Streamlit app files
COPY . /app
WORKDIR /app

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]