FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "front.py", "--server.port=8501", "--server.address=0.0.0.0"]