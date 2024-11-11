# Use a lightweight Python image with Python 3.10.12
FROM python:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the application code, models, and datasets
COPY app.py ./app.py
COPY recommendation.py ./recommendation.py
COPY models/ ./models/

# Install dependencies
RUN pip install fastapi uvicorn pydantic numpy tensorflow==2.17.0 pandas scikit-learn matplotlib jwt kafka-python psycopg2-binary httpx google-cloud-secret-manager apscheduler

# Expose the app port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
