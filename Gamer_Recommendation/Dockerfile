FROM python:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY app.py ./app.py

# Copy the recommendation system files
COPY cache/ ./cache/
COPY recommendation_system/ ./recommendation_system/

# Copy the core files
COPY core/ ./core/

# Copy the models
COPY models/ ./models/

# Copy the services
COPY services/ ./services/

# Copy the utility files
COPY utils/ ./utils/

# Copy the database files
COPY database/ ./database/

# Copy the crew_scoring files
COPY crew_scoring/ ./crew_scoring/

# Copy the data_preprocessing files
COPY data_preprocessing/ ./data_preprocessing/

# Expose the app port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]