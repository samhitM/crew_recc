FROM python:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY Gamer_Recommendation/requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main application file
COPY Gamer_Recommendation/app.py ./app.py

# Copy the recommendation system files
COPY Gamer_Recommendation/recommendation_cache.py ./recommendation_cache.py
COPY Gamer_Recommendation/recommendation_system/ ./recommendation_system/

# Copy the core files
COPY Gamer_Recommendation/core/ ./core/

# Copy the models
COPY Gamer_Recommendation/models/ ./models/

# Copy the services
COPY Gamer_Recommendation/services/ ./services/

# Copy the utility files
COPY Gamer_Recommendation/util/ ./util/

# Copy the data_preprocessing files
COPY Gamer_Recommendation/data_preprocessing/ ./data_preprocessing/

# Expose the app port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
