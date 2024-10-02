# Dockerfile for Streamlit app
FROM python:3.9

WORKDIR /app

# Copy the app.py file from the correct folder inside the container
COPY ./code/deployment/app/app.py /app/app.py

RUN pip install streamlit requests

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
