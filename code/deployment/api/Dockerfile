## Dockerfile for FastAPI API
#FROM python:3.9
#
#WORKDIR /app
#
## Copy the app.py file from the correct folder inside the container
#COPY ./code/deployment/api/app.py /app/app.py
#COPY ./models /app/models
#
#RUN pip install fastapi uvicorn joblib
#
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.9

WORKDIR /app

COPY ./code/deployment/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./code/deployment/api /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
