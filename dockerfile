# Train Models
FROM python:3.10 AS model_training
WORKDIR /app
COPY src/model_training.py /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
RUN pip install ucimlrepo
RUN python model_training.py\

# Predictions
FROM python:3.10 AS serving
WORKDIR /app
COPY --from=model_training /app/banknote_model.pth /app/
COPY --from=model_training /app/scaler.pkl /app/
COPY src/main.py /app/
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY src/templates /app/templates
COPY src/statics /app/statics
EXPOSE 4000
CMD ["python", "main.py"]