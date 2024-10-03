import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_breast_cancer
import joblib

# Загрузим реальный набор данных - Breast Cancer Wisconsin Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Разделим данные на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Сохраним обученную модель
joblib.dump(model, '../../models/model.pkl')

# Сделаем предсказания на тестовых данных
y_pred = model.predict(X_test)

# Оценим качество модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Выведем метрики
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model Precision: {precision:.4f}")
print(f"Model Recall: {recall:.4f}")
print(f"Model F1-Score: {f1:.4f}")

# Сохраним метрики модели в CSV-файл для последующего анализа
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Score": [accuracy, precision, recall, f1]
})

metrics.to_csv('../../models/model_metrics.csv', index=False)

# Опционально: можно также сохранить метки классов и предсказания для дальнейшего анализа
results = pd.DataFrame({
    "True Label": y_test,
    "Predicted Label": y_pred
})

results.to_csv('../../models/model_predictions.csv', index=False)
