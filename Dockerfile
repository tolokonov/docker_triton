# Используем базовый образ Python 3.9
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR .

# Копируем requirements.txt в контейнер
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы приложения в контейнер
COPY . .

# Указываем команду для запуска приложения
CMD ["python", "app.py"]