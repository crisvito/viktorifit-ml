# FROM python:3.13
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 1: install dependencies
FROM python:3.13-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: copy source + deps
FROM python:3.13-slim
WORKDIR /app
COPY --from=build /usr/local /usr/local
COPY . .
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
