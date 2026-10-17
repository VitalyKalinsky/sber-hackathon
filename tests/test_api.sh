#!/bin/bash
# Шаблон скрипта самопроверки (ШАБЛОН ОТ ОРГАНИЗАТОРОВ)
# Самопроверка API. Запустите после поднятия контейнера.
# Usage: bash tests/test_api.sh [BASE_URL]

set -e
BASE_URL="${1:-http://localhost:8000}"

echo "=== 1. Health check ==="
curl -sf "$BASE_URL/health" | jq .
echo "OK"
echo

echo "=== 2. POST /predict ==="
curl -sf -X POST "$BASE_URL/predict" \
  -F "sales=@data/sample_sales.csv" \
  -F "stock=@data/sample_stock.csv" \
  -o /tmp/forecast.csv
echo "Получен файл: $(wc -l < /tmp/forecast.csv) строк"
head -3 /tmp/forecast.csv
echo "OK"
echo

echo "=== 3. GET /predict?article= ==="
curl -sf "$BASE_URL/predict?article=char_article_1" | head -3
echo "OK"
echo

echo "=== 4. GET /predict?warehouse= ==="
curl -sf "$BASE_URL/predict?warehouse=warehouse_name_1" | head -3
echo "OK"
echo

echo "=== 5. GET /predict?article=&warehouse= ==="
curl -sf "$BASE_URL/predict?article=char_article_1&warehouse=warehouse_name_1" | head -3
echo "OK"
echo

echo "=== 6. GET /predict/all?format=csv ==="
curl -sf "$BASE_URL/predict/all?format=csv" -o /tmp/full_forecast.csv
echo "Получен файл: $(wc -l < /tmp/full_forecast.csv) строк"
echo "OK"
echo

echo "=== 7. GET /predict/all?format=json ==="
curl -sf "$BASE_URL/predict/all?format=json" | jq '.[0:3]'
echo "OK"
echo

echo "=== 8. Error: GET /predict без параметров ==="
curl -sf "$BASE_URL/predict" 2>&1 || true
echo "Ожидается 400"
echo

echo "=== Все тесты пройдены! ==="