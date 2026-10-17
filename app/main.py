# FastAPI-приложение (ШАБЛОН ОТ ОРГАНИЗАТОРОВ)


import io
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse

from .predictor import Predictor
from .validators import validate_sales_csv, validate_stock_csv

app = FastAPI(
    title="Hackathon Forecast API",
    description="Прогноз продаж по складам и товарам",
    version="1.0.0",
)

# Загрузка модели при старте
predictor = Predictor.load(Path("model"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_post(
    sales: bytes = File(...),
    stock: bytes = File(...),
):
    """
    Принимает два CSV (sales + stock) и возвращает CSV с прогнозом на 64 дня.
    """
    try:
        df_sales = pd.read_csv(io.BytesIO(sales))
        df_stock = pd.read_csv(io.BytesIO(stock))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка парсинга CSV: {str(e)}"
        )

    errors = validate_sales_csv(df_sales) + validate_stock_csv(df_stock)
    if errors:
        raise HTTPException(
            status_code=422,
            detail={"errors": errors, "code": "VALIDATION_ERROR"}
        )

    df_forecast = predictor.predict(df_sales, df_stock)

    output = io.StringIO()
    df_forecast.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=forecast.csv"}
    )


@app.get("/predict")
async def predict_get(
    article: Optional[str] = Query(None, description="Артикул товара"),
    warehouse: Optional[str] = Query(None, description="Наименование склада"),
):
    """
    Возвращает прогноз по конкретному артикулу и/или складу в формате CSV.
    Хотя бы один параметр обязателен.
    """
    if not article and not warehouse:
        raise HTTPException(
            status_code=400,
            detail="Укажите хотя бы один параметр: article или warehouse"
        )

    df_forecast = predictor.get_cached_forecast()

    mask = pd.Series(True, index=df_forecast.index)
    if article:
        mask &= df_forecast["article"] == article
    if warehouse:
        mask &= df_forecast["warehouse_name"] == warehouse

    result = df_forecast[mask].copy()

    output = io.StringIO()
    result.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=filtered_forecast.csv"}
    )


@app.get("/predict/all")
async def predict_all(
    format: str = Query("csv", description="Формат ответа: csv или json"),
):
    """
    Возвращает весь прогноз целиком (562 496 строк).
    """
    df_forecast = predictor.get_cached_forecast()

    if format == "csv":
        output = io.StringIO()
        df_forecast.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition":
                    "attachment; filename=full_forecast.csv"
            }
        )
    elif format == "json":
        result = df_forecast.to_dict(orient="records")
        return JSONResponse(content=result)
    else:
        raise HTTPException(
            status_code=400,
            detail="Параметр format должен быть 'csv' или 'json'"
        )