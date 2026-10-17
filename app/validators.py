# Валидация входных CSV (ШАБЛОН ОТ ОРГАНИЗАТОРОВ)


import pandas as pd
from typing import List

SALES_REQUIRED_COLS = [
    "date", "warehouse_name", "article", "quantity",
    "sales_sum", "cost_price", "discount_sum", "volume_weight",
    "composite_division", "cities_name",
    "category_lvl_1", "category_lvl_2", "category_lvl_3",
    "characteristic",
]

STOCK_REQUIRED_COLS = ["date", "warehouse_name", "char_article", "stock"]

MIN_HISTORY_DAYS = 180


def validate_sales_csv(df: pd.DataFrame) -> List[str]:
    errors = []
    missing = set(SALES_REQUIRED_COLS) - set(df.columns)
    if missing:
        errors.append(f"Отсутствуют колонки в sales: {missing}")
        return errors

    try:
        dates = pd.to_datetime(df["date"], format="%Y-%m-%d")
    except Exception:
        errors.append("Неверный формат даты в sales (ожидается YYYY-MM-DD)")
        return errors

    history_days = (dates.max() - dates.min()).days
    if history_days < MIN_HISTORY_DAYS:
        errors.append(
            f"Недостаточно истории: {history_days} дней, "
            f"минимум {MIN_HISTORY_DAYS}"
        )

    if (df["quantity"] < 0).any():
        errors.append("Найдены отрицательные значения в quantity")

    return errors


def validate_stock_csv(df: pd.DataFrame) -> List[str]:
    errors = []
    missing = set(STOCK_REQUIRED_COLS) - set(df.columns)
    if missing:
        errors.append(f"Отсутствуют колонки в stock: {missing}")
        return errors

    try:
        pd.to_datetime(df["date"], format="%Y-%m-%d")
    except Exception:
        errors.append("Неверный формат даты в stock (ожидается YYYY-MM-DD)")
        return errors

    if (df["stock"] < 0).any():
        errors.append("Найдены отрицательные значения в stock")

    return errors