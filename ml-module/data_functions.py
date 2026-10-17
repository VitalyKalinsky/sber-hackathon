###########################
# ИМПОРТЫ И ЗАВИСИМОСТИ
###########################
import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


###########################
# КОНФИГУРАЦИЯ И ПУТИ К ДАННЫМ
###########################
PROJECT_ROOT = Path.cwd().parent

MODELS_DIR = PROJECT_ROOT / "model"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Константы задачи из ТЗ хакатона
FORECAST_HORIZON = 64
NUM_WAREHOUSES = 11
NUM_ARTICLES = 799
EXPECTED_ROWS = NUM_WAREHOUSES * NUM_ARTICLES * FORECAST_HORIZON  # 562 496
MIN_HISTORY_DAYS = 180

REQUIRED_SUBMISSION_COLS = ["date", "warehouse_name", "article", "quantity"]

METRIC_WEIGHTS: Dict[str, float] = {
    "smape": 0.50,
    "wape":  0.20,
    "rmse":  0.15,
    "bias":  0.10,
    "rmsle": 0.05,
}

PATHS = {
    # Исходные датасеты
    "sales_train": PROJECT_ROOT / "data" / "sales_dataset_hakaton_train.csv",
    "stock_train": PROJECT_ROOT / "data" / "stock_dataset_hakaton_train.csv",

    # Сабмит и бейзлайн
    "baseline_submission": PROJECT_ROOT / "data" / "naive_forecast.csv",
    "output_submission": PROJECT_ROOT / "submission.csv",

    # Директории для моделей и логов CatBoost
    "models_dir": MODELS_DIR,
    "catboost_info": MODELS_DIR / "catboost_info",
    "finetune_info": MODELS_DIR / "fine-tuning",
    "model_save_path": MODELS_DIR / "catboost_model.cbm",
}

PATHS["catboost_info"].mkdir(parents=True, exist_ok=True)
PATHS["finetune_info"].mkdir(parents=True, exist_ok=True)


###########################
# ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ (FEATURES)
###########################
TARGET_COL = "quantity"
ID_COLS = ["date", "warehouse_name", "article"]

CAT_COLS = [
    "warehouse_name",
    "article",
    "cities_name",
    "category_lvl_1",
    "category_lvl_2",
    "category_lvl_3",
    "characteristic",
]

FEATURE_COLS = [
    # Категориальные признаки
    *CAT_COLS,
    # Временные признаки (TODO: дополнить при разработке)
    "dayofweek",
    "day",
    "month",
    "is_weekend",
    # Признаки остатков и цен (TODO: дополнить при разработке)
    "stock",
    "cost_price",
    "discount_sum",
    "volume_weight",
    # Лаги и оконные статистики продаж (TODO: дополнить при разработке)
    "quantity_lag_64",
    "quantity_roll_mean_7",
    "quantity_roll_mean_28",
]


###########################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И МЕТРИКИ
###########################
def build_forecast_grid(
    sales_df: pd.DataFrame,
    start_date: pd.Timestamp,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Формирует полную декартову сетку (date × warehouse_name × article)
    на заданный горизонт прогнозирования (64 дня).
    """
    # TODO: при необходимости скорректировать формирование списка уникальных пар
    forecast_dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    combos = sales_df[["warehouse_name", "article"]].drop_duplicates()

    grid = combos.assign(_key=1).merge(
        pd.DataFrame({"date": forecast_dates, "_key": 1}),
        on="_key",
    ).drop(columns=["_key"])

    return grid[["date", "warehouse_name", "article"]].sort_values(
        ["date", "warehouse_name", "article"]
    ).reset_index(drop=True)


def compute_all_metrics(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
) -> Dict[str, float]:
    """Считает все 5 метрик хакатона (SMAPE, WAPE, RMSE, |Bias|, RMSLE)."""
    # TODO: перенести или подключить функции расчёта из score.py
    return {
        "smape": 0.0,
        "wape": 0.0,
        "rmse": 0.0,
        "bias": 0.0,
        "rmsle": 0.0,
    }


def validate_submission(df: pd.DataFrame) -> List[str]:
    """Проверяет формат сабмита на соответствие требованиям ТЗ
    (колонки, отсутствие NaN и отрицательных значений, число строк, непрерывность дат).
    """
    # TODO: добавить детальную валидацию из ТЗ
    warnings_list: List[str] = []
    return warnings_list


###########################
# ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
###########################
def load_and_preprocess_sales(sales_path: Path | io.BytesIO | str) -> pd.DataFrame:
    """Загружает данные о продажах (sales_dataset_hakaton.csv),
    приводит типы дат и агрегирует департаменты (composite_division)
    до уровня (date, warehouse_name, article).
    """
    df = pd.read_csv(sales_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # TODO: реализовать агрегацию по ["date", "warehouse_name", "article"]
    # (суммирование quantity, sales_sum, discount_sum и агрегация цен/категорий),
    # а также фильтрацию аномалий и заполнение пропущенных дней нулевыми продажами.

    return df


def load_and_preprocess_stock(stock_path: Path | io.BytesIO | str) -> pd.DataFrame:
    """Загружает данные об остатках на складах (stock_dataset_hakaton.csv),
    переименовывает char_article в article и выполняет базовую очистку.
    """
    df = pd.read_csv(stock_path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    if "char_article" in df.columns:
        df = df.rename(columns={"char_article": "article"})

    # TODO: реализовать обработку дубликатов, пропусков дат и аномалий в stock

    return df


###########################
# ГЕНЕРАЦИЯ ПРИЗНАКОВ И СБОРКА ДАТАСЕТА
###########################
def compute_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует календарные признаки, лаги целевой переменной,
    скользящие статистики по сериям (warehouse_name × article),
    а также признаки динамики остатков и цен.
    """
    df_feat = df.copy()

    # TODO: реализовать генерацию признаков:
    # 1. Календарные признаки (день недели, месяц, праздники, циклическое кодирование)
    # 2. Лаги и скользящие окна по (warehouse_name, article) без утечки из будущего
    # 3. Признаки взаимодействия с остатками (stock == 0, отношение продаж к остаткам)
    # 4. Агрегаты по категориям (category_lvl_1/2/3) и складам

    return df_feat


def build_dataset(
    sales_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """Объединяет предобработанные продажи и остатки в единый датафрейм,
    достраивает сетку дат (при необходимости) и вызывает генерацию признаков.
    """
    # TODO: реализовать объединение sales_df и stock_df по ["date", "warehouse_name", "article"],
    # генерацию будущей сетки на 64 дня при is_train=False и вызов compute_time_series_features

    merged = sales_df.merge(
        stock_df,
        on=["date", "warehouse_name", "article"],
        how="left",
    )
    merged = compute_time_series_features(merged)
    return merged


###########################
# ОБУЧЕНИЕ И ИНФЕРЕНС МОДЕЛИ
###########################
def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    target_col: str = TARGET_COL,
    fit_on_all: bool = False,
) -> CatBoostRegressor:
    """Обучает модель прогнозирования количества проданных единиц (quantity),
    выполняет валидацию по 5 метрикам хакатона и сохраняет веса.
    """
    # TODO: реализовать разбиение по времени (например, последние 64 дня в val_df),
    # трансформацию таргета (например, log1p), настройку гиперпараметров и обучение модели

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=100,
        train_dir=str(PATHS["catboost_info"]),
    )

    # TODO: вызвать model.fit(...), посчитать compute_all_metrics(...) и сохранить модель
    return model


def predict(model: CatBoostRegressor, test_df: pd.DataFrame) -> np.ndarray:
    """Выполняет инференс модели на подготовленном датафрейме признаков
    и применяет базовую постобработку (отсечение отрицательных значений).
    """
    # TODO: реализовать получение предсказаний, обратную трансформацию таргета (если применялся log1p)
    preds = np.zeros(len(test_df), dtype=float)
    return np.clip(preds, a_min=0.0, a_max=None)


def fine_tune(
    base_model: CatBoostRegressor | Path | str,
    new_data_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    target_col: str = TARGET_COL,
    additional_iterations: int = 150,
    learning_rate: float = 0.025,
    output_model_path: Optional[Path | str] = None,
) -> CatBoostRegressor:
    """Дообучает существующую модель на свежих данных (например, перед финальным сабмитом
    или при получении полугодового окна данных в контейнере).
    """
    # TODO: реализовать логику дообучения через init_model по аналогии с основным пайплайном
    if isinstance(base_model, CatBoostRegressor):
        return base_model

    model = CatBoostRegressor()
    model.load_model(str(base_model))
    return model


###########################
# ЭКСПОРТ САБМИТА
###########################
def make_submission(
    model: CatBoostRegressor,
    test_df: pd.DataFrame,
    output_path: Path = PATHS["output_submission"],
) -> pd.DataFrame:
    """Формирует итоговый файл прогноза на 64 дня в формате:
    date,warehouse_name,article,quantity (562 496 строк, без NaN и значений < 0).
    """
    sub = test_df[["date", "warehouse_name", "article"]].copy()
    sub["quantity"] = predict(model, test_df)

    # Постобработка согласно правилам хакатона
    sub["quantity"] = sub["quantity"].fillna(0.0).clip(lower=0.0)
    sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")

    # TODO: добавить вызов validate_submission(sub) перед сохранением
    sub.to_csv(output_path, index=False)
    print(f"Сабмит успешно сохранен в {output_path}. Строк: {len(sub)}")

    return sub