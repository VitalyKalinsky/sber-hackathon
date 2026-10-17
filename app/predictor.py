# Логика загрузки модели и инференса(ШАБЛОН ОТ ОРГАНИЗАТОРОВ)


import json
from pathlib import Path
from typing import Optional
import pandas as pd


class Predictor:
    """Обёртка над моделью для инференса."""

    def __init__(self, model, preprocessor, config):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self._cached_forecast: Optional[pd.DataFrame] = None

    @classmethod
    def load(cls, model_dir: Path) -> "Predictor":
        """Загрузка модели из директории. Реализовать под свой формат."""
        config = json.loads((model_dir / "config.json").read_text())

        # Пример для pickled-модели:
        import pickle
        model = pickle.loads((model_dir / "model.pkl").read_bytes())
        preprocessor = pickle.loads(
            (model_dir / "preprocessor.pkl").read_bytes()
        )
        return cls(model, preprocessor, config)

    def predict(
        self, df_sales: pd.DataFrame, df_stock: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Основная логика прогноза.
        Принимает sales и stock, возвращает DataFrame в формате submission.
        """
        # ── Заглушка: замените на свою логику ──
        # 1. Объединить sales + stock
        # 2. Сгенерировать признаки
        # 3. Применить модель
        # 4. Собрать submission-формат

        forecast_dates = pd.date_range(
            start=df_sales["date"].max(),
            periods=65,
            freq="D"
        )[1:]  # 64 дня

        combos = (
            df_sales[["warehouse_name", "article"]]
            .drop_duplicates()
        )

        rows = []
        for _, row in combos.iterrows():
            for d in forecast_dates:
                rows.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "warehouse_name": row["warehouse_name"],
                    "article": row["article"],
                    "quantity": 0.0,  # ← замените на прогноз модели
                })

        df_forecast = pd.DataFrame(rows)
        self._cached_forecast = df_forecast
        return df_forecast

    def get_cached_forecast(self) -> pd.DataFrame:
        """Возвращает кэшированный прогноз (для GET-эндпоинтов)."""
        if self._cached_forecast is None:
            raise RuntimeError(
                "Прогноз не рассчитан. Сначала вызовите POST /predict."
            )
        return self._cached_forecast