"""Entrena el modelo de ACV y lo guarda en la carpeta models/."""

import json
from pathlib import Path

import joblib

from src import config
from src.data_loading import load_raw_data, validate_columns, split_features_target
from src.preprocessing import fill_missing_values, detect_feature_types
from src.model_building import build_preprocessor, build_classifier, build_pipeline
from src.evaluation import train_test_split_stratified, evaluate_model


def main():
    print("📥 Cargando datos...")
    df = load_raw_data()
    validate_columns(df)

    df_clean = fill_missing_values(df)
    X, y = split_features_target(df_clean)

    numeric_cols, categorical_cols = detect_feature_types(X)

    print("🤖 Preparando pipeline...")
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    classifier = build_classifier()
    pipeline = build_pipeline(preprocessor, classifier)

    print("✂ Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    print("🚀 Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    print("📊 Evaluando modelo...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Casos positivos en test: {metrics['support']}")

    # Guardar modelo
    Path(config.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    print(f"💾 Modelo guardado en: {config.MODEL_PATH}")

    # Guardar métricas
    metrics_path = Path(config.MODEL_PATH).with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"📁 Métricas guardadas en: {metrics_path}")
    print("✅ Entrenamiento completado.")


if __name__ == "__main__":
    main()
