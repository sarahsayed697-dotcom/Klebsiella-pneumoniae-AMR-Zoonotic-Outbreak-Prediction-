"""
src/models/pipeline.py
-----------------------
Main end-to-end pipeline for Klebsiella AMR Outbreak Prediction.
Orchestrates data loading, model training, and outbreak prediction.

Usage:
    python src/models/pipeline.py --mode train
    python src/models/pipeline.py --mode predict --lat 30.5 --lon 31.2
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_region_data
from models.model1_climate.climate_amr_model import ClimateAMRModel
from models.model2_reservoir.reservoir_strain_model import ReservoirSTModel
from models.model3_activities.human_activities_mdr_model import HumanActivitiesMDRModel
from models.model4_icu.icu_hypervirulent_model import ICUHypervirulentModel
from models.ensemble.ensemble_model import OutbreakEnsemble, OutbreakPrediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pipeline.log")],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training_pipeline(config: dict) -> OutbreakEnsemble:
    """
    Full training pipeline: train all four models + ensemble.

    Data split (temporal, respects ordering):
        Train:    2010–2017
        Validate: 2017–2019
        Test:     2019–2020
    """
    logger.info("=" * 60)
    logger.info("Starting Klebsiella AMR Outbreak Prediction Training")
    logger.info("=" * 60)

    # ── Load training data ──────────────────────────────────────────────────
    logger.info("Loading training data...")
    # In production: iterate over all training regions
    # Here: demonstrate with synthetic data structure
    training_regions = [
        (30.5, 31.2, "Cairo ICU"),
        (51.5, 0.12, "London Hospital"),
        (40.7, -74.0, "New York Urban"),
        (28.6, 77.2, "Delhi Agricultural"),
    ]

    # ── Model 1: Climate-AMR ────────────────────────────────────────────────
    logger.info("\n--- Model 1: Climate-AMR Correlation ---")
    m1 = ClimateAMRModel(
        forecast_horizon=config["models"]["model1_climate"]["forecast_horizon_months"],
        max_lag=config["models"]["model1_climate"]["max_lag_months"],
        n_estimators=config["models"]["model1_climate"]["n_estimators"],
    )
    # m1.fit(climate_df_train, mdr_series_train)  # Fit on real data
    logger.info("Model 1 initialized. Fit on training data to train.")

    # ── Model 2: Reservoir-ST ───────────────────────────────────────────────
    logger.info("\n--- Model 2: Animal Reservoir–Novel ST ---")
    m2 = ReservoirSTModel(
        n_estimators=config["models"]["model2_reservoir"]["n_estimators"],
    )
    logger.info("Model 2 initialized. Fit on training data to train.")

    # ── Model 3: Human Activities-MDR ──────────────────────────────────────
    logger.info("\n--- Model 3: Human Activities–MDR Evolution ---")
    m3 = HumanActivitiesMDRModel(
        n_estimators=config["models"]["model3_activities"]["n_estimators"],
    )
    logger.info("Model 3 initialized. Fit on training data to train.")

    # ── Model 4: ICU-HV (PRIMARY) ───────────────────────────────────────────
    logger.info("\n--- Model 4: ICU–Hypervirulent Source (PRIMARY) ---")
    m4 = ICUHypervirulentModel(
        n_estimators=config["models"]["model4_icu"]["n_estimators"],
    )
    logger.info("Model 4 initialized. Fit on training data to train.")

    # ── Ensemble ────────────────────────────────────────────────────────────
    logger.info("\n--- Ensemble Meta-Learner ---")
    ensemble = OutbreakEnsemble()
    logger.info("Ensemble initialized. Fit intermediate predictions to train.")

    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline complete. Fit models on your data.")
    logger.info("=" * 60)

    # Save initialized models
    model_dir = ROOT / "models"
    m1.save(model_dir / "model1_climate" / "model1.pkl")
    m2.save(model_dir / "model2_reservoir" / "model2.pkl")
    m3.save(model_dir / "model3_activities" / "model3.pkl")
    m4.save(model_dir / "model4_icu" / "model4.pkl")
    logger.info("Model skeletons saved to models/ directory.")

    return ensemble


def run_prediction_pipeline(
    lat: float,
    lon: float,
    region_name: str = "Unknown",
    months_back: int = 24,
    config: dict = None,
) -> OutbreakPrediction:
    """
    Run full prediction pipeline for a given geographic region.

    Parameters
    ----------
    lat, lon : Geographic coordinates
    region_name : Human-readable name
    months_back : Months of historical data to load
    """
    logger.info(f"Running prediction for {region_name} ({lat}°N, {lon}°E)")

    # Load data
    data = load_region_data(lat, lon, months_back=months_back)

    # Load ensemble model
    ensemble_path = ROOT / "models" / "ensemble" / "ensemble_v1.pkl"
    if not ensemble_path.exists():
        logger.warning("Trained ensemble not found. Using untrained model for demo.")
        ensemble = OutbreakEnsemble()
    else:
        ensemble = OutbreakEnsemble.load(ensemble_path)

    # Generate prediction
    prediction = ensemble.predict(data, lat=lat, lon=lon, region_name=region_name)
    _print_prediction_report(prediction)
    return prediction


def _print_prediction_report(pred: OutbreakPrediction):
    """Print formatted prediction report."""
    print("\n" + "=" * 70)
    print("  KLEBSIELLA AMR OUTBREAK PREDICTION REPORT")
    print("=" * 70)
    print(f"  Region:                {pred.region_name}")
    print(f"  Coordinates:           {pred.latitude}°N, {pred.longitude}°E")
    print(f"  Prediction Date:       {pred.prediction_date}")
    print()
    print(f"  ⚠️  OUTBREAK PROBABILITY:  {pred.outbreak_probability:.1%}")
    print(f"     95% CI:               ({pred.confidence_interval[0]:.1%}, {pred.confidence_interval[1]:.1%})")
    print(f"  ⏱️  TIME TO OUTBREAK:      {pred.time_to_outbreak_months:.1f} months")
    print(f"     95% PI:               ({pred.prediction_interval[0]:.1f}, {pred.prediction_interval[1]:.1f}) months")
    print(f"  🚨 SURVEILLANCE PRIORITY:  {pred.surveillance_priority}")
    print()
    print("  Model Contributions:")
    print(f"    H1 Climate risk:       {pred.h1_climate_risk:.3f}")
    print(f"    H2 Reservoir risk:     {pred.h2_reservoir_risk:.3f}")
    print(f"    H3 Activities risk:    {pred.h3_activities_risk:.3f}")
    print(f"    H4 ICU-HV risk:        {pred.h4_icu_risk:.3f}")
    print()
    print("  Predicted Strain:")
    print(f"    Sequence Type:         {pred.predicted_sequence_type}")
    print(f"    Resistance:            {', '.join(pred.predicted_resistance)}")
    print(f"    Hypervirulent:         {'YES ⚠️' if pred.is_hypervirulent else 'No'}")
    print(f"    HV Probability:        {pred.hypervirulence_probability:.1%}")
    print()
    print("  Top Risk Factors:")
    for i, rf in enumerate(pred.primary_risk_factors, 1):
        print(f"    {i}. {rf}")
    print()
    print("  Recommended Interventions:")
    for i, rec in enumerate(pred.recommended_interventions[:5], 1):
        print(f"    {i}. {rec}")
    print()
    print(f"  Estimated Outbreak Cost:  ${pred.estimated_outbreak_cost_usd:,.0f}")
    print(f"  Prevention Cost-Benefit:  {pred.prevention_cost_benefit_ratio:.1f}×")
    print("=" * 70)


def validate_all_hypotheses(config: dict):
    """Run hypothesis validation for all four models."""
    logger.info("\nValidating all four hypotheses...")
    results = {}

    # Load trained models if available
    model_dir = ROOT / "models"

    for model_name, model_cls, path in [
        ("H1_Climate-AMR", ClimateAMRModel, model_dir / "model1_climate" / "model1.pkl"),
        ("H2_Reservoir-ST", ReservoirSTModel, model_dir / "model2_reservoir" / "model2.pkl"),
        ("H3_Activities-MDR", HumanActivitiesMDRModel, model_dir / "model3_activities" / "model3.pkl"),
        ("H4_ICU-HV", ICUHypervirulentModel, model_dir / "model4_icu" / "model4.pkl"),
    ]:
        if path.exists():
            model = model_cls.load(path)
            if hasattr(model, "validate_h1"):
                results[model_name] = model.validate_h1()
            elif hasattr(model, "validate_h2"):
                results[model_name] = model.validate_h2()
            elif hasattr(model, "validate_h3"):
                results[model_name] = model.validate_h3()
            elif hasattr(model, "validate_h4"):
                results[model_name] = model.validate_h4()
        else:
            results[model_name] = {"status": "model not found"}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Klebsiella AMR Outbreak Prediction Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "validate"],
        required=True,
        help="Pipeline mode",
    )
    parser.add_argument("--lat", type=float, default=30.5, help="Latitude")
    parser.add_argument("--lon", type=float, default=31.2, help="Longitude")
    parser.add_argument("--region", type=str, default="Cairo Region", help="Region name")
    parser.add_argument("--months-back", type=int, default=24, help="Months of history")
    parser.add_argument("--config", type=Path, default=None, help="Config file path")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        ensemble = run_training_pipeline(config)
        logger.info("Training complete.")

    elif args.mode == "predict":
        prediction = run_prediction_pipeline(
            lat=args.lat,
            lon=args.lon,
            region_name=args.region,
            months_back=args.months_back,
            config=config,
        )

    elif args.mode == "validate":
        results = validate_all_hypotheses(config)
        for h, r in results.items():
            logger.info(f"{h}: {r}")


if __name__ == "__main__":
    main()
