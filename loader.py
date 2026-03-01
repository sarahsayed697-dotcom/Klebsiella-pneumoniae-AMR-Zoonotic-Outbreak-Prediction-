"""
src/data/loader.py
------------------
Data loading utilities for the Klebsiella AMR outbreak prediction pipeline.
Handles clinical, genomic, climate, environmental, animal reservoir,
and socioeconomic data streams.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import yaml
import logging

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR = ROOT / "data" / "external"


# ── Master loader ──────────────────────────────────────────────────────────────

def load_region_data(
    lat: float,
    lon: float,
    months_back: int = 24,
    config_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load and merge all data streams for a given geographic region.

    Parameters
    ----------
    lat : float
        Latitude of the target region.
    lon : float
        Longitude of the target region.
    months_back : int
        How many months of historical data to include.
    config_path : Path, optional
        Path to YAML config file. Defaults to configs/default.yaml.

    Returns
    -------
    dict with keys: 'clinical', 'genomic', 'climate', 'environment',
                    'animal', 'socioeconomic'
    """
    config = _load_config(config_path)
    logger.info(f"Loading data for region ({lat:.2f}°N, {lon:.2f}°E), "
                f"lookback={months_back} months")

    return {
        "clinical":      load_clinical_data(lat, lon, months_back),
        "genomic":       load_genomic_data(lat, lon),
        "climate":       load_climate_data(lat, lon, months_back),
        "environment":   load_environmental_data(lat, lon, months_back),
        "animal":        load_animal_reservoir_data(lat, lon),
        "socioeconomic": load_socioeconomic_data(lat, lon),
    }


# ── Individual loaders ─────────────────────────────────────────────────────────

def load_clinical_data(
    lat: float, lon: float, months_back: int = 24
) -> pd.DataFrame:
    """
    Load clinical & microbiological data.

    Variables (25+):
        - patient_demographics, apache_ii_score, device_utilization
        - infection_rates, ast_results (MICs for 6 AB classes)
        - colonization_data, outbreak_records, icu_occupancy
        - hand_hygiene_compliance, ab_consumption_ddd
    Sources: EHR, LIS, AST systems, NHSN surveillance
    """
    path = PROCESSED_DIR / "clinical" / f"clinical_{lat:.1f}_{lon:.1f}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded clinical data: {df.shape}")
        return df

    logger.warning(f"Clinical data file not found at {path}. "
                   "Returning empty template.")
    return _empty_clinical_template(months_back)


def load_genomic_data(lat: float, lon: float) -> pd.DataFrame:
    """
    Load genomic & AMR gene data.

    Variables (50+):
        - whole_genome_sequences, mlst_profiles (sequence types)
        - resistance_genes: ESBL (blaCTX-M), carbapenemases (blaKPC, blaNDM,
          blaOXA-48), colistin (mcr-1 to mcr-10)
        - virulence_genes: iucABCD, iroNB, rmpA, rmpA2 (hypervirulence markers)
        - plasmid_replicons: IncFII, IncX3, IncHI1B
        - snps, phylogenetic_placement
    Sources: NCBI GenBank/SRA, PubMLST, CARD, VFDB, PATRIC
    """
    path = PROCESSED_DIR / "genomic" / f"genomic_{lat:.1f}_{lon:.1f}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded genomic data: {df.shape}")
        return df

    logger.warning("Genomic data not found. Returning empty template.")
    return _empty_genomic_template()


def load_climate_data(
    lat: float, lon: float, months_back: int = 120
) -> pd.DataFrame:
    """
    Load climate & environmental variables.

    Variables (15):
        - mean_temperature_c, temperature_anomaly, heatwave_frequency
        - precipitation_mm, humidity_pct
        - bacterial_growth_rate_proxy, mutation_rate_proxy, hgt_rate_proxy
        - el_nino_index, uhi_effect, wind_speed_ms
    Sources: NOAA, NASA GISS, ERA5 reanalysis (Copernicus)
    Range: Temperature -20 to 45°C; Precip 0-5000mm; Humidity 10-100%
    """
    path = PROCESSED_DIR / "climate" / f"climate_{lat:.1f}_{lon:.1f}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded climate data: {df.shape}")
        return df

    logger.warning("Climate data not found. Returning empty template.")
    return _empty_climate_template(months_back)


def load_environmental_data(
    lat: float, lon: float, months_back: int = 24
) -> pd.DataFrame:
    """
    Load environmental contamination data.

    Variables (40+):
        Antibiotic residues by compartment:
          - soil_betalactams_ug_kg  (0–1000)
          - soil_fluoroquinolones_ug_kg  (0–500)
          - soil_colistin_ug_kg  (0–100)
          - surface_water_betalactams_ng_l  (0–5000)
          - hospital_ww_betalactams_mg_l  (0–100)
        Heavy metals: copper_mg_kg, zinc_mg_kg (co-selection markers)
        Industrial effluent concentrations
        Wastewater treatment metrics
    Sources: EPA databases, WW treatment plant records, water quality networks
    """
    path = PROCESSED_DIR / "environment" / f"env_{lat:.1f}_{lon:.1f}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded environmental data: {df.shape}")
        return df

    logger.warning("Environmental data not found. Returning empty template.")
    return _empty_environmental_template(months_back)


def load_animal_reservoir_data(lat: float, lon: float) -> pd.DataFrame:
    """
    Load animal reservoir surveillance data.

    8 Animal Types with prevalence, contact frequency, transmission routes:
        - cattle      (0–60% prevalence) — milk, meat, manure
        - swine       (0–80%)            — pork, workers, waste
        - poultry     (0–70%)            — eggs, meat, litter
        - dogs        (0–30%)            — direct contact
        - cats        (0–25%)            — scratches, litter
        - rodents     (0–50%)            — contamination
        - wild_birds  (0–40%)            — droppings, migration
        - other       (0–20%)
    Environmental persistence: soil 30–365 days, water 7–180 days
    Sources: NAHMS, FAO, OIE, veterinary diagnostic labs
    """
    path = PROCESSED_DIR / "animal" / f"animal_{lat:.1f}_{lon:.1f}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded animal reservoir data: {df.shape}")
        return df

    logger.warning("Animal data not found. Returning empty template.")
    return _empty_animal_template()


def load_socioeconomic_data(lat: float, lon: float) -> pd.DataFrame:
    """
    Load socioeconomic indicators.

    Variables (20+):
        - population_density_per_km2  (10–10,000)
        - sanitation_coverage_pct     (0–100)
        - gdp_per_capita_usd
        - healthcare_expenditure_pct_gdp
        - icu_beds_per_100k
        - ab_consumption_ddd_per_1000_pt_days  (0–2000)
        - physicians_per_1000
        - human_development_index
    Sources: World Bank, WHO, National census, UN databases
    """
    path = PROCESSED_DIR / "socioeconomic" / f"socio_{lat:.1f}_{lon:.1f}.parquet"

    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded socioeconomic data: {df.shape}")
        return df

    logger.warning("Socioeconomic data not found. Returning empty template.")
    return _empty_socioeconomic_template()


# ── Template generators (for missing data) ─────────────────────────────────────

def _empty_clinical_template(months_back: int) -> pd.DataFrame:
    cols = [
        "date", "icu_occupancy_pct", "hand_hygiene_compliance_pct",
        "ab_consumption_ddd", "device_utilization_ratio",
        "infection_rate_per_1000", "colonization_pressure_pct",
        "apache_ii_mean", "kp_isolates_count", "mdr_isolates_count",
        "carbapenem_resistant_count", "hypervirulent_count"
    ]
    return pd.DataFrame(columns=cols)


def _empty_genomic_template() -> pd.DataFrame:
    cols = [
        "isolate_id", "collection_date", "sequence_type", "capsule_type",
        "blaCTX_M", "blaKPC", "blaNDM", "blaOXA_48", "mcr_1",
        "iucA", "rmpA", "rmpA2", "iroN",
        "plasmid_IncFII", "plasmid_IncX3",
        "snp_cluster", "source"
    ]
    return pd.DataFrame(columns=cols)


def _empty_climate_template(months_back: int) -> pd.DataFrame:
    cols = [
        "date", "mean_temp_c", "temp_anomaly_c", "max_temp_c",
        "heatwave_events", "precipitation_mm", "humidity_pct",
        "wind_speed_ms", "uhi_effect_c", "el_nino_index"
    ]
    return pd.DataFrame(columns=cols)


def _empty_environmental_template(months_back: int) -> pd.DataFrame:
    cols = [
        "date", "soil_betalactams_ug_kg", "soil_fluoroquinolones_ug_kg",
        "soil_colistin_ug_kg", "surface_water_betalactams_ng_l",
        "surface_water_fluoroquinolones_ng_l", "hospital_ww_betalactams_mg_l",
        "hospital_ww_fluoroquinolones_mg_l", "copper_mg_kg", "zinc_mg_kg",
        "pharma_effluent_score", "cafo_manure_tons_ha"
    ]
    return pd.DataFrame(columns=cols)


def _empty_animal_template() -> pd.DataFrame:
    animals = ["cattle", "swine", "poultry", "dogs", "cats",
               "rodents", "wild_birds", "other"]
    cols = (["survey_date"] +
            [f"{a}_prevalence_pct" for a in animals] +
            [f"{a}_contacts_per_month" for a in animals] +
            ["livestock_density_per_km2", "ag_ab_use_kg_ton_biomass",
             "manure_application_rate", "food_contamination_incidents"])
    return pd.DataFrame(columns=cols)


def _empty_socioeconomic_template() -> pd.DataFrame:
    cols = [
        "year", "population_density_per_km2", "sanitation_coverage_pct",
        "gdp_per_capita_usd", "healthcare_expenditure_pct_gdp",
        "icu_beds_per_100k", "ab_consumption_ddd_per_1000",
        "physicians_per_1000", "hdi"
    ]
    return pd.DataFrame(columns=cols)


def _load_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        config_path = ROOT / "configs" / "default.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}
