# trimmed_outlier_detection.py
import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def setup_logging(output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(fmt)
    logger.handlers = []  # reset
    logger.addHandler(ch)

# ------------------ Univariate detectors (kept) ----------------------------

def _to_float_array(x):
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    return np.array(x, dtype=float)

def detect_outliers_iqr(series: Union[pd.Series, List, np.ndarray], multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    arr = _to_float_array(series)
    mask = ~np.isnan(arr)
    clean = arr[mask]
    if clean.size == 0:
        return np.array([]), np.array([])
    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    out_mask = (arr < lower) | (arr > upper)
    out_mask = out_mask & mask
    indices = np.where(out_mask)[0]
    values = arr[indices]
    return values, indices

def detect_outliers_modified_z(series: Union[pd.Series, List, np.ndarray], threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    arr = _to_float_array(series)
    mask = ~np.isnan(arr)
    clean = arr[mask]
    if clean.size == 0:
        return np.array([]), np.array([])
    median = np.median(clean)
    mad = np.median(np.abs(clean - median))
    if mad == 0:
        return np.array([]), np.array([])
    modified_z = 0.6745 * (clean - median) / mad
    out_mask_clean = np.abs(modified_z) > threshold
    clean_indices = np.where(out_mask_clean)[0]
    original_indices = np.where(mask)[0][clean_indices]
    return arr[original_indices], original_indices

def detect_outliers_zscore(series: Union[pd.Series, List, np.ndarray], threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    arr = _to_float_array(series)
    mask = ~np.isnan(arr)
    clean = arr[mask]
    if clean.size <= 1:
        return np.array([]), np.array([])
    mean = np.mean(clean)
    std = np.std(clean, ddof=1)
    if std == 0:
        return np.array([]), np.array([])
    z = (clean - mean) / std
    out_clean_idx = np.where(np.abs(z) > threshold)[0]
    original_idx = np.where(mask)[0][out_clean_idx]
    return arr[original_idx], original_idx

# ------------------ Mahalanobis with prechecks & shrinkage -----------------

def _numeric_subframe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return df[cols].select_dtypes(include=[np.number]).copy()

def _drop_constant_cols(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=const_cols), const_cols

def _has_enough_samples(clean_arr: np.ndarray) -> bool:
    n, p = clean_arr.shape
    return n >= p + 1

def detect_outliers_mahalanobis(df: pd.DataFrame,
                               cols: List[str],
                               chi2_alpha: float = 0.001,
                               cond_threshold: float = 1e12,
                               use_shrinkage_if_available: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns ['index', 'distance', 'threshold'] for multivariate outliers.
    Only runs if prechecks pass. Uses LedoitWolf shrinkage if available to mitigate singular covariances.
    """
    sub = _numeric_subframe(df, cols)
    sub, const_cols = _drop_constant_cols(sub)
    if const_cols:
        logger.info("Dropped constant columns for Mahalanobis: %s", const_cols)
    if sub.shape[1] < 2:
        logger.info("Need >=2 numeric variables after dropping constants; skipping Mahalanobis for these columns.")
        return pd.DataFrame(columns=['index', 'distance', 'threshold'])
    # drop rows with any NaN
    mask = ~sub.isna().any(axis=1)
    clean = sub.loc[mask].to_numpy(dtype=float)
    if clean.shape[0] == 0:
        logger.info("No complete rows to compute Mahalanobis; skipping.")
        return pd.DataFrame(columns=['index', 'distance', 'threshold'])
    if not _has_enough_samples(clean):
        logger.info("Not enough samples (%d) for %d features; skipping Mahalanobis.", clean.shape[0], clean.shape[1])
        return pd.DataFrame(columns=['index', 'distance', 'threshold'])
    # compute covariance safely
    mean = np.mean(clean, axis=0)
    cov = np.cov(clean.T)
    # condition number check
    cond = np.linalg.cond(cov)
    if cond > cond_threshold:
        logger.info("Covariance matrix ill-conditioned (cond=%.3e) > %.3e; attempting shrinkage.", cond, cond_threshold)
    cov_inv = None
    # try Ledoit-Wolf shrinkage if available and requested
    if use_shrinkage_if_available:
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(clean)
            cov = lw.covariance_
            cov_inv = np.linalg.inv(cov)
            logger.info("Using Ledoit-Wolf shrinkage covariance for Mahalanobis.")
        except Exception:
            cov_inv = None
    if cov_inv is None:
        try:
            cov_inv = inv(cov)
        except Exception:
            cov_inv = np.linalg.pinv(cov)
            logger.info("Using pseudo-inverse for covariance (singular or near-singular).")
    # compute distances for the clean rows
    distances = np.array([mahalanobis(row, mean, cov_inv) for row in clean])
    threshold = np.sqrt(stats.chi2.ppf(1 - chi2_alpha, cov.shape[0]))
    out_idx_clean = np.where(distances > threshold)[0]
    # map back to original indices in df
    original_indices = np.where(mask)[0][out_idx_clean]
    results = pd.DataFrame({
        'index': original_indices,
        'distance': distances[out_idx_clean],
        'threshold': threshold
    })
    return results

# ------------------ Runner that outputs combined CSV -----------------------

def analyze_and_save(csv_path: str,
                     output_folder: str,
                     variable_groups: dict,
                     methods: List[str] = ['iqr', 'modified_z', 'z_score', 'mahalanobis']):
    setup_logging(output_folder)
    try:
        data = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Failed to load %s: %s", csv_path, e)
        return

    all_records = []
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    for group, cols in variable_groups.items():
        available = [c for c in cols if c in data.columns]
        if not available:
            logger.info("Group %s: no available variables; skipping.", group)
            continue
        logger.info("Analyzing group: %s (%d variables)", group, len(available))

        # quick diagnostics
        numeric = _numeric_subframe(data, available)
        zero_var = [c for c in numeric.columns if numeric[c].nunique(dropna=True) <= 1]
        nan_rates = numeric.isna().mean().to_dict()
        logger.info("  zero_var=%s; sample NaN rates[example]=%s", zero_var, {k: round(nan_rates[k],3) for k in list(nan_rates)[:3]})

        if 'iqr' in methods:
            for col in available:
                vals, idx = detect_outliers_iqr(data[col])
                for v, i in zip(vals, idx):
                    all_records.append({
                        'detection_method': 'IQR',
                        'group': group,
                        'variable': col,
                        'sol_index': int(i),
                        'outlier_value': float(v),
                        'timestamp': run_ts
                    })
        if 'modified_z' in methods:
            for col in available:
                vals, idx = detect_outliers_modified_z(data[col])
                for v, i in zip(vals, idx):
                    all_records.append({
                        'detection_method': 'Modified_Z',
                        'group': group,
                        'variable': col,
                        'sol_index': int(i),
                        'outlier_value': float(v),
                        'timestamp': run_ts
                    })
        if 'z_score' in methods:
            for col in available:
                vals, idx = detect_outliers_zscore(data[col])
                for v, i in zip(vals, idx):
                    all_records.append({
                        'detection_method': 'Z_Score',
                        'group': group,
                        'variable': col,
                        'sol_index': int(i),
                        'outlier_value': float(v),
                        'timestamp': run_ts
                    })

        if 'mahalanobis' in methods:
            mahal_df = detect_outliers_mahalanobis(data, available)
            if not mahal_df.empty:
                for _, r in mahal_df.iterrows():
                    rec = {
                        'detection_method': 'Mahalanobis',
                        'group': group,
                        'variable': ','.join(available),
                        'sol_index': int(r['index']),
                        'outlier_value': None,
                        'distance': float(r['distance']),
                        'threshold': float(r['threshold']),
                        'timestamp': run_ts
                    }
                    all_records.append(rec)

    results = pd.DataFrame(all_records)
    if results.empty:
        logger.info("No outliers detected by selected methods.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        outpath = os.path.join(output_folder, f"outliers_combined_{run_ts}.csv")
        results.to_csv(outpath, index=False)
        logger.info("Saved combined outliers to: %s (rows=%d)", outpath, len(results))

# ------------------ Example usage & default variable groups ----------------

if __name__ == "__main__":
    DEFAULT_CSV = r"C:\Users\agsse\data_analysis\mars_mission_master_dataset_optimized.csv"
    OUTPUT_FOLDER = r"C:\Users\agsse\data_analysis\max_data"
    VARIABLE_GROUPS = {
        'temperature': ['temp_min','temp_max','temp_avg','temp_range','temp_p05','temp_median','temp_p95','avg_temp'],
        'wind_speed': ['wind_speed_avg','wind_speed_max','wind_speed_std','wind_speed_median','wind_speed_p95','wind_gust_factor'],
        'wind_direction': ['wind_direction_avg','wind_direction_std','wind_resultant_length','wind_u_mean','wind_v_mean','wind_vector_mean_speed','wind_vector_mean_dir'],
        'operational': ['activity_count','operational_status_rate','criticality_rate','power_status_rate','activity_intensity'],
        'environmental': ['solar_longitude_deg','dust_season_risk','orbital_thermal_factor','temp_volatility_7sol','temp_trend_7sol'],
        'system_health': ['equipment_age_factor','wear_accumulation','equipment_reliability_score','environmental_stress_index','operational_risk_index']
    }
    # Default: keep IQR and Modified Z as primary detectors, optionally run Z and Mahalanobis
    analyze_and_save(DEFAULT_CSV, OUTPUT_FOLDER, VARIABLE_GROUPS, methods=['iqr','modified_z','mahalanobis'])
