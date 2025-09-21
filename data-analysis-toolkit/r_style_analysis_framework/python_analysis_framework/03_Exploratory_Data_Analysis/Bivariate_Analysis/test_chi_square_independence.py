import os
import logging
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def cramers_v(confusion_matrix: np.ndarray) -> float:
    chi2, p, dof, expected = stats.chi2_contingency(confusion_matrix, correction=False)
    n = confusion_matrix.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    denom = min((kcorr-1), (rcorr-1))
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)

def test_independence(df: pd.DataFrame,
                      row: str,
                      col: str,
                      fisher_threshold: int = 5,
                      output_folder: str = r'C:\Users\agsse\data_analysis\max_data') -> Dict:
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub = df[[row, col]].dropna()
    table = pd.crosstab(sub[row], sub[col])
    
    try:
        if table.size == 0:
            raise ValueError("Empty table")
        if table.shape == (2,2) and (table.values < fisher_threshold).sum() > 0:
            from scipy.stats import fisher_exact
            oddsratio, pvalue = fisher_exact(table.values)
            method = 'fisher_exact'
            chi2 = np.nan; dof = np.nan; expected = np.nan
        else:
            chi2, pvalue, dof, expected = stats.chi2_contingency(table, correction=False)
            method = 'chi2'
    except Exception as e:
        logger.exception("Error testing independence: %s", e)
        raise
    
    try:
        v = cramers_v(table.values)
    except Exception:
        v = np.nan
    
    out = {
        'row': row, 'col': col, 'method': method,
        'chi2': float(chi2) if 'chi2' in locals() and not pd.isna(chi2) else np.nan,
        'pvalue': float(pvalue) if pvalue is not None else np.nan,
        'dof': int(dof) if 'dof' in locals() and not pd.isna(dof) else np.nan,
        'cramers_v': float(v),
        'n_obs': int(table.values.sum())
    }
    
    table.to_csv(os.path.join(output_folder, f'chi_contingency_table_{row}_by_{col}_{ts}.csv'))
    try:
        pd.DataFrame(expected, index=table.index, columns=table.columns).to_csv(
            os.path.join(output_folder, f'chi_expected_{row}_by_{col}_{ts}.csv'))
    except Exception:
        pass
    logger.info("Chi/ Fisher test saved for %s x %s (method=%s p=%s)", row, col, method, out['pvalue'])
    return out