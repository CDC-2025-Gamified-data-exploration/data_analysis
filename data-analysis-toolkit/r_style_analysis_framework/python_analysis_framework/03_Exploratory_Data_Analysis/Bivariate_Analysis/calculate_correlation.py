import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _setup(output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    return ts

def _pairwise_pvals(series_df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    cols = series_df.columns
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j <= i:
                try:
                    x = series_df[a].dropna()
                    y = series_df[b].dropna()
                    xy = pd.concat([series_df[a], series_df[b]], axis=1).dropna()
                    if xy.shape[0] < 3:
                        p = np.nan
                    else:
                        if method == 'pearson':
                            _, p = stats.pearsonr(xy.iloc[:,0], xy.iloc[:,1])
                        elif method == 'spearman':
                            _, p = stats.spearmanr(xy.iloc[:,0], xy.iloc[:,1])
                        elif method == 'kendall':
                            _, p = stats.kendalltau(xy.iloc[:,0], xy.iloc[:,1])
                        else:
                            raise ValueError("Unsupported method")
                    pvals.loc[a,b] = p
                    pvals.loc[b,a] = p
                except Exception:
                    pvals.loc[a,b] = np.nan
                    pvals.loc[b,a] = np.nan
    return pvals

def calculate_correlations(df: pd.DataFrame,
                           cols: List[str] = None,
                           method: str = 'pearson',
                           dropna_pairwise: bool = True,
                           output_folder: str = r'C:\Users\agsse\data_analysis\max_data',
                           save_heatmap: bool = True,
                           heatmap_kwargs: dict = None) -> Dict[str, pd.DataFrame]:
    ts = _setup(output_folder)
    if cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        cols = numeric
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    corr = sub.corr(method=method)
    pvals = _pairwise_pvals(sub, method=method)
    
    corr_path = os.path.join(output_folder, f'correlation_{method}_{ts}.csv')
    pval_path = os.path.join(output_folder, f'correlation_pvalues_{method}_{ts}.csv')
    corr.to_csv(corr_path)
    pvals.to_csv(pval_path)
    logger.info("Saved correlation matrix to %s and p-values to %s", corr_path, pval_path)
    
    if save_heatmap:
        plt.figure(figsize=(max(8, 0.25*len(cols)), max(6, 0.2*len(cols))))
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        kwargs = dict(annot=True, fmt=".2f", cmap="vlag", center=0, mask=mask, linewidths=.5, cbar_kws={"shrink": .5})
        if heatmap_kwargs:
            kwargs.update(heatmap_kwargs)
        sns.heatmap(corr, **kwargs)
        plt.title(f'{method.title()} correlation (lower triangle) - {ts}')
        plt.tight_layout()
        heatmap_path = os.path.join(output_folder, f'correlation_heatmap_{method}_{ts}.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        logger.info("Saved heatmap to %s", heatmap_path)
    return {'correlation': corr, 'pvalues': pvals}