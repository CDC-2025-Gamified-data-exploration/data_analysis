import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def plot_scatter(df: pd.DataFrame,
                 x: str,
                 y: str,
                 hue: Optional[str] = None,
                 add_regression: bool = True,
                 method_corr: str = 'pearson',
                 output_folder: str = r'C:\Users\agsse\data_analysis\max_data') -> str:
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    sub = df[[x,y,hue] if hue else [x,y]].dropna()
    plt.figure(figsize=(8,6))
    ax = sns.scatterplot(data=sub, x=x, y=y, hue=hue)
    if add_regression:
        try:
            sns.regplot(data=sub, x=x, y=y, scatter=False, ax=ax, line_kws={'color':'red'})
        except Exception:
            logger.debug("Could not add regression line")
    
    try:
        if method_corr == 'pearson':
            r, p = stats.pearsonr(sub[x], sub[y])
        elif method_corr == 'spearman':
            r, p = stats.spearmanr(sub[x], sub[y])
        elif method_corr == 'kendall':
            r, p = stats.kendalltau(sub[x], sub[y])
        else:
            r, p = (np.nan, np.nan)
    except Exception:
        r, p = (np.nan, np.nan)
    ax.text(0.02, 0.95, f'{method_corr}: r={r:.3f} p={p:.2g}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
    plt.title(f'{y} vs {x}')
    plt.tight_layout()
    outpath = os.path.join(output_folder, f'scatter_{x}_vs_{y}_{ts}.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    logger.info("Saved scatter plot to %s", outpath)
    return outpath