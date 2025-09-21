import os
import logging
from datetime import datetime
from typing import Optional, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def plot_grouped_boxplot(df: pd.DataFrame,
                         numeric_col: str,
                         category_col: str,
                         hue: Optional[str] = None,
                         order: Optional[List] = None,
                         output_folder: str = r'C:\Users\agsse\data_analysis\max_data',
                         annotate_pairs: bool = False) -> str:
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(10,6))
    ax = sns.boxplot(x=category_col, y=numeric_col, hue=hue, data=df, order=order)
    ax.set_title(f'Boxplot {numeric_col} by {category_col}')
    plt.tight_layout()
    outpath = os.path.join(output_folder, f'boxplot_{numeric_col}_by_{category_col}_{ts}.png')
    plt.savefig(outpath, dpi=150)
    plt.close()
    logger.info("Saved boxplot to %s", outpath)

    if annotate_pairs:
        cats = order if order else sorted(df[category_col].dropna().unique())
        pairs = list(itertools.combinations(cats, 2))
        stats_rows = []
        for a,b in pairs:
            a_vals = df.loc[df[category_col]==a, numeric_col].dropna()
            b_vals = df.loc[df[category_col]==b, numeric_col].dropna()
            if len(a_vals)>=3 and len(b_vals)>=3:
                try:
                    stat, p = stats.mannwhitneyu(a_vals, b_vals, alternative='two-sided')
                except Exception:
                    stat, p = (np.nan, np.nan)
            else:
                stat, p = (np.nan, np.nan)
            stats_rows.append({'group_a': a, 'group_b': b, 'stat': stat, 'pvalue': p})
        stats_df = pd.DataFrame(stats_rows)
        stats_path = os.path.join(output_folder, f'boxplot_pairwise_stats_{numeric_col}_by_{category_col}_{ts}.csv')
        stats_df.to_csv(stats_path, index=False)
        logger.info("Saved pairwise stats to %s", stats_path)
    return outpath