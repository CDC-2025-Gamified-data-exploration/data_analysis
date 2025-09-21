import os
import logging
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def make_contingency_table(df: pd.DataFrame,
                           row: str,
                           col: str,
                           normalize: bool = False,
                           output_folder: str = r'C:\Users\agsse\data_analysis\max_data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_folder, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    sub = df[[row, col]].dropna()
    ctab = pd.crosstab(sub[row], sub[col])
    ctab_prop = pd.crosstab(sub[row], sub[col], normalize='index' if normalize else None)
    
    ctab.to_csv(os.path.join(output_folder, f'contingency_counts_{row}_by_{col}_{ts}.csv'))
    ctab_prop.to_csv(os.path.join(output_folder, f'contingency_props_{row}_by_{col}_{ts}.csv'))
    logger.info("Saved contingency tables for %s x %s", row, col)
    return ctab, ctab_prop