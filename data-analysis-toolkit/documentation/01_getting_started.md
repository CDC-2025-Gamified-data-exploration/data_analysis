# Getting Started with the Data Analysis Toolkit

This guide provides instructions on how to set up and use the Data Analysis Toolkit.

## 1. Installation

First, clone the repository to your local machine:
```bash
git clone <repository_url>
cd data-analysis-toolkit
```

Next, it is highly recommended to create a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Finally, install the required Python packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*Note: Some functions may require additional libraries not listed in `requirements.txt` (e.g., `pygam`, `pmdarima`). These should be installed as needed.*

## 2. Repository Structure

The repository is organized into several key directories:
- **/data**: Store all your data here. Raw data goes in `raw/`, and cleaned datasets will be saved to `cleaned/`.
- **/functions**: This is the core library of the toolkit, containing all the Python modules for analysis.
- **/scripts**: This directory contains example scripts and analysis pipelines.
- **/outputs**: All generated files, such as plots and reports, will be saved here.
- **/documentation**: You are here! This directory contains all project documentation.

## 3. Running an Analysis

To run an analysis, you can either use the pre-built pipelines in `scripts/pipelines` or write your own custom script in the `scripts` directory.

An example of a custom script might look like this:
```python
# in scripts/my_analysis.py
import pandas as pd
from functions import data_preparation, univariate_analysis

# Load data
df = pd.read_csv('../data/raw/my_data.csv')

# Clean data
df_cleaned = data_preparation.impute_mean(df, 'my_column')

# Perform analysis
univariate_analysis.plot_histogram(df_cleaned, 'my_column')
```

To run this script, execute it from the `scripts` directory:
```bash
cd scripts
python my_analysis.py
```
