# AQI Weather Prediction — Beijing Air Quality Analysis

A Jupyter/Colab notebook that cleans, analyzes, and models hourly air quality and weather data from Beijing, then forecasts the Air Quality Index (AQI) with LSTM and GRU neural networks — built at the American University of Phnom Penh.
Group members: Chhin Sokhom Sidhisakdeh, Sam Chankroesna

**Notebook:** `AQI_Weather_prediction.ipynb`
**Repo:** `https://github.com/Kroesna-S/AQI_Weather_Prediction`

## Tech stack

- Python 3 (Google Colab)
- pandas + NumPy (data cleaning and feature engineering)
- Matplotlib + Seaborn (visualization)
- SciPy + statsmodels (hypothesis tests, regression, time series)
- scikit-learn (scaling, encoding, metrics)
- TensorFlow / Keras (LSTM and GRU models)

## Dataset

[Beijing Multi-Site Air-Quality Data](https://www.kaggle.com/datasets/aravindpcoder/beijing-multi-site-air-quality-data) on Kaggle: hourly pollutant and weather readings from March 2013 to February 2017.

| Property          | Value                                                    |
|-------------------|----------------------------------------------------------|
| Records used      | 264,202                                                  |
| Stations loaded   | 9                                                        |
| Pollutants        | PM2.5, PM10, SO2, NO2, CO, O3                            |
| Weather variables | TEMP, PRES, DEWP, RAIN, WSPM, wind direction (`wd`)      |

The dataset isn't included in this repo. Download it from the link above and keep the CSV named `Beijing Multisite air Quality data.csv`.

## Getting started

**Google Colab (recommended)**

1. Open `AQI_Weather_prediction.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run the setup cell to install and import all libraries.
3. Upload the dataset CSV when the upload prompt appears.
4. Run the remaining cells top to bottom.

**Locally**

```bash
pip install scikit-learn tensorflow pandas numpy matplotlib seaborn scipy statsmodels jupyter
jupyter notebook AQI_Weather_prediction.ipynb
```

The notebook uses `google.colab.files` for uploads and downloads. Locally, replace the upload cell with a direct `pd.read_csv(...)` path and remove the `files.download(...)` calls.

## How the notebook works

**Part 1 — Data collection & cleaning** — builds a `datetime` column, label-encodes stations, maps the 16 wind directions to numbers, and fills the 46,751 missing values with forward/backward fill per station. It then computes AQI from PM2.5 using US EPA breakpoints and assigns one of six health categories (Good through Hazardous), plus season and hour-of-day features. The cleaned data is saved as `Beijing_AQI_Cleaned.csv`.

**Part 2 — Descriptive statistics** — mean, median, mode, standard deviation, skewness, and kurtosis for every numeric feature, plus AQI averages by station, season, and year.

**Part 3 — Data visualization** — average AQI by station, monthly trend, seasonal bars, category pie chart, year-by-month heatmap, per-station boxplots, hourly pattern, and pollutant trends over time.

**Part 4 — Exploratory data analysis** — feature correlation matrix and ranked correlations with AQI.

**Part 5 — Hypothesis testing** — Pearson correlation, multiple linear regression (with 95% confidence intervals and assumption checks), Chi-Square (AQI category vs. season, with Cramér's V), and one-way ANOVA (AQI across seasons, with η²).

**Part 6 — Time series analysis** — additive decomposition of daily AQI, Augmented Dickey-Fuller stationarity test, ACF/PACF plots, and 7-day / 30-day rolling averages.

**Part 7 — LSTM & GRU forecasting** — 24-hour look-back sequences over 12 input features, an 80/20 train/test split, and evaluation with MAE, RMSE, R², and AQI-category accuracy.

**Part 8 — Findings & recommendations** — final model comparison table and written conclusions.

## Models

Both networks share the same structure so the comparison is fair.

| Setting          | Value                                                         |
|------------------|---------------------------------------------------------------|
| Architecture     | 2 recurrent layers (128 → 64), Dropout(0.2), Dense(32), Dense(1) |
| Look-back window | 24 hours                                                      |
| Optimizer / loss | Adam / MSE                                                    |
| Batch size       | 128                                                           |
| Epochs           | Up to 50, early stopping (patience 5)                         |
| Scaling          | MinMaxScaler                                                  |
| Split            | 211,342 train / 52,836 test (10% of train held out for validation) |

## Results

| Model | MAE   | RMSE  | R²     | Category accuracy | Parameters |
|-------|-------|-------|--------|-------------------|------------|
| LSTM  | 13.86 | 21.44 | 0.9383 | 77.93%            | 124,225    |
| GRU   | 12.98 | 20.33 | 0.9446 | 78.65%            | 94,273     |

GRU scored better on every metric while using about 24% fewer parameters.

Statistical highlights:

| Test                 | Result                                                              |
|----------------------|---------------------------------------------------------------------|
| Pearson correlation  | PM2.5 r = 0.972, PM10 0.859, CO 0.745, NO2 0.676, SO2 0.474; wind speed r = −0.309 |
| OLS regression       | R² = 0.955 on a 50,000-row sample; F-test significant               |
| Chi-Square           | AQI category depends on season (χ² = 11,958, Cramér's V = 0.123, small effect) |
| One-way ANOVA        | Seasonal means differ (F = 709.3) but η² = 0.008, a negligible effect |
| ADF test             | Daily AQI is stationary (statistic −18.26, p < 0.001)               |

Mean AQI by season: Winter 154.65, Autumn 147.80, Spring 147.20, Summer 133.00.

## Key findings

1. **Winter is the worst season** — highest mean AQI and the most Hazardous readings, consistent with heating emissions and weaker dispersion.
2. **Pollutant concentrations drive AQI** — PM2.5 and PM10 dominate, followed by CO and NO2. Of the weather variables, wind speed matters most.
3. **AQI is strongly time-dependent** — stationarity and autocorrelation results support sequence models over plain regression.
4. **GRU is the better and leaner model here** — higher R² and category accuracy with fewer parameters.
5. **Air quality is generally poor** — roughly a quarter of readings are Very Unhealthy or Hazardous, and only about 10% are Good.

Recommendations: focus winter public-health alerts on sensitive groups, target PM2.5 sources (vehicles, coal burning), use GRU for real-time deployment, test a 48-hour look-back for extreme pollution events, and integrate a live weather API for real-time forecasting.

## Outputs

Running the notebook generates:

```
Beijing_AQI_Cleaned.csv             cleaned dataset with AQI, categories, and time features
viz_*.png                           overview, heatmap, station boxplot, hourly/distribution, pollutant trends
eda_correlation_matrix.png
hypothesis_chi2.png
hypothesis_regression_assumptions.png
ts_decomposition.png
ts_acf_pacf.png
ts_rolling_avg.png
ml_*.png                            feature correlation, training curves, predicted vs. actual, scatter, error distribution
lstm_aqi_final.h5
gru_aqi_final.h5
```

## Known limitations

- **Target leakage** — AQI is calculated directly from PM2.5, and PM2.5 is also an input feature. This inflates R² for both the regression and the neural networks, so the models are closer to reconstructing AQI than forecasting it from weather alone. A stricter setup would drop PM2.5 (and possibly PM10) from the inputs or predict future AQI from past values only.
- **Imputation** — forward/backward filling can smooth over long sensor outages.
- **Sequence construction** — windows are built over the concatenated station data, so a few windows near station boundaries mix records from different stations.
- **Small seasonal effect** — season is statistically significant but explains under 1% of AQI variance.
- **Scope** — results reflect Beijing from 2013 to 2017 and may not generalize to other cities or years.
- **Legacy model format** — models are saved as `.h5`, which Keras now considers legacy; use `.keras` for new work.
