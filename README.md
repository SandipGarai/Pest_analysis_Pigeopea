# Pest Infestation Dynamics in Relation to Weather Parameters in Pigeonpea

**Study location:** Agricultural College, Warangal, Telangana, India (18.01°N, 79.60°E)  
**Study period:** Standard meteorological weeks 42–52 (October–December), 2020–2024  
**Forecast period:** Standard weeks 42–52, 2025 (out-of-sample)  
**Crop:** Pigeonpea (_Cajanus cajan_)  
**Pests:** _Maruca vitrata_ · _Helicoverpa armigera_ · Podfly (_Melanagromyza obtusa_)

---

## Repository Contents

```
├── pest_analysis_describe_and_forecast_v3.2.py   # Complete analysis pipeline
├── data_csv_final.csv                             # Historical pest + weather data (2020-2024)
├── weather_2025_weekly.csv                        # 2025 weather data for out-of-sample forecast
├── figures/                                       # Generated figures (fig1-fig21, PNG)
├── results/                                       # Generated tables (table1-table14, CSV)
└── README.md
```

---

## Background

This study investigates the weekly population dynamics of three economically important pigeonpea pests across five consecutive crop seasons. The analysis combines classical statistical tests, machine learning regression, time-series decomposition, and a multi-model forecasting framework to quantify weather–infestation relationships, build predictive models, and generate out-of-sample forecasts for 2025 using real observed weather data.

The ensemble forecast uses the **Canine Olfactory Optimization (COO)** algorithm (Garai et al., 2026) as a black-box optimizer to learn optimal combination weights from leave-one-year-out (LOYO) cross-validated predictions, rather than relying on naive R²-based weighting.

---

## Data Description

### `data_csv_final.csv`

Historical weekly observations, 2020–2024. One row per pest per week per year (165 rows per pest, 495 total).

| Column        | Description                                      |
| ------------- | ------------------------------------------------ |
| `Date_range`  | Date range of the standard week (e.g., 15-21oct) |
| `Std_Week_No` | Standard meteorological week number (42–52)      |
| `Tmax`        | Maximum temperature (°C)                         |
| `Tmin`        | Minimum temperature (°C)                         |
| `RHI`         | Morning relative humidity (%)                    |
| `RHII`        | Evening relative humidity (%)                    |
| `Rainfall_mm` | Weekly rainfall (mm)                             |
| `SSH`         | Bright sunshine hours (hrs/day)                  |
| `P1`–`P5`     | Pest count on each of 5 sample plants            |
| `Pest`        | Pest species (`Maruca`, `Helicoverpa`, `Podfly`) |
| `Year`        | Crop season year (2020–2024)                     |

### `weather_2025_weekly.csv`

Out-of-sample weather data for 2025 forecasting. Same weather columns as above; no pest count columns.

---

## Installation

```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels joblib coo-algorithm
```

Python 3.10 or later is recommended.

---

## Usage

```bash
python pest_analysis_describe_and_forecast_v3.2.py
```

All outputs are written automatically to `figures/` and `results/`.

**VS Code (Jupyter cell mode):** Each `# %%` block is an independent executable cell. Run sequentially using the Jupyter extension.

**Backend:** The script uses `matplotlib.use('TkAgg')` by default. Change to `'Agg'` on headless servers or in Jupyter notebooks.

---

## Outputs

### Figures (saved to `figures/`)

| Figure | File                               | Description                                                           |
| ------ | ---------------------------------- | --------------------------------------------------------------------- |
| 1      | `fig1_yearwise_bar.png`            | Year-wise total seasonal pest counts (2020–2024)                      |
| 2      | `fig2_stacked_bar.png`             | Stacked bar — species contribution to annual infestation load         |
| 3      | `fig3_weekly_lines.png`            | Weekly infestation by year with 5-year mean overlay                   |
| 4      | `fig4_boxplots.png`                | Box plots of weekly infestation distribution by year                  |
| 5      | `fig5_heatmap.png`                 | Heatmap of infestation intensity (year × standard week)               |
| 6      | `fig6_spearman_heatmap.png`        | Spearman rank correlation matrices (weather vs infestation)           |
| 7      | `fig7_scatter_phenology.png`       | Scatter plots (strongest weather correlate) and phenological patterns |
| 8      | `fig8_glm_coef.png`                | NB-GLM regression coefficients (red = p < 0.05)                       |
| 9      | `fig9_rf_importance.png`           | Random Forest feature importance (MDI, ranked)                        |
| 10     | `fig10_lasso_coef.png`             | LASSO standardised coefficients for retained predictors               |
| 11     | `fig11_rf_pred_obs.png`            | RF predicted vs observed — 20% hold-out test set                      |
| 12     | `fig12_stl_decomposition.png`      | STL decomposition (trend, seasonal, residual; period = 11 weeks)      |
| 13     | `fig13_weather_trends.png`         | Weekly weather parameter trends 2020–2024 with 5-year mean            |
| 14     | `fig14_prediagnostic.png`          | Pre-diagnostic bars (ADF, KPSS, Durbin-Watson, Ljung-Box)             |
| 15     | `fig15_acf_pacf.png`               | ACF and PACF plots (lags 0–20), informing SARIMAX order               |
| 16     | `fig16_sarimax_fit.png`            | SARIMAX(1,0,1)(1,0,1)₁₁ in-sample fit vs observed (2020–2024)         |
| 17     | `fig17_coo_convergence.png`        | COO convergence curves (LOYO ensemble R² per iteration)               |
| 18     | `fig18_coo_weights_comparison.png` | Naive vs COO ensemble weights and LOYO metric comparison              |
| 19     | `fig19_forecast2025.png`           | 2025 out-of-sample forecasts — all models vs 2024 observed            |
| 20     | `fig20_historical_forecast.png`    | Historical (2020–2024) overlaid with 2025 COO ensemble forecast       |
| 21     | `fig21_forecast_heatmap.png`       | Heatmap of 2025 forecasts across all models (weeks 42–52)             |

### Tables (saved to `results/`)

| Table | File                                  | Description                                                        |
| ----- | ------------------------------------- | ------------------------------------------------------------------ |
| 1     | `table1_descriptive_stats.csv`        | Descriptive statistics of weekly pest infestation (2020–2024)      |
| 2     | `table2_shapiro_wilk.csv`             | Shapiro-Wilk normality test results by year and pest               |
| 3     | `table3_kruskal_wallis.csv`           | Kruskal-Wallis H test results for year-wise comparisons            |
| 4     | `table4_dunn_posthoc.csv`             | Dunn post-hoc pairwise comparisons (Bonferroni corrected)          |
| 5     | `table5_spearman_correlation.csv`     | Spearman rank correlation coefficients (weather vs infestation)    |
| 6     | `table6_glm_fit_summary.csv`          | NB-GLM goodness-of-fit statistics (AIC, deviance, pseudo-R²)       |
| 7     | `table7_glm_coef_Maruca.csv`          | NB-GLM coefficients and IRR for _Maruca vitrata_                   |
| 8     | `table8_glm_coef_Helicoverpa.csv`     | NB-GLM coefficients and IRR for _Helicoverpa armigera_             |
| 9     | `table9_glm_coef_Podfly.csv`          | NB-GLM coefficients and IRR for Podfly (_M. obtusa_)               |
| 10    | `table10_ml_performance.csv`          | RF and LASSO performance (5-fold CV and 20% hold-out test)         |
| 11    | `table11_pre_diagnostic.csv`          | Pre-diagnostic tests for time-series forecasting suitability       |
| 12    | `table12_forecast_model_metrics.csv`  | Forecasting model performance (SARIMAX AIC/BIC/R², RF/LASSO CV R²) |
| 13    | `table13_ensemble_comparison_coo.csv` | LOYO ensemble comparison — naive vs COO-optimised (ΔR², ΔRMSE%)    |
| 14    | `table14_forecast_2025.csv`           | 2025 out-of-sample weekly forecasts — all models and ensembles     |

---

## Methods Summary

### Section A — Descriptive and Statistical Analysis (Figures 1–13, Tables 1–10)

| Method                                        | Purpose                                              | Reference                 |
| --------------------------------------------- | ---------------------------------------------------- | ------------------------- |
| Descriptive statistics (mean, SD, CV%, total) | Characterise infestation variability                 | —                         |
| Shapiro-Wilk test                             | Assess normality of weekly count distributions       | Shapiro & Wilk (1965)     |
| Kruskal-Wallis H test                         | Year-wise differences in median infestation          | Kruskal & Wallis (1952)   |
| Dunn post-hoc test (Bonferroni)               | Pairwise year comparisons                            | Dunn (1964)               |
| Spearman rank correlation                     | Weather–infestation monotonic associations           | Spearman (1904)           |
| Negative Binomial GLM                         | Overdispersed count modelling; incidence rate ratios | McCullagh & Nelder (1989) |
| Random Forest (B = 300, 5-fold CV)            | Non-parametric feature importance and prediction     | Breiman (2001)            |
| LASSO regression (log1p response)             | Sparse variable selection under L1 regularisation    | Tibshirani (1996)         |
| STL decomposition (period = 11 weeks)         | Trend, seasonal, and residual separation             | Cleveland et al. (1990)   |

### Section B — Forecasting (Figures 14–21, Tables 11–14)

| Method                  | Purpose                                            | Reference                 |
| ----------------------- | -------------------------------------------------- | ------------------------- |
| ADF test                | Test for unit root (non-stationarity)              | Dickey & Fuller (1979)    |
| KPSS test               | Complementary stationarity test                    | Kwiatkowski et al. (1992) |
| Durbin-Watson test      | First-order autocorrelation in OLS residuals       | Durbin & Watson (1950)    |
| Ljung-Box test          | Serial autocorrelation at lag 10                   | Ljung & Box (1978)        |
| ACF/PACF plots          | Identify SARIMAX model order                       | Box et al. (2015)         |
| SARIMAX(1,0,1)(1,0,1)₁₁ | Autocorrelation + weather covariate modelling      | Seabold & Perktold (2010) |
| Leave-One-Year-Out CV   | True out-of-sample validation of component models  | —                         |
| COO algorithm           | Black-box optimisation of ensemble weights         | Garai et al. (2026)       |
| Ensemble forecast       | Weighted combination of SARIMAX, NB-GLM, RF, LASSO | Timmermann (2006)         |

**COO configuration:** 4 packs × 15 candidates, max 150 iterations, RF surrogate (activation threshold R² ≥ 0.55), random seed 42.

---

## Key Findings

- **Podfly dominates:** 61.2% of total recorded individuals (1,192 / 1,947); peak in weeks 50–52 consistent with pod maturation phenology.
- **Temperature drives Podfly:** Strong negative correlations with Tmax (rₛ = −0.68) and Tmin (rₛ = −0.69); NB-GLM IRR = 0.552 per 1°C Tmax increase (p < 0.001), meaning each 1°C rise reduces expected Podfly infestation by 44.8%.
- **Phenology dominates predictions:** Standard week number is the top RF predictor (importance 0.92 for Podfly), confirming crop-stage timing as the primary infestation driver.
- **COO improves ensemble accuracy:** RMSE reduced by 7.54% for Podfly (12.733 → 11.772) with R² increasing from 0.7748 to 0.8075 vs naive weighting. COO discovers a small SARIMAX contribution for Helicoverpa (5.6%) that naive R²-based weighting would suppress.
- **2025 forecasts:** Peak Maruca and Helicoverpa during weeks 47–48; Podfly escalation through week 52 (COO ensemble: ~76.5 counts per 5 plants), projecting a below-2024 but above-2020 season.

---

## References

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time Series Analysis: Forecasting and Control_ (5th ed.). Wiley. https://doi.org/10.1002/9781118619193

Breiman, L. (2001). Random Forests. _Machine Learning_, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

Clark, J. S., et al. (2001). Ecological forecasts: An emerging imperative. _Science_, 293(5530), 657–660. https://doi.org/10.1126/science.293.5530.657

Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. _Journal of Official Statistics_, 6(1), 3–73.

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. _JASA_, 74(366), 427–431. https://doi.org/10.1080/01621459.1979.10482531

Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression. _Biometrika_, 37(3–4), 409–428. https://doi.org/10.1093/biomet/37.3-4.409

Garai, S., Kanaka, K. K., Manik, S., Naskar, S., & Bhadana, V. P. (2026). Canine Olfactory Optimization (COO): A surrogate-assisted metaheuristic algorithm. _SSRN Preprint_. https://doi.org/10.2139/ssrn.6278350

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. _Computing in Science & Engineering_, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Kannihalli, S., Rachappa, V., & Sahana, M. (2024). Influence of different sowing dates on seasonal incidence of pod fly in pigeonpea. _Int. J. Agriculture Extension and Social Development_, 7(8S), 163–165. https://doi.org/10.33545/26180723.2024.v7.i8Sc.956

Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. _JASA_, 47(260), 583–621. https://doi.org/10.1080/01621459.1952.10483441

Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. _Journal of Econometrics_, 54(1–3), 159–178. https://doi.org/10.1016/0304-4076(92)90104-Y

Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. _Biometrika_, 65(2), 297–303. https://doi.org/10.1093/biomet/65.2.297

McCullagh, P., & Nelder, J. A. (1989). _Generalized Linear Models_. Springer. https://doi.org/10.1007/978-1-4899-3242-6

Paul, R. K., Garai, S., et al. (2026). Novel CEEMDAN-based deep learning for rainfall prediction. _Journal of Hydrology_, 664, 134339. https://doi.org/10.1016/j.jhydrol.2025.134339

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. _JMLR_, 12, 2825–2830.

Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. _Proceedings SciPy 2010_, 92–96. https://doi.org/10.25080/majora-92bf1922-011

Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality. _Biometrika_, 52(3–4), 591–611. https://doi.org/10.1093/biomet/52.3-4.591

Singh, M. K., Dwivedi, S. K., & Yadav, H. S. (2024). Population dynamics of pod fly and natural enemies on pigeonpea. _ENTOMON_, 49(4), 519–526. https://doi.org/10.33307/entomon.v49i4.1341

Spearman, C. (1904). The proof and measurement of association between two things. _American Journal of Psychology_, 15(1), 72. https://doi.org/10.2307/1412159

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. _JRSS-B_, 58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x

Timmermann, A. (2006). Forecast combinations. In _Handbook of Economic Forecasting_ (Vol. 1, pp. 135–196). Elsevier. https://doi.org/10.1016/S1574-0706(05)01004-9

Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. _Nature Methods_, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2

Waskom, M. (2021). seaborn: Statistical data visualization. _JOSS_, 6(60), 3021. https://doi.org/10.21105/joss.03021

---

## Citation

If you use this code or data, please cite:

> Garai, S. & Veeranna, D. (2026). _Pest infestation dynamics in relation to weather parameters in pigeonpea_ [Analysis pipeline]. GitHub. https://github.com/SandipGarai/Pest_analysis_Pigeopea

---

## License

MIT License
