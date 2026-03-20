import os
import sys
import pickle
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scipy import stats
from scipy.stats import kruskal, shapiro, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

try:
    from coo import COO
    HAS_COO = True
except ImportError:
    HAS_COO = False

warnings.filterwarnings('ignore')

# %% ── SETUP ─────────────────────────────────────────────────────────────────
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

COLORS = {'Maruca': '#E74C3C', 'Helicoverpa': '#2980B9', 'Podfly': '#27AE60'}
YEAR_COLORS = {2020: '#1A237E', 2021: '#1565C0', 2022: '#0097A7',
               2023: '#2E7D32', 2024: '#F57F17'}
FC_COLORS = {'SARIMAX': '#8E44AD', 'NB_GLM': '#E67E22',
             'RandomForest': '#16A085', 'LASSO': '#C0392B',
             'Naive': '#7F8C8D', 'COO': '#2C3E50'}

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
    'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'axes.labelweight': 'bold',
})

PESTS    = ['Maruca', 'Helicoverpa', 'Podfly']
WEATHER  = ['Tmax', 'Tmin', 'RHI', 'RHII', 'Rainfall_mm', 'SSH']
FEATURES = WEATHER + ['Std_Week_No']

# %% ── 1. LOAD & PREPROCESS ──────────────────────────────────────────────────
df = pd.read_csv('data_csv_final.csv')
df.columns = df.columns.str.strip()
df['Year']      = df['Year'].astype(int)
df['Total']     = df[['P1', 'P2', 'P3', 'P4', 'P5']].sum(axis=1)
df['Mean_inf']  = df['Total'] / 5
df['Tmean']     = (df['Tmax'] + df['Tmin']) / 2
df['RHmean']    = (df['RHI']  + df['RHII']) / 2
df['Log_Total'] = np.log1p(df['Total'])

df25 = pd.read_csv('weather_2025_weekly.csv')
df25.columns = df25.columns.str.strip()
df25['Tmean']  = (df25['Tmax'] + df25['Tmin']) / 2
df25['RHmean'] = (df25['RHI']  + df25['RHII']) / 2

print('Dataset shape:', df.shape)
print('\nYear x Pest seasonal totals:')
print(df.groupby(['Year', 'Pest'])['Total'].sum().unstack())

# %% ── 2. TABLE 1: DESCRIPTIVE STATISTICS ───────────────────────────────────
desc_rows = []
for pest in PESTS:
    sub = df[df['Pest'] == pest]
    for yr in sorted(sub['Year'].unique()):
        v = sub[sub['Year'] == yr]['Total'].values
        desc_rows.append({
            'Pest': pest, 'Year': yr,
            'Mean': round(np.mean(v), 2),
            'SD':   round(np.std(v, ddof=1), 2),
            'Min':  int(np.min(v)), 'Max': int(np.max(v)),
            'Median': round(np.median(v), 1),
            'CV%': round(np.std(v, ddof=1) / np.mean(v) * 100, 1) if np.mean(v) > 0 else 0,
            'Total': int(np.sum(v)),
        })
df_desc = pd.DataFrame(desc_rows)
df_desc.to_csv('results/table1_descriptive_stats.csv', index=False)
print('\nTable 1 saved: Descriptive statistics of weekly pest infestation (2020-2024)')

# %% ── 3. FIGURE 1: YEAR-WISE GROUPED BAR ───────────────────────────────────
ydf = df.groupby(['Year', 'Pest'])['Total'].sum().reset_index()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
print('Figure 1: Year-wise total seasonal pest counts (2020-2024)')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    sub  = ydf[ydf['Pest'] == pest]
    bars = ax.bar(sub['Year'].astype(str), sub['Total'],
                  color=[YEAR_COLORS[y] for y in sub['Year']],
                  edgecolor='white', linewidth=1.2, width=0.6)
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    for bar, val in zip(bars, sub['Total']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(val)), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylim(0, sub['Total'].max() * 1.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Total Pest Count', fontweight='bold') if idx == 0 else ax.set_ylabel('')
fig.text(0.5, -0.02, 'Year', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig1_yearwise_bar.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 4. FIGURE 2: STACKED BAR ─────────────────────────────────────────────
ydf2 = df.groupby(['Year', 'Pest'])['Total'].sum().unstack(fill_value=0)[PESTS]
fig, ax = plt.subplots(figsize=(10, 5))
print('Figure 2: Stacked bar showing species contribution to annual infestation load (2020-2024)')
bottoms = np.zeros(len(ydf2))
x = np.arange(len(ydf2))
for pest in PESTS:
    vals = ydf2[pest].values
    ax.bar(x, vals, bottom=bottoms, color=COLORS[pest], edgecolor='white',
           linewidth=0.8, width=0.6, label=pest)
    for xi, (val, bot) in enumerate(zip(vals, bottoms)):
        if val > 5:
            ax.text(xi, bot + val / 2, str(int(val)),
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    bottoms += vals
for xi, total in enumerate(bottoms):
    ax.text(xi, total + 2, str(int(total)),
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')
ax.set_xticks(x)
ax.set_xticklabels([str(y) for y in ydf2.index])
ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Total Count', fontweight='bold')
ax.set_ylim(0, bottoms.max() * 1.12)
ax.legend(title='Pest', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig('figures/fig2_stacked_bar.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 5. FIGURE 3: WEEKLY PHENOLOGY LINES ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
legend_handles, legend_labels = [], []
print('Figure 3: Weekly infestation by year with 5-year mean overlay')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    for yr, grp in df[df['Pest'] == pest].groupby('Year'):
        wk    = grp.groupby('Std_Week_No')['Total'].sum()
        line, = ax.plot(wk.index, wk.values, marker='o', markersize=5,
                        color=YEAR_COLORS[yr], linewidth=2, label=str(yr))
        if idx == 0:
            legend_handles.append(line); legend_labels.append(str(yr))
    mean_wk = df[df['Pest'] == pest].groupby('Std_Week_No')['Total'].mean()
    ml, = ax.plot(mean_wk.index, mean_wk.values, 'k--', linewidth=2.5, label='Mean', zorder=5)
    if idx == 0:
        legend_handles.append(ml); legend_labels.append('Mean')
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.set_xticks(range(42, 53))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Total Count', fontweight='bold') if idx == 0 else ax.set_ylabel('')
fig.text(0.45, -0.02, 'Standard Week', ha='center', fontsize=11, fontweight='bold')
fig.legend(legend_handles, legend_labels, title='Year', loc='center right',
           bbox_to_anchor=(1.0, 0.5), fontsize=9, title_fontsize=10, framealpha=0.9)
plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig('figures/fig3_weekly_lines.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 6. FIGURE 4: BOX PLOTS ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
print('Figure 4: Box plots of weekly infestation distribution by year')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    sub          = df[df['Pest'] == pest]
    data_by_year = [sub[sub['Year'] == yr]['Total'].values for yr in sorted(sub['Year'].unique())]
    bp           = ax.boxplot(data_by_year, patch_artist=True)
    for patch, yr in zip(bp['boxes'], sorted(sub['Year'].unique())):
        patch.set_facecolor(YEAR_COLORS[yr]); patch.set_alpha(0.8)
    for line in bp['medians']:
        line.set_color('black'); line.set_linewidth(2)
    ax.set_xticklabels([str(y) for y in sorted(sub['Year'].unique())])
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Weekly Count', fontweight='bold') if idx == 0 else ax.set_ylabel('')
fig.text(0.5, -0.02, 'Year', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig4_boxplots.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 7. FIGURE 5: HEATMAP ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
print('Figure 5: Heatmap of total infestation intensity by year and standard week')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    sub   = df[df['Pest'] == pest]
    pivot = sub.pivot_table(index='Year', columns='Std_Week_No', values='Total', aggfunc='sum')
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='g', linewidths=0.5,
                cbar=(idx == 2), cbar_kws={'label': 'Count'} if idx == 2 else {},
                annot_kws={'size': 9})
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.set_ylabel('Year', fontweight='bold') if idx == 0 else ax.set_ylabel('')
fig.text(0.5, -0.02, 'Standard Week', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig5_heatmap.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 8. TABLES 2-4: SHAPIRO-WILK + KRUSKAL-WALLIS + DUNN ─────────────────
shapiro_rows, kw_rows, dunn_rows = [], [], []
for pest in PESTS:
    sub    = df[df['Pest'] == pest]
    groups = [sub[sub['Year'] == yr]['Total'].values for yr in sorted(sub['Year'].unique())]
    for yr, v in zip(sorted(sub['Year'].unique()), groups):
        W, p = shapiro(v)
        shapiro_rows.append({'Pest': pest, 'Year': yr, 'W': round(W, 4),
                             'p': round(p, 4), 'Normal': 'Yes' if p > 0.05 else 'No'})
    H, p = kruskal(*groups)
    kw_rows.append({'Pest': pest, 'H': round(H, 4), 'p': round(p, 4),
                    'Sig': 'Yes' if p < 0.05 else 'No'})
    all_v  = np.concatenate(groups)
    all_y  = np.concatenate([[yr] * len(g) for yr, g in zip(sorted(sub['Year'].unique()), groups)])
    N, rnks = len(all_v), stats.rankdata(all_v)
    for y1, y2 in combinations(sorted(sub['Year'].unique()), 2):
        r1, r2 = rnks[all_y == y1], rnks[all_y == y2]
        z      = abs(np.mean(r1) - np.mean(r2)) / np.sqrt((N * (N + 1) / 12) * (1 / len(r1) + 1 / len(r2)))
        p_raw  = 2 * (1 - stats.norm.cdf(z))
        dunn_rows.append({'Pest': pest, 'Y1': y1, 'Y2': y2, 'Z': round(z, 4),
                          'p_raw': round(p_raw, 4), 'p_bonf': round(min(p_raw * 10, 1), 4)})

pd.DataFrame(shapiro_rows).to_csv('results/table2_shapiro_wilk.csv', index=False)
pd.DataFrame(kw_rows).to_csv('results/table3_kruskal_wallis.csv', index=False)
pd.DataFrame(dunn_rows).to_csv('results/table4_dunn_posthoc.csv', index=False)
print('Table 2 saved: Shapiro-Wilk normality test results by year and pest')
print('Table 3 saved: Kruskal-Wallis H test results for year-wise comparisons')
print('Table 4 saved: Dunn post-hoc pairwise comparisons (Bonferroni corrected)')

# %% ── 9. TABLE 5 + FIGURES 6-7: SPEARMAN CORRELATION ───────────────────────
sp_rows = []
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
print('Figure 6: Spearman rank correlation matrices (weather vs infestation)')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    sub      = df[df['Pest'] == pest]
    corr_mat = sub[WEATHER + ['Total']].corr(method='spearman')
    sns.heatmap(corr_mat, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                annot_kws={'size': 9}, cbar=(idx == 2),
                cbar_kws={'shrink': 0.8} if idx == 2 else {})
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    for w in WEATHER:
        rs, p = spearmanr(sub[w], sub['Total'])
        sig   = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        sp_rows.append({'Pest': pest, 'Variable': w, 'rs': round(rs, 4),
                        'p': round(p, 4), 'Sig': sig})
df_sp = pd.DataFrame(sp_rows)
plt.tight_layout()
plt.savefig('figures/fig6_spearman_heatmap.png', dpi=180, bbox_inches='tight')
plt.close()
df_sp.to_csv('results/table5_spearman_correlation.csv', index=False)
print('Table 5 saved: Spearman rank correlation coefficients (weather vs infestation)')

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
tlh, tll  = [], []
print('Figure 7: Scatter plots (top-ranked weather variable) and phenological patterns')
for col, pest in enumerate(PESTS):
    sub     = df[df['Pest'] == pest]
    top_var = df_sp[df_sp['Pest'] == pest].iloc[
        df_sp[df_sp['Pest'] == pest]['rs'].abs().argsort().values[-1]]['Variable']
    rs_v    = df_sp[(df_sp['Pest'] == pest) & (df_sp['Variable'] == top_var)]['rs'].values[0]
    p_v     = df_sp[(df_sp['Pest'] == pest) & (df_sp['Variable'] == top_var)]['p'].values[0]
    ax      = axes[0, col]
    for yr in sorted(sub['Year'].unique()):
        s  = sub[sub['Year'] == yr]
        sc = ax.scatter(s[top_var], s['Total'], c=YEAR_COLORS[yr], alpha=0.8, s=60, edgecolors='white')
        if col == 0: tlh.append(sc); tll.append(str(yr))
    z  = np.polyfit(sub[top_var], sub['Total'], 1)
    xr = np.linspace(sub[top_var].min(), sub[top_var].max(), 100)
    ax.plot(xr, np.poly1d(z)(xr), 'k--', linewidth=2)
    ax.set_xlabel(top_var, fontweight='bold')
    ax.set_title(f'{pest}\nrs={rs_v:.2f}, p={p_v:.3f}', fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Total Infestation', fontweight='bold') if col == 0 else ax.set_ylabel('')
    ax2 = axes[1, col]
    for yr in sorted(sub['Year'].unique()):
        s = sub[sub['Year'] == yr]
        ax2.scatter(s['Std_Week_No'], s['Total'], c=YEAR_COLORS[yr], alpha=0.7, s=60, edgecolors='white')
    mwk = sub.groupby('Std_Week_No')['Total'].mean()
    ax2.plot(mwk.index, mwk.values, color=COLORS[pest], linewidth=2.5, marker='D', markersize=6)
    ax2.set_title(f'{pest} - Phenological Pattern', fontweight='bold', color=COLORS[pest])
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.set_ylabel('Total Infestation', fontweight='bold') if col == 0 else ax2.set_ylabel('')
fig.text(0.5, 0.02, 'Standard Week', ha='center', fontsize=11, fontweight='bold')
fig.legend(tlh, tll, title='Year', loc='upper right', bbox_to_anchor=(1.0, 0.97),
           fontsize=9, title_fontsize=10, framealpha=0.9)
plt.tight_layout(rect=[0, 0.04, 0.93, 1])
plt.savefig('figures/fig7_scatter_phenology.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 10. TABLES 6-9 + FIGURE 8: NEGATIVE BINOMIAL GLM ────────────────────
glm_fit, glm_coef = [], {}
for pest in PESTS:
    sub  = df[df['Pest'] == pest].copy()
    X, y = sm.add_constant(sub[WEATHER]), sub['Total']
    od   = sm.GLM(y, X, family=sm.families.Poisson()).fit().deviance / \
           sm.GLM(y, X, family=sm.families.Poisson()).fit().df_resid
    nb   = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(disp=False)
    rows = []
    for vname in X.columns:
        c, se, z, p = nb.params[vname], nb.bse[vname], nb.tvalues[vname], nb.pvalues[vname]
        rows.append({'Variable': vname, 'Coef': round(c, 4), 'IRR': round(np.exp(c), 4),
                     'SE': round(se, 4), 'z': round(z, 4), 'p': round(p, 4),
                     'CI95_lo': round(np.exp(c - 1.96 * se), 4),
                     'CI95_hi': round(np.exp(c + 1.96 * se), 4),
                     'Sig': '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'})
    glm_coef[pest] = pd.DataFrame(rows)
    pest_tbl = PESTS.index(pest) + 7
    glm_coef[pest].to_csv(f'results/table{pest_tbl}_glm_coef_{pest}.csv', index=False)
    print(f'Table {pest_tbl} saved: NB-GLM coefficients and IRR for {pest}')
    glm_fit.append({'Pest': pest, 'AIC': round(nb.aic, 2), 'Deviance': round(nb.deviance, 2),
                    'df_resid': int(nb.df_resid), 'Poisson_OD': round(od, 3),
                    'Pseudo_R2': round(1 - nb.deviance / nb.null_deviance, 4)})
pd.DataFrame(glm_fit).to_csv('results/table6_glm_fit_summary.csv', index=False)
print('Table 6 saved: NB-GLM goodness-of-fit statistics (AIC, deviance, pseudo-R2)')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
print('Figure 8: NB-GLM regression coefficients (red = p<0.05, grey = not significant)')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    ct  = glm_coef[pest][glm_coef[pest]['Variable'] != 'const'].copy()
    col = ['#C0392B' if p < 0.05 else '#BDC3C7' for p in ct['p']]
    ax.barh(ct['Variable'], ct['Coef'], color=col, edgecolor='white', height=0.6)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_title(f'{pest} - NB GLM', fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.02, 'Coefficient', ha='center', fontsize=11, fontweight='bold')
legend_patches = [mpatches.Patch(color='#C0392B', label='p < 0.05'),
                  mpatches.Patch(color='#BDC3C7', label='p >= 0.05')]
fig.legend(handles=legend_patches, title='Significance', loc='center right',
           bbox_to_anchor=(1.0, 0.5), fontsize=9, title_fontsize=10, framealpha=0.9)
plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.savefig('figures/fig8_glm_coef.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 11. TABLE 10 + FIGURES 9-11: RANDOM FOREST & LASSO ──────────────────
ml_rows, rf_imp, lasso_coef_d = [], {}, {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for pest in PESTS:
    sub     = df[df['Pest'] == pest]
    X, y    = sub[FEATURES].values, sub['Total'].values
    X_sc    = StandardScaler().fit_transform(X)
    rf      = RandomForestRegressor(n_estimators=300, max_depth=10,
                                    min_samples_leaf=2, random_state=42)
    r2_cv   = cross_val_score(rf, X, y, cv=kf, scoring='r2')
    rmse_cv = np.sqrt(-cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_squared_error'))
    rf.fit(X, y)
    imp              = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    rf_imp[pest]     = imp
    lasso            = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_sc, np.log1p(y))
    coef_s           = pd.Series(lasso.coef_, index=FEATURES)
    lasso_coef_d[pest] = coef_s
    lasso_r2         = cross_val_score(lasso, X_sc, np.log1p(y), cv=kf, scoring='r2')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(Xtr, ytr); ypred = rf.predict(Xte)
    ml_rows.append({
        'Pest': pest,
        'RF_CV_R2':    f'{r2_cv.mean():.3f}+/-{r2_cv.std():.3f}',
        'RF_RMSE_CV':  round(rmse_cv.mean(), 3),
        'RF_TestR2':   round(r2_score(yte, ypred), 3),
        'RF_TestRMSE': round(np.sqrt(mean_squared_error(yte, ypred)), 3),
        'LASSO_alpha': round(lasso.alpha_, 5),
        'LASSO_CVR2':  round(lasso_r2.mean(), 3),
        'TopFeature':  imp.index[0],
    })
pd.DataFrame(ml_rows).to_csv('results/table10_ml_performance.csv', index=False)
print('Table 10 saved: RF and LASSO performance metrics (5-fold CV and 20% hold-out test)')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
print('Figure 9: Random Forest feature importance (MDI, ranked highest to lowest)')
for ax, pest in zip(axes, PESTS):
    imp  = rf_imp[pest]
    cols = ['#E74C3C', '#2980B9', '#27AE60'] + ['#95A5A6'] * (len(imp) - 3)
    ax.barh(imp.index[::-1], imp.values[::-1], color=cols[::-1], edgecolor='white', height=0.6)
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    for i, (n, v) in enumerate(zip(imp.index[::-1], imp.values[::-1])):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
fig.text(0.5, -0.02, 'Feature Importance (MDI)', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig9_rf_importance.png', dpi=180, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
print('Figure 10: LASSO standardised regression coefficients for retained predictors')
for ax, pest in zip(axes, PESTS):
    nz = lasso_coef_d[pest][lasso_coef_d[pest] != 0].sort_values()
    ax.barh(nz.index, nz.values,
            color=['#C0392B' if v > 0 else '#2980B9' for v in nz],
            edgecolor='white', height=0.5)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, -0.02, 'Standardised LASSO Coefficient', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig10_lasso_coef.png', dpi=180, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
print('Figure 11: RF predicted vs observed (20% hold-out test set, 1:1 reference line)')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    sub     = df[df['Pest'] == pest]
    X, y    = sub[FEATURES].values, sub['Total'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf2     = RandomForestRegressor(n_estimators=300, max_depth=10,
                                    min_samples_leaf=2, random_state=42)
    rf2.fit(Xtr, ytr); ypred = rf2.predict(Xte)
    ax.scatter(yte, ypred, color=COLORS[pest], alpha=0.8, s=70, edgecolors='white')
    lim = max(max(yte), max(ypred)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.5)
    ax.set_title(f'{pest}\nR2={r2_score(yte, ypred):.3f}, RMSE={np.sqrt(mean_squared_error(yte, ypred)):.2f}',
                 fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Predicted', fontweight='bold') if idx == 0 else ax.set_ylabel('')
fig.text(0.5, -0.02, 'Observed', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig11_rf_pred_obs.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 12. FIGURE 12: STL DECOMPOSITION ─────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(15, 14))
print('Figure 12: STL decomposition (period=11 weeks) into trend, seasonal, residual')
row_titles = ['Original', 'Trend', 'Seasonal', 'Residual']
row_colors = [None, '#2C3E50', '#8E44AD', '#7F8C8D']
for col, pest in enumerate(PESTS):
    sub  = df[df['Pest'] == pest].sort_values(['Year', 'Std_Week_No'])
    ts   = sub['Total'].values.astype(float)
    res  = STL(ts, period=11, robust=True).fit()
    for row, (comp, rcolor, rlabel) in enumerate(zip(
        [ts, res.trend, res.seasonal, res.resid], row_colors, row_titles
    )):
        ax    = axes[row, col]
        color = COLORS[pest] if row == 0 else rcolor
        ax.plot(comp, color=color, linewidth=1.5)
        if row == 3: ax.axhline(0, color='red', linestyle='--', linewidth=1)
        ax.spines[['top', 'right']].set_visible(False)
        if row == 0: ax.set_title(pest, fontweight='bold', color=COLORS[pest])
        if col == 0: ax.set_ylabel(rlabel, fontweight='bold')
fig.text(0.5, 0.01, 'Observation Index', ha='center', fontsize=12, fontweight='bold')
fig.suptitle('STL Seasonal Decomposition', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('figures/fig12_stl_decomposition.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 13. FIGURE 13: WEATHER TRENDS ────────────────────────────────────────
wdf   = df[df['Pest'] == 'Maruca'].copy()
wlabs = {'Tmax': 'Max Temp (C)', 'Tmin': 'Min Temp (C)', 'RHI': 'RH Morning (%)',
         'RHII': 'RH Evening (%)', 'Rainfall_mm': 'Rainfall (mm)', 'SSH': 'Sunshine Hours'}
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
w_handles, w_labels = [], []
print('Figure 13: Weekly weather parameter trends (2020-2024) with 5-year mean')
for i, wvar in enumerate(WEATHER):
    ax = axes.flatten()[i]
    for yr, grp in wdf.groupby('Year'):
        line, = ax.plot(grp['Std_Week_No'], grp[wvar], marker='o', markersize=4,
                        color=YEAR_COLORS[yr], linewidth=1.5)
        if i == 0: w_handles.append(line); w_labels.append(str(yr))
    mw = wdf.groupby('Std_Week_No')[wvar].mean()
    ml, = ax.plot(mw.index, mw.values, 'k--', linewidth=2.5, zorder=5)
    if i == 0: w_handles.append(ml); w_labels.append('Mean')
    ax.set_title(wlabs[wvar], fontweight='bold')
    ax.set_ylabel(wlabs[wvar], fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
fig.text(0.5, 0.01, 'Standard Week', ha='center', fontsize=12, fontweight='bold')
fig.legend(w_handles, w_labels, title='Year', loc='center right',
           bbox_to_anchor=(1.0, 0.5), fontsize=9, title_fontsize=10, framealpha=0.9)
plt.tight_layout(rect=[0, 0.04, 0.93, 1])
plt.savefig('figures/fig13_weather_trends.png', dpi=180, bbox_inches='tight')
plt.close()

# =============================================================================
# SECTION B: FORECASTING
# =============================================================================

# %% ── 14. TABLE 11 + FIGURES 14-15: PRE-DIAGNOSTIC TESTS ──────────────────
diag_rows = []
for pest in PESTS:
    sub   = df[df['Pest'] == pest].sort_values(['Year', 'Std_Week_No'])
    ts    = sub['Total'].values.astype(float)
    adf_r = adfuller(ts, autolag='AIC')
    kp_r  = kpss(ts, regression='c', nlags='auto')
    X_ols = np.column_stack([np.arange(len(ts)), np.ones(len(ts))])
    resid = ts - X_ols @ np.linalg.lstsq(X_ols, ts, rcond=None)[0]
    dw    = durbin_watson(resid)
    lb_p  = acorr_ljungbox(ts, lags=[10], return_df=True)['lb_pvalue'].values[0]
    sw_w, sw_p = shapiro(resid)
    diag_rows.append({
        'Pest': pest,
        'ADF_stat': round(adf_r[0], 4), 'ADF_p': round(adf_r[1], 4),
        'ADF_result': 'Stationary' if adf_r[1] < 0.05 else 'Non-stationary',
        'KPSS_stat': round(kp_r[0], 4), 'KPSS_p': round(kp_r[1], 4),
        'KPSS_result': 'Stationary' if kp_r[1] > 0.05 else 'Non-stationary',
        'DW_stat': round(dw, 4),
        'DW_result': 'No autocorr' if 1.5 < dw < 2.5 else 'Autocorr present',
        'LjungBox_p': round(lb_p, 4),
        'LB_result': 'No serial corr' if lb_p > 0.05 else 'Serial autocorr',
        'SW_W': round(sw_w, 4), 'SW_p': round(sw_p, 4),
        'SW_result': 'Normal' if sw_p > 0.05 else 'Non-normal',
    })
df_diag = pd.DataFrame(diag_rows)
df_diag.to_csv('results/table11_pre_diagnostic.csv', index=False)
print('Table 11 saved: Pre-diagnostic tests for time-series forecasting suitability')

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
print('Figure 14: Pre-diagnostic bar charts (ADF, KPSS, Durbin-Watson, Ljung-Box)')
tests = [
    ('ADF_p',      'ADF p-value',          0.05,       'below'),
    ('KPSS_p',     'KPSS p-value',         0.05,       'above'),
    ('DW_stat',    'Durbin-Watson',         [1.5, 2.5], 'range'),
    ('LjungBox_p', 'Ljung-Box p (lag 10)', 0.05,       'above'),
]
x = np.arange(len(PESTS))
for idx, (col, label, thresh, direction) in enumerate(tests):
    ax   = axes[idx]; vals = df_diag[col].values
    if direction == 'below':
        cbars = ['#27AE60' if v < thresh else '#E74C3C' for v in vals]
    elif direction == 'above':
        cbars = ['#27AE60' if v > thresh else '#E74C3C' for v in vals]
    else:
        cbars = ['#27AE60' if thresh[0] <= v <= thresh[1] else '#E74C3C' for v in vals]
    bars = ax.bar(x, vals, color=cbars, edgecolor='white', width=0.5)
    if isinstance(thresh, list):
        ax.axhline(thresh[0], color='navy', linestyle='--', linewidth=1.5, label=f'Lower={thresh[0]}')
        ax.axhline(thresh[1], color='navy', linestyle=':', linewidth=1.5, label=f'Upper={thresh[1]}')
    else:
        ax.axhline(thresh, color='navy', linestyle='--', linewidth=1.5, label=f'a={thresh}')
    ax.set_xticks(x); ax.set_xticklabels(PESTS, rotation=15)
    ax.set_title(label, fontweight='bold'); ax.spines[['top', 'right']].set_visible(False)
    if idx == 0: ax.set_ylabel('Statistic / p-value', fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig14_prediagnostic.png', dpi=180, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
print('Figure 15: ACF and PACF plots (lags 0-20) informing SARIMAX(1,0,1)(1,0,1)_11 order selection')
for col, pest in enumerate(PESTS):
    sub = df[df['Pest'] == pest].sort_values(['Year', 'Std_Week_No'])
    ts  = sub['Total'].values.astype(float)
    plot_acf(ts,  lags=20, ax=axes[0, col], color=COLORS[pest], alpha=0.05, title='')
    axes[0, col].set_title(f'{pest} - ACF', fontweight='bold', color=COLORS[pest])
    axes[0, col].spines[['top', 'right']].set_visible(False)
    if col == 0: axes[0, col].set_ylabel('Autocorrelation', fontweight='bold')
    plot_pacf(ts, lags=20, ax=axes[1, col], color=COLORS[pest], alpha=0.05, title='', method='ywm')
    axes[1, col].set_title(f'{pest} - PACF', fontweight='bold', color=COLORS[pest])
    axes[1, col].spines[['top', 'right']].set_visible(False)
    if col == 0: axes[1, col].set_ylabel('Partial Autocorrelation', fontweight='bold')
fig.text(0.5, 0.01, 'Lag (weeks)', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('figures/fig15_acf_pacf.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 15. LOYO OUT-OF-FOLD PREDICTIONS FOR COO ──────────────────────────────
component_preds = {}
component_true  = {}
kf_fc = KFold(n_splits=5, shuffle=True, random_state=42)

for pest in PESTS:
    sub   = df[df['Pest'] == pest].sort_values(['Year', 'Std_Week_No']).reset_index(drop=True)
    ts    = sub['Total'].values.astype(float)
    X_h   = sub[FEATURES].values
    oof_s = np.zeros(len(ts))
    oof_g = np.zeros(len(ts))
    oof_r = np.zeros(len(ts))
    oof_l = np.zeros(len(ts))
    for yr in sorted(sub['Year'].unique()):
        test_mask  = (sub['Year'] == yr).values
        train_mask = ~test_mask
        ts_tr, ts_te = ts[train_mask], ts[test_mask]
        X_tr, X_te   = X_h[train_mask], X_h[test_mask]
        ex_tr = sub[WEATHER].values[train_mask]
        ex_te = sub[WEATHER].values[test_mask]
        try:
            m = SARIMAX(ts_tr, exog=ex_tr, order=(1, 0, 1), seasonal_order=(1, 0, 1, 11),
                        enforce_stationarity=False, enforce_invertibility=False)
            oof_s[test_mask] = np.maximum(
                m.fit(disp=False, maxiter=300).forecast(len(ts_te), exog=ex_te), 0)
        except:
            oof_s[test_mask] = ts_tr.mean()
        try:
            X_sm = sm.add_constant(pd.DataFrame(X_tr, columns=FEATURES))
            nb   = sm.GLM(ts_tr, X_sm, family=sm.families.NegativeBinomial()).fit(disp=False)
            X_te_df = pd.DataFrame(sm.add_constant(X_te, has_constant='add'), columns=X_sm.columns)
            oof_g[test_mask] = np.maximum(nb.predict(X_te_df).values, 0)
        except:
            oof_g[test_mask] = ts_tr.mean()
        rfm = RandomForestRegressor(n_estimators=300, max_depth=10,
                                    min_samples_leaf=2, random_state=42)
        rfm.fit(X_tr, ts_tr)
        oof_r[test_mask] = np.maximum(rfm.predict(X_te), 0)
        sc = StandardScaler(); X_sc_tr = sc.fit_transform(X_tr); X_sc_te = sc.transform(X_te)
        lm = LassoCV(cv=3, random_state=42, max_iter=5000)
        lm.fit(X_sc_tr, np.log1p(ts_tr))
        oof_l[test_mask] = np.expm1(np.maximum(lm.predict(X_sc_te), 0))
    component_preds[pest] = np.column_stack([oof_s, oof_g, oof_r, oof_l])
    component_true[pest]  = ts
    print(f'{pest} LOYO R2 -- SARIMAX:{r2_score(ts,oof_s):.3f} '
          f'GLM:{r2_score(ts,oof_g):.3f} RF:{r2_score(ts,oof_r):.3f} LASSO:{r2_score(ts,oof_l):.3f}')

# %% ── 16. COO WEIGHT OPTIMISATION ──────────────────────────────────────────
coo_weights   = {}
naive_weights = {}
loyo_metrics  = {}

for pest in PESTS:
    preds    = component_preds[pest]
    y        = component_true[pest]
    loyo_r2  = [max(r2_score(y, preds[:, i]), 0) for i in range(4)]
    total_w  = sum(loyo_r2)
    naive_w  = np.array(loyo_r2) / total_w if total_w > 0 else np.ones(4) / 4

    if HAS_COO:
        def objective(w):
            w = np.maximum(w, 0); s = w.sum()
            if s < 1e-12: return -1e9
            return float(r2_score(y, preds @ (w / s)))
        opt = COO(bounds=[(0.0, 1.0)] * 4, n_packs=4, init_pack_size=15,
                  max_iterations=150, surrogate_enabled=True, surrogate_kind='rf',
                  surrogate_min_samples=15, random_state=42, verbose=False)
        best_w, _, conv_hist, diag, _ = opt.optimize(objective)
        best_w = np.maximum(best_w, 0); best_w /= best_w.sum()
        coo_w  = best_w
    else:
        coo_w     = naive_w.copy()
        conv_hist = [float(r2_score(y, preds @ naive_w))]
        diag      = {'exact_evals': 0, 'iterations': 0}

    naive_ens = np.maximum(preds @ naive_w, 0)
    coo_ens   = np.maximum(preds @ coo_w, 0)
    coo_weights[pest]   = coo_w
    naive_weights[pest] = naive_w
    loyo_metrics[pest]  = {
        'naive': {'r2':   r2_score(y, naive_ens),
                  'rmse': np.sqrt(mean_squared_error(y, naive_ens)),
                  'mae':  mean_absolute_error(y, naive_ens)},
        'coo':   {'r2':   r2_score(y, coo_ens),
                  'rmse': np.sqrt(mean_squared_error(y, coo_ens)),
                  'mae':  mean_absolute_error(y, coo_ens)},
        'conv':  conv_hist, 'diag': diag,
    }
    print(f'{pest}: Naive R2={loyo_metrics[pest]["naive"]["r2"]:.4f} '
          f'-> COO R2={loyo_metrics[pest]["coo"]["r2"]:.4f}  '
          f'weights={np.round(coo_w, 3)}')

# %% ── 17. TABLE 12 + FIGURE 16: FULL-TRAINING FORECASTING MODELS ──────────
forecast_results = {}
metric_rows      = []
weeks_2025       = df25['Std_Week_No'].values

for pest in PESTS:
    sub      = df[df['Pest'] == pest].sort_values(['Year', 'Std_Week_No']).reset_index(drop=True)
    ts       = sub['Total'].values.astype(float)
    X_h      = sub[FEATURES].values
    ex_hist  = sub[WEATHER].values
    ex_2025  = df25[WEATHER].values
    X_pred   = df25[FEATURES].values

    try:
        mod   = SARIMAX(ts, exog=ex_hist, order=(1, 0, 1), seasonal_order=(1, 0, 1, 11),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit_s = mod.fit(disp=False, maxiter=500)
        in_s  = np.maximum(fit_s.fittedvalues, 0)
        fc_s  = np.maximum(fit_s.forecast(11, exog=ex_2025), 0)
        sar_r2, sar_rmse = r2_score(ts, in_s), np.sqrt(mean_squared_error(ts, in_s))
        sar_aic, sar_bic = fit_s.aic, fit_s.bic
    except Exception as e:
        print(f'{pest} SARIMAX error: {e}')
        fc_s = in_s = np.zeros(11)
        sar_r2 = sar_rmse = sar_aic = sar_bic = np.nan

    X_sm     = sm.add_constant(sub[FEATURES])
    nb_m     = sm.GLM(sub['Total'], X_sm, family=sm.families.NegativeBinomial()).fit(disp=False)
    X_p25_df = pd.DataFrame(sm.add_constant(X_pred, has_constant='add'), columns=X_sm.columns)
    fc_g     = np.maximum(nb_m.predict(X_p25_df).values, 0)
    glm_r2   = r2_score(sub['Total'], nb_m.fittedvalues)

    rfm      = RandomForestRegressor(n_estimators=300, max_depth=10,
                                     min_samples_leaf=2, random_state=42)
    r2_cv_rf = cross_val_score(rfm, X_h, ts, cv=kf_fc, scoring='r2')
    rfm.fit(X_h, ts)
    fc_r     = np.maximum(rfm.predict(X_pred), 0)

    sc       = StandardScaler(); X_sc = sc.fit_transform(X_h); X_p25_sc = sc.transform(X_pred)
    lm       = LassoCV(cv=5, random_state=42, max_iter=10000)
    lm.fit(X_sc, np.log1p(ts))
    r2_cv_la = cross_val_score(lm, X_sc, np.log1p(ts), cv=kf_fc, scoring='r2')
    fc_l     = np.expm1(np.maximum(lm.predict(X_p25_sc), 0))

    fc_stack = np.column_stack([fc_s, fc_g, fc_r, fc_l])
    fc_naive = np.maximum(fc_stack @ naive_weights[pest], 0)
    fc_coo   = np.maximum(fc_stack @ coo_weights[pest], 0)

    forecast_results[pest] = {
        'ts': ts, 'in_sample_sarimax': in_s,
        'sarimax_fc': fc_s, 'glm_fc': fc_g, 'rf_fc': fc_r, 'lasso_fc': fc_l,
        'naive_ensemble_fc': fc_naive, 'coo_ensemble_fc': fc_coo,
        'weeks_2025': weeks_2025,
        'naive_weights': naive_weights[pest], 'coo_weights': coo_weights[pest],
        'sarimax_aic': sar_aic, 'sarimax_bic': sar_bic,
        'sarimax_r2': sar_r2, 'sarimax_rmse': sar_rmse,
        'rf_cv_r2': r2_cv_rf.mean(), 'rf_cv_r2_std': r2_cv_rf.std(),
        'lasso_cv_r2': r2_cv_la.mean(), 'glm_r2': glm_r2,
        'loyo_r2_coo':     loyo_metrics[pest]['coo']['r2'],
        'loyo_rmse_coo':   loyo_metrics[pest]['coo']['rmse'],
        'loyo_mae_coo':    loyo_metrics[pest]['coo']['mae'],
        'loyo_r2_naive':   loyo_metrics[pest]['naive']['r2'],
        'loyo_rmse_naive': loyo_metrics[pest]['naive']['rmse'],
        'loyo_mae_naive':  loyo_metrics[pest]['naive']['mae'],
        'coo_conv': loyo_metrics[pest]['conv'],
        'coo_diag': loyo_metrics[pest]['diag'],
    }
    metric_rows.append({
        'Pest': pest,
        'SARIMAX_AIC':  round(sar_aic, 2) if not np.isnan(sar_aic) else 'NA',
        'SARIMAX_BIC':  round(sar_bic, 2) if not np.isnan(sar_bic) else 'NA',
        'SARIMAX_R2':   round(sar_r2, 3) if not np.isnan(sar_r2) else 'NA',
        'SARIMAX_RMSE': round(sar_rmse, 2) if not np.isnan(sar_rmse) else 'NA',
        'RF_CV_R2':     f'{r2_cv_rf.mean():.3f}+/-{r2_cv_rf.std():.3f}',
        'LASSO_CV_R2':  round(r2_cv_la.mean(), 3),
        'GLM_R2':       round(glm_r2, 3),
    })

pd.DataFrame(metric_rows).to_csv('results/table12_forecast_model_metrics.csv', index=False)
print('Table 12 saved: Forecasting model performance (SARIMAX AIC/BIC/R2, RF/LASSO CV R2)')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
print('Figure 16: SARIMAX(1,0,1)(1,0,1)_11 in-sample fitted vs observed (training 2020-2024)')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    r = forecast_results[pest]
    ax.plot(np.arange(len(r['ts'])), r['ts'], color=COLORS[pest],
            linewidth=2, label='Observed', marker='o', markersize=4)
    ax.plot(np.arange(len(r['ts'])), r['in_sample_sarimax'], 'k--',
            linewidth=2, label='SARIMAX Fitted')
    ax.set_title(f'{pest}\nR2={r["sarimax_r2"]:.3f}, RMSE={r["sarimax_rmse"]:.2f}',
                 fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    if idx == 0: ax.set_ylabel('Weekly Count', fontweight='bold'); ax.legend(fontsize=9)
fig.text(0.5, -0.02, 'Observation Index (2020-2024)', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig16_sarimax_fit.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 18. TABLE 13 + FIGURES 17-18: COO ENSEMBLE PERFORMANCE ───────────────
cmp_rows = []
for pest in PESTS:
    r = forecast_results[pest]
    cmp_rows.append({
        'Pest': pest,
        'Naive_R2':       round(r['loyo_r2_naive'],   4),
        'Naive_RMSE':     round(r['loyo_rmse_naive'],  3),
        'Naive_MAE':      round(r['loyo_mae_naive'],   3),
        'COO_R2':         round(r['loyo_r2_coo'],      4),
        'COO_RMSE':       round(r['loyo_rmse_coo'],    3),
        'COO_MAE':        round(r['loyo_mae_coo'],     3),
        'DeltaR2':        round(r['loyo_r2_coo'] - r['loyo_r2_naive'], 4),
        'DeltaRMSE_pct':  round((r['loyo_rmse_naive'] - r['loyo_rmse_coo']) /
                                r['loyo_rmse_naive'] * 100, 2),
        'COO_weights':    str(np.round(r['coo_weights'], 4).tolist()),
    })
pd.DataFrame(cmp_rows).to_csv('results/table13_ensemble_comparison_coo.csv', index=False)
print('Table 13 saved: LOYO ensemble comparison (naive vs COO-optimised weights, delta RMSE%)')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
print('Figure 17: COO convergence curves showing LOYO ensemble R2 across optimisation iterations')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    conv   = forecast_results[pest]['coo_conv']
    n_iter = forecast_results[pest]['coo_diag'].get('iterations', len(conv))
    ax.plot(range(len(conv)), conv, color=COLORS[pest], linewidth=2.5, marker='o', markersize=3)
    ax.set_title(f'{pest}\nFinal R2={conv[-1]:.4f} ({n_iter} iters)',
                 fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    if idx == 0: ax.set_ylabel('Ensemble R2 (LOYO CV)', fontweight='bold')
fig.text(0.5, -0.02, 'COO Iteration', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig17_coo_convergence.png', dpi=180, bbox_inches='tight')
plt.close()

model_names = ['SARIMAX', 'NB-GLM', 'RF', 'LASSO']
fig, axes   = plt.subplots(2, 3, figsize=(18, 9))
print('Figure 18: Naive vs COO ensemble weights (top) and LOYO metric comparison (bottom)')
for col, pest in enumerate(PESTS):
    r     = forecast_results[pest]
    x     = np.arange(len(model_names))
    width = 0.35
    ax    = axes[0, col]
    ax.bar(x - width / 2, r['naive_weights'], width, color='#7F8C8D',
           label='Naive', alpha=0.8, edgecolor='white')
    ax.bar(x + width / 2, r['coo_weights'],   width, color=COLORS[pest],
           label='COO',   alpha=0.9, edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
    ax.set_title(f'{pest} - Ensemble Weights', fontweight='bold', color=COLORS[pest])
    ax.spines[['top', 'right']].set_visible(False)
    if col == 0: ax.set_ylabel('Weight', fontweight='bold'); ax.legend(fontsize=9)
    ax2     = axes[1, col]
    metrics = ['R2', 'RMSE', 'MAE']
    naive_v = [r['loyo_r2_naive'], r['loyo_rmse_naive'], r['loyo_mae_naive']]
    coo_v   = [r['loyo_r2_coo'],   r['loyo_rmse_coo'],   r['loyo_mae_coo']]
    x2      = np.arange(len(metrics))
    ax2.bar(x2 - width / 2, naive_v, width, color='#7F8C8D',
            label='Naive', alpha=0.8, edgecolor='white')
    ax2.bar(x2 + width / 2, coo_v,   width, color=COLORS[pest],
            label='COO',   alpha=0.9, edgecolor='white')
    ax2.set_xticks(x2); ax2.set_xticklabels(metrics)
    ax2.set_title(f'{pest} - LOYO Metrics', fontweight='bold', color=COLORS[pest])
    ax2.spines[['top', 'right']].set_visible(False)
    if col == 0: ax2.set_ylabel('Value', fontweight='bold'); ax2.legend(fontsize=9)
    delta = (r['loyo_rmse_naive'] - r['loyo_rmse_coo']) / r['loyo_rmse_naive'] * 100
    ax2.text(x2[1] + width / 2, coo_v[1] * 1.04,
             f'-{delta:.1f}%', ha='center', fontsize=9, color='darkgreen', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig18_coo_weights_comparison.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 19. TABLE 14 + FIGURES 19-21: 2025 OUT-OF-SAMPLE FORECASTS ──────────
rows_fc = []
for pest in PESTS:
    r = forecast_results[pest]
    for i, wk in enumerate(r['weeks_2025']):
        rows_fc.append({
            'Pest': pest, 'Std_Week_No': int(wk),
            'SARIMAX':        round(r['sarimax_fc'][i], 2),
            'NB_GLM':         round(r['glm_fc'][i], 2),
            'RandomForest':   round(r['rf_fc'][i], 2),
            'LASSO':          round(r['lasso_fc'][i], 2),
            'Naive_Ensemble': round(r['naive_ensemble_fc'][i], 2),
            'COO_Ensemble':   round(r['coo_ensemble_fc'][i], 2),
        })
pd.DataFrame(rows_fc).to_csv('results/table14_forecast_2025.csv', index=False)
print('Table 14 saved: 2025 out-of-sample weekly forecasts (weeks 42-52, all models + ensembles)')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
print('Figure 19: 2025 out-of-sample forecasts - all models, naive and COO ensemble vs 2024 observed')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    r     = forecast_results[pest]
    sub24 = df[(df['Pest'] == pest) & (df['Year'] == 2024)].sort_values('Std_Week_No')
    ax.plot(sub24['Std_Week_No'], sub24['Total'], 'k-', linewidth=2,
            marker='s', markersize=5, label='2024 Observed', zorder=10)
    ax.plot(weeks_2025, r['sarimax_fc'],        color=FC_COLORS['SARIMAX'],
            linewidth=1.8, marker='^', markersize=5, linestyle='--', label='SARIMAX')
    ax.plot(weeks_2025, r['glm_fc'],            color=FC_COLORS['NB_GLM'],
            linewidth=1.8, marker='o', markersize=5, linestyle=':', label='NB-GLM')
    ax.plot(weeks_2025, r['rf_fc'],             color=FC_COLORS['RandomForest'],
            linewidth=1.8, marker='D', markersize=5, linestyle='-.', label='Random Forest')
    ax.plot(weeks_2025, r['lasso_fc'],          color=FC_COLORS['LASSO'],
            linewidth=1.8, marker='v', markersize=5, linestyle='--', label='LASSO')
    ax.plot(weeks_2025, r['naive_ensemble_fc'], color=FC_COLORS['Naive'],
            linewidth=2.5, marker='*', markersize=7, alpha=0.7, label='Naive Ensemble')
    ax.plot(weeks_2025, r['coo_ensemble_fc'],   color=FC_COLORS['COO'],
            linewidth=3, marker='P', markersize=9, label='COO Ensemble', zorder=9)
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.set_xticks(range(42, 53)); ax.spines[['top', 'right']].set_visible(False)
    if idx == 0: ax.set_ylabel('Forecast Count (per 5 plants)', fontweight='bold')
    if idx == 1: ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
fig.text(0.5, -0.02, 'Standard Week (2025)', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig19_forecast2025.png', dpi=180, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
print('Figure 20: Historical infestation (2020-2024) overlaid with 2025 COO ensemble forecast')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    r   = forecast_results[pest]
    sub = df[df['Pest'] == pest].sort_values(['Year', 'Std_Week_No'])
    for yr, grp in sub.groupby('Year'):
        wk = grp.groupby('Std_Week_No')['Total'].sum()
        ax.plot(wk.index, wk.values, marker='o', markersize=4,
                color=YEAR_COLORS[yr], linewidth=1.8, alpha=0.7, label=str(yr))
    ax.plot(weeks_2025, r['coo_ensemble_fc'], color='#FF6B35', linewidth=3,
            marker='P', markersize=9, label='2025 (COO Ensemble)', zorder=10)
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    ax.set_xticks(range(42, 53)); ax.spines[['top', 'right']].set_visible(False)
    if idx == 0: ax.set_ylabel('Total Count (per 5 plants)', fontweight='bold')
    if idx == 2: ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
fig.text(0.5, -0.02, 'Standard Week', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig20_historical_forecast.png', dpi=180, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
print('Figure 21: Heatmap of 2025 forecasts across all models and COO ensemble (weeks 42-52)')
for idx, (ax, pest) in enumerate(zip(axes, PESTS)):
    r      = forecast_results[pest]
    fc_mat = np.array([r['sarimax_fc'], r['glm_fc'], r['rf_fc'],
                       r['lasso_fc'], r['naive_ensemble_fc'], r['coo_ensemble_fc']])
    fc_df  = pd.DataFrame(fc_mat,
                          index=['SARIMAX', 'NB-GLM', 'RF', 'LASSO', 'Naive Ens.', 'COO Ens.'],
                          columns=[f'Wk{int(w)}' for w in weeks_2025])
    sns.heatmap(fc_df, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.5,
                cbar=(idx == 2), annot_kws={'size': 8},
                cbar_kws={'label': 'Count'} if idx == 2 else {})
    ax.set_title(pest, fontweight='bold', color=COLORS[pest])
    if idx == 0: ax.set_ylabel('Model', fontweight='bold')
fig.text(0.5, -0.02, 'Standard Week (2025)', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig21_forecast_heatmap.png', dpi=180, bbox_inches='tight')
plt.close()

# %% ── 20. SAVE PICKLE & PRINT SUMMARY ──────────────────────────────────────
with open('results/forecast_results.pkl', 'wb') as f:
    pickle.dump(forecast_results, f)

print('\n' + '=' * 60)
print('ALL OUTPUTS COMPLETE')
print('=' * 60)
print('\nFigures  ->  figures/   (fig1 - fig21)')
print('Tables   ->  results/   (table1 - table14)\n')
rows = [
    ( 1, 'table1_descriptive_stats.csv',          'fig1_yearwise_bar.png'),
    ( 2, '',                                        'fig2_stacked_bar.png'),
    ( 3, '',                                        'fig3_weekly_lines.png'),
    ( 4, '',                                        'fig4_boxplots.png'),
    ( 5, '',                                        'fig5_heatmap.png'),
    ( 6, 'table5_spearman_correlation.csv',        'fig6_spearman_heatmap.png'),
    ( 7, '',                                        'fig7_scatter_phenology.png'),
    ( 8, 'table6_glm_fit_summary.csv + table7-9',  'fig8_glm_coef.png'),
    ( 9, 'table10_ml_performance.csv',             'fig9_rf_importance.png'),
    (10, '',                                        'fig10_lasso_coef.png'),
    (11, '',                                        'fig11_rf_pred_obs.png'),
    (12, '',                                        'fig12_stl_decomposition.png'),
    (13, '',                                        'fig13_weather_trends.png'),
    (14, 'table11_pre_diagnostic.csv',             'fig14_prediagnostic.png'),
    (15, '',                                        'fig15_acf_pacf.png'),
    (16, 'table12_forecast_model_metrics.csv',     'fig16_sarimax_fit.png'),
    (17, 'table13_ensemble_comparison_coo.csv',    'fig17_coo_convergence.png'),
    (18, '',                                        'fig18_coo_weights_comparison.png'),
    (19, 'table14_forecast_2025.csv',              'fig19_forecast2025.png'),
    (20, '',                                        'fig20_historical_forecast.png'),
    (21, '',                                        'fig21_forecast_heatmap.png'),
]
for n, tbl, fig_f in rows:
    tbl_str = f'  Table: {tbl}' if tbl else ''
    print(f'  Figure {n:2d}: figures/{fig_f}{tbl_str}')
