# Manuscript changes for the five-split COSMOS-Web results

The paper source is not stored in this repository, so apply the following edits in the manuscript. The tracked result parquets still contain the original ten seeds for provenance; the updated canonical COSMOS-Web and scaling artifacts use seeds 0--4 only. The separate full-width PCA control remains a historical ten-seed analysis.

## 1. Linear-probe protocol — required

In **Physical interpretation experiments**, replace:

> For both HSC and JWST, we run 10-fold cross-validation using the 45,000 selected galaxies with matched physical parameter measurements from the COSMOS-Web catalog (40,000 train / 5,000 validation per fold).

with:

> For both HSC and JWST, we evaluate five repeated random train--test splits of the 45,000 selected galaxies with matched physical-parameter measurements from the COSMOS-Web catalog (40,000 training and 5,000 test galaxies per split). The independently seeded test sets may overlap and are not disjoint cross-validation folds.

## 2. Main COSMOS-Web results table — replace

Replace the old ten-fold/ten-seed table with:

```latex
\input{figures/cosmosweb_table.tex}
```

Copy or link `assets/plots/cosmosweb_table.tex` to that manuscript path. Its caption and all means and standard deviations now use seeds 0--4.

If the manuscript states that Gestalt/MCCA is significant at $p=0.002$, remove that claim. With five paired splits, Gestalt beats the best single model in every split and ranks first in all six cells, but the exact two-sided Wilcoxon result is $p=0.0625$ in each cell—the smallest value attainable with five pairs.

## 3. Main physics-scaling figure and text — replace and revise

Replace any model-size plot computed from ten COSMOS-Web splits with:

- `assets/plots/cosmos_r2_vs_model_size.pdf` for the property-averaged figure;
- `assets/plots/cosmos_r2_vs_model_size_per_property.pdf` for the appendix breakdown.

For the property-averaged result, report:

```latex
A Spearman rank test gives a positive correlation between model capacity and mean probe performance for JWST ($\rho=0.630$, $p=0.0017$) and HSC ($\rho=0.472$, $p=0.026$).
```

Do **not** retain the current sentence claiming significance in every per-property panel. With five splits, the updated per-property tests are:

| Modality | Property | $\rho$ | $p$ |
| --- | ---: | ---: | ---: |
| HSC | redshift | 0.399 | 0.066 |
| HSC | $\log M_\star$ | 0.378 | 0.083 |
| HSC | sSFR | 0.440 | 0.041 |
| JWST | redshift | 0.608 | 0.0027 |
| JWST | $\log M_\star$ | 0.596 | 0.0034 |
| JWST | sSFR | 0.543 | 0.0090 |

Suggested appendix-caption ending:

```latex
The trend is significant for all three JWST properties and HSC sSFR; HSC redshift and stellar mass show positive but non-significant trends under the five-split protocol.
```

## 4. Figure 1 (Gestalt versus individual models) — replace

Use `assets/plots/basket_vs_singles_pooled_cosmos1024.pdf`. It now reports the means and standard errors from seeds 0--4. The pooled values are:

| Modality | Gestalt | concat$\to$PCA | best single |
| --- | ---: | ---: | ---: |
| HSC | $0.568\pm0.006$ | $0.538\pm0.003$ | 0.516 |
| JWST | $0.663\pm0.007$ | $0.623\pm0.007$ | 0.615 |

The uncertainties in this figure are standard errors; the table uncertainties are standard deviations.

## 5. Basket-size scaling figure and tables — replace

Use:

- `assets/plots/scaling_curves.pdf`;
- `assets/plots/scaling_curves_per_property.pdf` where the appendix needs individual properties;
- `assets/plots/scaling_values_table.tex`;
- `assets/plots/scaling_fits_table.tex`.

If the text quotes the average scaling sequence, replace it with:

```latex
Mean $R^2$ across the six COSMOS-Web cells increases from 0.514 at $k=2$ to 0.558, 0.593, 0.609, and 0.615 at $k=4,8,16,$ and 22, respectively; $k=8$ captures approximately 79\% of the total $k=2\to22$ gain.
```

## 6. Alignment ablation text — revise statistics only

The D=256 alignment-mode parquet also contains the same ten historical probe seeds. When reporting the five-split protocol, retain seeds 0--4 only. MCCA remains rank 1 in every cell, but replace any statement such as “MCCA wins every cell ($p\leq0.01$)” with:

```latex
MCCA has the highest mean $R^2$ in every cell and exceeds the best single model in all five retained splits. Exact paired two-sided Wilcoxon tests give $p=0.0625$ in each cell and are therefore descriptive at this sample size.
```

## 7. Figure 3 — no change

`assets/plots/probes_confusion.pdf` is based on one full-data probe fit rather than repeated train--test seeds. Its values, caption, and interpretation do not change when COSMOS-Web reporting is reduced to five splits.

## 8. Global terminology sweep — required

Search the manuscript for all of the following and update COSMOS-Web references:

- `10-fold cross-validation`
- `10 k-folds`
- `10 seeds`
- `ten seeds`
- `per fold`
- `p=0.002` or `p = 0.002`
- claims that every per-property size correlation is significant

Use **five repeated random train--test splits** or **five repeated holdouts**, never **five-fold cross-validation**.
