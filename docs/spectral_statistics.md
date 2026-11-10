<!-- Rôle : Décrire la méthodologie, les unités et les limites de l’analyse statistique spectrale. -->

# Spectral separation analysis

Adapted from the user's `statistical_analysis.ipynb`. The source notebook is unchanged.
This is a descriptive feature-analysis module, not an explanation of SVM weights,
not permutation importance, and not a validated optimal filter selection.

## Data and units

Uses the selected model's band axis and saved spectral normalization, before scaler/PCA.
Default: median spectra grouped by `(source, class, roi_id)`. Reused ROI IDs do not
merge different classes or cubes. Without source/ROI metadata this mode fails explicitly;
the UI requires ROI provenance and offers no pixel fallback. Pixel mode remains
available only in the analysis helper for diagnostics/tests. Background, petri and both ink classes
are excluded by default; the checkbox includes them. There is no silent fallback.

One deterministic sample with equal counts per class (up to 1,000) is reused across
all wavelengths. The count, unit and source count are displayed. Multiple ROI medians
within a cube are still clustered; cubes are not necessarily independent specimens.
Pixel sampling does not repair this dependence. P-values are nominal/exploratory.

## Metrics

- Tie-corrected Kruskal–Wallis H. Effect: `eta_H² = max(0,(H-k+1)/(N-k))`.
  This preserves the notebook's formula but uses its rank eta-squared name, rather
  than confusing it with rank epsilon-squared `H/(N-1)`.
- Pairwise Cliff's delta: `2 U / (n_a n_b) - 1`, ties handled by average ranks.
  Its magnitude is plotted on a fixed 0–1 scale. Zero means no directional rank
  dominance, NOT identical distributions; 1 means complete ordering in these data.
  This is not classification accuracy. Sign/direction is available on hover/export.
- Dunn: common pooled ranks and tie correction. `2 * norm.sf(abs(z))` avoids
  cancellation from `1-cdf(z)`. BH adjustment covers every band/pair hypothesis
  together. Kruskal BH covers every band. Neither procedure averages p-values over
  subsamples, nor tests only the top bands selected from the same data.
- Mutual information with deterministic estimation; constant bands forced to zero.
- Absolute Spearman correlation for redundancy; constant-band correlations are
  undefined (NaN) and masked in the display, not shown as zero redundancy.
- Candidate list: normalized MI minus mean absolute Spearman correlation with
  selected bands. This is explicitly a heuristic, NOT classical MI-based mRMR.
  Up to eight candidates are retained; highly correlated neighbors can still appear.

KW/Dunn independence assumptions are not established by the ROI data. Small group
counts (<5) disable approximate p-values. No significance stars or claims of
biological significance are made. Candidate sets must be selected inside training
folds and evaluated on held-out specimens before claiming classification benefits.

## References

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
- https://scikit-posthocs.readthedocs.io/en/stable/generated/scikit_posthocs.posthoc_dunn.html
- https://easystats.github.io/effectsize/reference/rank_epsilon_squared.html
- https://scikit-learn.org/stable/modules/permutation_importance.html
