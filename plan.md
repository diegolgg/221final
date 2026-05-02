# Project Plan v2: Online Targeted Subsampling for SG-MCMC in HMMs

*Drafted 2026-05-02. Supersedes the daily-frequency, multi-task framing in the May 5 slides. Writeup deadline 2026-05-12 — 10 days from drafting, ~7 days post-presentation. Scope is calibrated to that window, not to an idealized project.*

---

## What the project is now

A methodological contribution on **online adaptation of importance weights** for stochastic gradient MCMC in long-series HMMs with rare latent states. We extend Ou et al. (2024)'s TASS by relaxing its second core approximation — that block weights computed at $\theta_{\mathrm{MAP}}$ remain valid throughout sampling — using a discounted Bayesian linear regression with Thompson sampling that re-fits the weights from gradient feedback as the chain explores the posterior. We deliver:

1. A method (online Bayesian/Thompson-sampled feature-informed weighting), with a clean component-specific extension paralleling Ou's Corollary 1.
2. Two theoretical results: a staleness bound for static TASS, and a tracking guarantee for the online estimator.
3. Three experiments: one synthetic drift demonstration, plus a gradient-RMSE benchmark and a posterior-recovery experiment on real intraday crypto data.

---

## What we're abandoning, and why

| Abandoning | Reason |
|---|---|
| Daily-frequency data ($T \approx 2000$) | Too short for SG-MCMC to be necessary. Full-batch NUTS handles it directly. The whole subsampling premise evaporates. |
| 10-day stress-prediction logistic regression as a headline result | $N \approx 200$ blocks, convex objective. Doesn't need MCMC, doesn't need subsampling. Defensible as a slide-deck microbenchmark, empty as a final deliverable. |
| Multivariate Gaussian $K=3$ HMM on raw daily returns | Collapsed under all four samplers including Stan/NUTS ($\hat R = 2.21$). The model itself is wrong for this data scale; no sampling improvement can fix it. |
| Multi-task framing (HMM + logistic + benchmark) | Diluted the methodological story. One model, done well. |

---

## What we're keeping

- **Crypto motivation.** Rare-regime detection in financial time series remains the application.
- **Block + buffer + SGLD machinery.** Ma et al. (2017) buffered gradients, JAX autodiff, gradient clipping, multi-chain $\hat R$ diagnostics.
- **Online weighting algorithm core.** Discounted regression of log gradient norm on block features, with $O(p^2)$ per-step recursive update — same compute, cleaner statistics (see method section).
- **Feature engineering for blocks.** Mean abs return, rolling vol, dispersion, tail counts, downside pressure.
- **Gradient-RMSE benchmark concept** (Ou §4.6 analog). Stays as a key methodological diagnostic.

---

## Data and model

### Data
- **15-minute bars** built from tick data, $T \approx 175{,}000$ over 5 years. (5-min would give 525k but 15-min is plenty for the SG-MCMC story and faster to wrangle. Bump up if there's slack on day 8.)
- **Equal-weight basket return** $r_t = \frac{1}{|\mathcal A|}\sum_a r_{t,a}$ across the 5 assets.
- **Realized variance** $\mathrm{RV}_t$ computed from intra-bar tick returns within each 15-min window.
- **Working series:** $y_t = \log\mathrm{RV}_t$, after subtracting an hour-of-day session mean to remove the diurnal pattern.

### Model
$K=2$ Gaussian HMM on $y_t$:

$$
y_t \mid z_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2), \quad z_t \mid z_{t-1} \sim \mathrm{Categorical}(A_{z_{t-1},:}).
$$

States interpreted as low-vol (calm) and high-vol (stressed) regimes. Univariate, identifiable, sidesteps the $K=3$ collapse and the multivariate-covariance label-switching headaches simultaneously. Crypto application is preserved — log-RV regimes are the right object for "rare stress" anyway.

**Priors.** $\mu_k \sim \mathcal{N}(0, 5^2)$, $\sigma_k^2 \sim \mathrm{IG}(2, 1)$, Dirichlet$(1,1)$ on transition rows. Identifiability via $\mu_1 < \mu_2$ ordering constraint (post-hoc relabel of chain samples).

**Why not $K=3$?** We tried $K=3$ on returns and it collapsed under Stan. $K=3$ on log-RV might work, but at 10-day budget the marginal value of a third state isn't worth the identifiability risk.

---

## Method

The current discounted-ridge regression at each SGLD step computes a point estimate $\hat\beta_t = A^{-1}b$ from the recursion $A \leftarrow \rho A + ww^\top$, $b \leftarrow \rho b + wf$. We upgrade this in two ways:

### Upgrade 1: Bayesian regression with Thompson sampling on $\beta$

Treat $\beta$ as a random variable with prior $\beta \sim \mathcal{N}(0, \tau^{-1}I)$ and Gaussian likelihood for the log-squared-norm feedback. Conjugacy gives a closed-form posterior:

$$
\beta \mid \text{history} \sim \mathcal{N}(\hat\mu_t, \hat\Sigma_t), \quad \hat\mu_t = A^{-1}b, \quad \hat\Sigma_t = \sigma^2 A^{-1}.
$$

The posterior mean $\hat\mu_t$ is *identical* to the discounted-ridge solution we already compute, and we already have $A^{-1}$ on hand. So the Bayesian view is essentially free.

At each SGLD step, instead of using the point estimate, we **sample** $\tilde\beta_t \sim \mathcal{N}(\hat\mu_t, \hat\Sigma_t)$ and compute weights $\tilde a_n = \exp(\tfrac12\tilde\beta_t^\top w_n)$. Block sampling probabilities are $p_n \propto \tilde a_n$ — **no $\lambda/N$ uniform exploration floor**. The exploration rate is automatically tuned by the posterior width: wide $\hat\Sigma_t$ early in the chain produces high block-weight variance and natural exploration; tight $\hat\Sigma_t$ late in the chain concentrates the weights and exploits.

This is a textbook contextual-bandit move (Russo–Van Roy 2014, Agrawal–Goyal 2013). Removes one hyperparameter ($\lambda$), gives cleaner exposition, and adds a footnote-worthy bandit-theory connection to the paper.

### Upgrade 2: Component-specific online weighting

Ou's Corollary 1 shows that running $d$ separate sub-samplings (one per parameter component) beats a single shared weighting in the static case. We extend this to the online case: maintain $d$ separate $\hat\beta^{(i)}_t$ vectors, one per parameter component $i \in \{1,\dots,d\}$. Each is fit on the component-specific feedback $f_s^{(i)} = \log(|g_{n_s}^{(i)}(\theta_s)|^2 + \epsilon)$. At each SGLD step, draw $d$ separate mini-batches using the component-specific weights, exactly as Ou does for the static case, but with our online updates.

For our $K=2$ Gaussian HMM, $d = 6$ ($\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, A_{1,2}, A_{2,1}$ after the Dirichlet simplex constraint). Compute scales linearly in $d$; the recursive updates are still $O(p^2)$ per component per step.

This is what makes T2.5 (the component-specific theoretical extension below) a clean direct generalization of Ou Cor 1 to non-stationary $\theta$.

---

## Theoretical contributions

### T2 — Staleness bound and tracking guarantee (must-have)

**Proposition T2.1 (staleness bound).** Under the assumption that $g_n(\theta)$ is $L_n$-Lipschitz in $\theta$ on a neighborhood of $\theta_{\mathrm{MAP}}$, the variance of the static-TASS gradient estimator at $\theta_t$ exceeds the oracle (using $a_n^\star(\theta_t)$) by at most
$$
\mathrm{Var}[\hat\nabla^{\mathrm{TASS}}_t U] - \mathrm{Var}[\hat\nabla^{\mathrm{opt}}_t U] \le C_1 \|\theta_t - \theta_{\mathrm{MAP}}\|^2,
$$
for an explicit constant $C_1$ depending on $\{L_n\}$ and the spectrum of the buffered gradients.

**Proof sketch.** Cauchy–Schwarz on the variance decomposition + Lipschitz on the per-block contributions. ~3–4 hours of focused work to state cleanly and prove.

**Proposition T2.2 (tracking guarantee for online).** For the discounted Bayesian online estimator with discount $\rho \in (0,1)$ and bounded per-step parameter drift $\|\theta_{t+1} - \theta_t\| \le D_t$, the expected tracking error satisfies
$$
\mathbb{E}\|\hat\beta_t - \beta^\star(\theta_t)\|^2 \le C_2\frac{1-\rho}{1-\rho^2}\,\sup_{s\le t} D_s^2 + C_3\,\rho^t,
$$
i.e. tracking error in the steady state is proportional to drift speed, with an exponentially decaying transient from initialization.

**Proof sketch.** Standard discounted RLS analysis (Ljung–Söderström 1983, Haykin Ch. 9) plus a perturbation argument linking $\beta^\star(\theta)$ to $\theta$. ~3–4 hours.

**Corollary.** Online's variance excess over oracle is bounded by $C_4 \cdot \text{drift}$, while static TASS's is bounded by $C_1 \cdot \text{cumulative drift from MAP}$. Online wins whenever the chain has wandered far enough from $\theta_{\mathrm{MAP}}$.

Total T2: ~6–8 hours.

### T2.5 — Component-specific extension (recommended add)

Lift Propositions T2.1 and T2.2 to the per-component case. Mechanical from T2; the proofs go through component by component with $g_n$ replaced by $g_n^{(i)}$. Yields:
- Per-component staleness bound that is *tighter* than the joint bound (each $L_n^{(i)}$ replaces a sum-aggregated $L_n$).
- Per-component tracking guarantee with the same form.

This directly extends Ou Cor 1 to non-stationary $\theta$ and is the cleanest novel theoretical contribution we can claim. ~3 hours on top of T2.

### T3 — End-to-end Wasserstein-2 bound (bonus stretch, attempt only if T2+T2.5 done by day 7)

Compose T2 + T2.5 with Dalalyan–Karagulyan's $W_2$ bound for SGLD with biased/noisy gradients. The fiddly bit: $\theta_t$ is itself driven by the noisy chain, so the variance bound and the chain dynamics are coupled. Workaround: bound everything *in expectation under the posterior*, using Bernstein–von Mises (Cappé et al. for HMMs) to argue $\mathbb{E}_\pi\|\theta - \theta_{\mathrm{MAP}}\|^2 = O(1/T)$. Yields a posterior-error bound on the online-weighted SGLD chain that is tighter than static TASS's whenever drift is non-negligible.

Estimated cost: 3–6 hours **if it works**. Real risk it doesn't, in which case we mention the framework in the paper and leave it as future work. Drop without ceremony if T2 + T2.5 take longer than expected.

---

## Experiments

### E1 — Synthetic drift demonstration (must-have)

Build a small simulator: Ou §4.1 setup ($K=3$ Gaussian, one rare state at 0.5% stationary mass) but with $\mu_3$ slowly drifting from $-20$ to $-15$ over the simulated series. $T = 10^5$ — enough for SG-MCMC to be the right tool, small enough that we can compute the true full gradient periodically.

Compare uniform / static TASS / online (with Bayesian-Thompson upgrade) on:
- Gradient RMSE vs the true full gradient at $t \in \{1\text{k}, 5\text{k}, 10\text{k}, 20\text{k}\}$.
- Parameter recovery error vs the *correct* $\theta(t)$.
- $\hat R$ across 4 chains.

**Expected result:** static TASS variance grows over time (weights stale), online tracks. Direct empirical validation of T2.

Budget: ~3–4 hours.

### E2 — Crypto gradient-RMSE benchmark with drift sweep (should-have)

15-min bars on log-RV basket. Pick $\theta_0$ near $\theta_{\mathrm{MAP}}$ from a Stan reference fit on a 50k-bar subset. Compute the true full gradient at $\theta_0$ (slow but feasible at $T = 175\text{k}$). Run uniform / static TASS / online stochastic gradient estimators many times at $\theta_0$ and compute RMSE.

Then sweep: take $\theta_0 + \delta v$ for various $\delta$ and a perturbation direction $v$ aligned with a posterior eigendirection. Plot RMSE vs $\delta$ for each method. **Expected result:** TASS RMSE grows with $\delta$ (Ou's §4.6 finding), online stays flat (the new finding).

This is the "rubber meets the road for the variance-reduction mechanism" experiment on real data.

Budget: ~5–7 hours including pipeline time.

### E3 — Crypto posterior recovery (nice-to-have)

Run full SGLD with all three methods on the full 175k series, 4 chains, ~10k iterations. Standard diagnostics: $\hat R$, ESS/sec, posterior summaries. Compare the posterior summaries against a Stan/NUTS reference fit on a 50k-bar subset (the part Stan can handle).

**If $\hat R$ is bad for all methods** (possible): report it honestly as a finding about SG-MCMC for HMMs, with a discussion of why label switching and finite-step bias remain hard for SGLD even in identifiable models. Does *not* invalidate E1 + E2.

**If $\hat R$ is good for at least one method:** standard posterior-recovery comparison plots, smoothed state probabilities through known stress events (May 2021, Terra/Luna, FTX).

Budget: ~4–5 hours including overnight runs.

---

## Codebase reuse

A pass through the existing code (Diego's `221finalproject_multichain.ipynb` and Stephen/Haozhe's pilots) shows ~70% of what we need is already implemented. We pull from it rather than rewriting.

### Reuse verbatim (zero or near-zero changes)
| Component | Source |
|---|---|
| JAX HMM forward / backward via `lax.scan` | `221finalproject_multichain.ipynb` cell 49 |
| Buffered block gradient with `stop_gradient` (Ma 2017) | `221finalproject_multichain.ipynb` cells 78–82 |
| SGLD update with gradient clipping + step-size schedule | `simulated_k2_univariate_part_a.ipynb` cell 27 (already has online variant) |
| Uniform sampler | trivial, anywhere |
| Static TASS sampler with $k$-means weights | `221finalproject_multichain.ipynb` cell 146 (`build_young_static_weights`) |
| `DiscountedRidge` class (recursive $A, b$ update) | `221finalproject_multichain.ipynb` cells 177–183 |
| `online_feature_sgld_one_step` harness | `221finalproject_multichain.ipynb` cell 191 |
| $\hat R$, ESS, trace diagnostics | `221finalproject_multichain.ipynb` cell 220, 224 |
| Synthetic K=2 univariate Gaussian HMM simulator | `simulated_k2_univariate_part_a.ipynb` |
| Plotting helpers (state-prob plots, mixing summaries) | `build_paper_assets.py` |

### Adapt lightly
| Component | Change | Effort |
|---|---|---|
| Emission likelihood | Swap multivariate Student-$t$ → univariate Gaussian on log-RV | ~30 min |
| Stan model | Simplify `gaussian_k2_rollstd_hmm.stan` from $d$-variate to univariate | ~1 hour |
| `make_blocks` feature engineering | Univariate features on log-RV instead of multivariate basket features | ~1–2 hours |

### Build new (does not exist in current code)
| Component | Effort |
|---|---|
| Tick → 15-min bar pipeline + realized-variance computation + session de-seasonalization | ~3–4 hours |
| **Bayesian-Thompson upgrade** to `DiscountedRidge` (sample $\tilde\beta_t$ from posterior, drop $\lambda$ floor) | ~1 hour |
| **Component-specific online weighting** (extending Ou Cor 1: $d$ separate $\hat\beta^{(i)}_t$ vectors and $d$ mini-batches per step) | ~2 hours |
| Drift simulator for E1 (extend simulated_k2 to allow time-varying $\theta^*$) | ~2 hours |

**Net effect on budget: -8 to -10 hours of code work** vs the original 36–55 hour estimate.

---

## Code organization

The current codebase is hard to navigate because everything — model, samplers, features, plots, experiments — lives in monolithic 4000-line notebooks, and the same forward-pass code is copy-pasted across five of them. We restructure as:

```
221final/
├── src/                          # Reusable library + one-shot data scripts, .py only
│   ├── data.py                   # Tick → 15-min bars, RV, session de-seas
│   ├── features.py               # Block partitioning + feature computation
│   ├── hmm.py                    # JAX forward/backward, buffered gradient
│   ├── samplers.py               # SGLD + uniform/TASS/online (Bayesian-Thompson)
│   ├── simulator.py              # Synthetic HMM with optional drift
│   ├── diagnostics.py            # R̂, ESS, trace summaries (ArviZ wrappers)
│   ├── plotting.py               # Paper-ready figure helpers
│   ├── build_bars.py             # Tick → 15-min bar artifact (one-shot script)
│   └── build_features.py         # Bars → blocks + feature matrices (one-shot script)
├── experiments/                  # One notebook per experiment, imports from src/
│   ├── e1_synthetic_drift.ipynb
│   ├── e2_gradient_rmse.ipynb
│   └── e3_posterior_recovery.ipynb
├── data/                         # Processed data artifacts (parquet/pkl)
├── stan/                         # Stan models
├── archive/                      # Legacy notebooks and pkls (do not delete; reference)
├── papers/                       # Reference papers
├── paper_assets/                 # Figures + tables consumed by writeup/paper.tex
├── slides/                       # Presentation .tex + build artifacts + PDF
├── writeup/                      # Paper .tex + references.bib + build artifacts
├── plan.md                       # this file
├── outline.pdf                   # original project outline
└── README.md
```

**Why `.py` for the library and `.ipynb` for experiments.** Anything used in more than one place goes in `.py` so we import once and never copy-paste again — that's exactly what made the existing codebase a mess. Notebooks are for *one experiment's* narrative: data load → method runs → plots → artifact dump. One-shot data scripts (`build_bars.py`, `build_features.py`) sit alongside the library since they're small and a separate `pipelines/` directory would be overkill at this scale.

**Cleanup approach: lazy, not big-bang.** We do not spend half a day refactoring before starting work. Steps:

1. **Day 1, ~30 min:** create the new directory skeleton; move all existing `.ipynb`, `.pkl`, `.csv` (except `pivot_plan.md`, `papers/`, `paper_assets/`, `final_project_*`) into `archive/`. Top-level becomes navigable.
2. **As we need each component:** copy the relevant cells from the archived notebooks into a clean `src/*.py` module, adapting as we go. We are not pre-extracting — extraction happens at the moment of use.
3. **No edits to archived notebooks.** They stay frozen as historical artifacts. If we want to verify Diego's results, we go look at them; we don't re-run them.

This keeps total cleanup overhead under an hour and avoids the failure mode of "spend day 1 reorganizing and never start the actual work."

---

## Hour budget

My work (on-the-ground execution, code, math drafting, plots):

| Task | Hours |
|---|---|
| Codebase cleanup (move legacy → `archive/`, scaffold `src/`) | 0.5 |
| Tick → 15-min bar pipeline + RV + session de-seasonalization | 3–4 |
| Block partitioning + univariate-log-RV features | 1–2 |
| Stan model adapt to univariate + reference fit on 50k subset | 2–3 |
| Extract reusable HMM + SGLD code from notebooks into `src/` modules | 1–2 |
| Bayesian-Thompson upgrade to `DiscountedRidge` | 1 |
| Component-specific online weighting | 2 |
| Drift simulator for E1 | 1–2 |
| E1 — synthetic drift experiment + plots | 3–4 |
| E2 — crypto gradient-RMSE benchmark + drift sweep + plots | 5–6 |
| E3 — crypto posterior recovery + diagnostics + plots | 4–5 |
| T2 — state and prove staleness bound + tracking guarantee | 6–8 |
| T2.5 — component-specific extension | 2–3 |
| T3 — bonus stretch (if attempted) | 3–6 |
| Paper-ready figures and tables | 2–3 |
| Writing support (method/results sections, integration with your prose) | 4–6 |

**Total my work: ~28–45 hours** (T3-stretch lands at the high end). Comfortable in 10 days at 3–5 hours/day; tighter if T3 is attempted.

Your work (direction, theory review, paper writing):

| Task | Hours |
|---|---|
| Engaging on direction + reviewing the model decisions | 2–3 |
| Reviewing T2 + T2.5 proofs critically | 2–4 |
| Writing the paper (intro, related work, discussion, integrating my drafts) | 12–18 |
| Final review pass | 2–3 |

**Total your work: ~18–28 hours.** Roughly 2–3 hours/day.

---

## Risk and mitigation

| Risk | Mitigation |
|---|---|
| Stan doesn't mix on log-RV either | Drop to $K=1$ (no HMM, just Gaussian on log-RV) as a fallback model — doesn't affect E1's validity (synthetic). Or relax to log-returns if log-RV has its own issues. |
| Tick data wrangling overruns | Start day 1. Worst case: fall back to pre-built 15-min OHLCV bars from an exchange API (less accurate RV but workable). |
| T3 doesn't go through | Drop. T2 + T2.5 alone are still a real methodological paper. |
| Online doesn't beat TASS in E2 | Diagnostic finding, report honestly. Means either chain doesn't drift far in practice, or feature space is misspecified. Either way it's information for the discussion. |
| E3 mixing is bad for all methods | Report as finding about SG-MCMC for HMMs. E1 and E2 still give us a publishable methodological contribution. |
| I'm slower than budgeted | Cut in this order: T3, E3, T2.5, the Bayesian-Thompson upgrade. Theory T2 + experiments E1 + E2 + adapted SGLD is the irreducible core. |

---

## Execution progression

The plan is to verify the load-bearing pieces *first*, before investing experiment time around them. Two verification gates: T2.1 sketch on day 1, and Stan reference fit by end of day 3 (confirming the model is identifiable). After those, the rest is execution.

### Today and tomorrow (pre-presentation, May 2–4)
| Step | Owner | Time |
|---|---|---|
| You confirm tick data access details (paths, format, date ranges per asset). | You | 0.5 h |
| **Gate 1:** I sketch T2.1 (staleness bound) proof on paper. If the Lipschitz + Cauchy-Schwarz argument is clean, T2 is locked. If it isn't, we revisit theoretical scope before investing further. | Me → you review | 1–2 h |
| Codebase cleanup: move legacy notebooks/data into `archive/`, scaffold `src/` and `experiments/`. | Me | 0.5 h |
| Start the tick → 15-min bar pipeline. | Me | 2–3 h |

### Post-presentation (May 5 onward)
| Day | What lands | Verification gate |
|---|---|---|
| **Day 1 (May 6)** | Bar pipeline finished, log-RV series + features + blocks generated and sanity-checked. T2.2 (tracking) proof written. | Sanity plots: bar prices, log-RV series with stress events overlaid, feature distributions. |
| **Day 2 (May 7)** | Univariate Stan model adapted; NUTS reference fit on 50k subset. SGLD code lifted from notebooks into `src/samplers.py`, adapted to univariate Gaussian. | **Gate 2:** Stan $\hat R < 1.05$ on the 50k subset. If NOT, $K=2$ on log-RV is unidentifiable and we fall back (drop to $K=1$ or model log-returns directly). |
| **Day 3 (May 8)** | Bayesian-Thompson upgrade and component-specific weights implemented. Drift simulator written. E1 (synthetic drift) running. | E1 produces interpretable RMSE-over-time curves. |
| **Day 4 (May 9)** | E1 plots finalized. T2.5 (component-specific) extension written. E2 (crypto gradient RMSE) starts. | T2 + T2.5 are written up to the point where they could go in the paper. |
| **Day 5 (May 10)** | E2 finished with drift sweep. E3 (posterior recovery) started, chains running overnight. | E2 plot shows the predicted online-vs-TASS gap as $\delta$ grows. |
| **Day 6 (May 11)** | E3 diagnostics; if T2+T2.5 are clean, attempt T3. Final figures generated. Paper draft mostly complete. | All experimental results in hand. |
| **Day 7 (May 12)** | Final review pass, integrate everything, submit. | Submission. |

### Cut order if we're behind
If by Day 4 we're behind schedule, cut in this order: T3 → E3 → T2.5 → Bayesian-Thompson upgrade. Surviving core (T2 + E1 + E2 with the basic point-estimate ridge online method) is still a complete paper.