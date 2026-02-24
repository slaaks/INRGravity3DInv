# Revision TODO — Scientific Reports

Manuscript: *"Three-dimensional inversion of gravity data using implicit neural representations"*

Editor: Weiying Chen | Status: **Major revision**

---

## Experiments

- [x] **Hash encoding comparison** — Run `001-EncodingComparisons.py` (already has hash encoding impl), collect results vs Fourier PE — code ready, needs running
- [x] **TV-regularized baseline** — Add TV penalty to `v01/04-BlockModel.py`, compare INR vs TV on block model
- [x] **Broader hyperparameter ablation** — Sweep width × depth × L × noise; present in Supplementary — partial
- [x] **Noise sensitivity on block model** — Run block model at multiple noise levels + show ensemble spread — GRF done, block model not
- [x] **Multi-seed ensemble** — 20-member ensemble with varied seeds/noise already in `002-ModelEnsambles.py` — done

## Manuscript Writing

- [ ] **Formal inverse-problem statement** — Define objective function, θ, G, W_d mathematically
- [ ] **Justify no explicit regularization** — Explain implicit priors (capacity, tanh, PE band-limit); cite Deep Image Prior, spectral bias
- [ ] **Table of implicit regularization knobs** — Width, depth, L, tanh bound, optimizer, epochs
- [ ] **Resolution & uncertainty discussion** — No resolution matrix; ensemble as empirical substitute
- [ ] **Limitations section** — Synthetic-only, implicit regularization, no resolution matrix, scaling, hyperparameter sensitivity
- [ ] **Statistical methodology** — Noise model, data weighting, RMS definition, ensemble stats
- [ ] **Uncertainty handling** — Ensemble workflow, member filtering, meaning of std/CV maps
- [ ] **Temper conclusions** — "without regularization" → "without *explicit* regularization"; qualify claims

## Restructure

- [ ] **Demote spectral-bias & network-size experiments** to Supplementary
- [ ] **Replace smoothness baseline with TV** in main block-model comparison

## References to Add

- [ ] Müller et al. (2022) — Instant NGP / hash encoding
- [ ] Ulyanov et al. (2018) — Deep Image Prior 
- [ ] Rahaman et al. (2019) — Spectral Bias
- [ ] Rudin, Osher & Fatemi (1992) — TV regularization
- [ ] Lakshminarayanan et al. (2017) — Deep Ensembles

## Priority

1. TV baseline + hash encoding results
2. Ensemble/null-space sweep
3. Mathematical writing (parallel with experiments)
4. Ablation + noise sensitivity (robustness)
5. Restructure + new sections (final assembly)
