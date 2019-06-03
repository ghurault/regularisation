# Case study on regularisation methods for statistics and machine learning

This case study was made in the context of a tutorial on regularisation methods, which presentation is available in `January_2019_Tutorial_Presentation__Regularisation_.pdf`.

It includes an introduction on Ordinary Least Squares (OLS) regression, the motivations behind regularisation as well as the different interpretations (optimisation, geometric, Bayesian) for common regularisation methods:
- Ordinary Least Squares (no regularisation)
- Lasso (L<sub>1</sub>) regularisation
- Ridge (L<sub>2</sub>) regularisation
- Elastic Net (mixture of L<sub>1</sub> and L<sub>2</sub>)
- Bridge (L<sub>p</sub>) regularisation

Further methods are then discussed to overcome overshrinkage:
- Hybrid Lasso (Lasso followed by OLS)
- Relaxed Lasso (Lasso followed by Lasso)
- Horseshoe and regularised horseshoe

The different methods are compared in a simulation study to evaluate how they fare for different datasets, in the presence of multicollinearity or low/high signal-to-noise (SNR) ratio.

Results are described in `January_2019_Tutorial_Presentation__Regularisation_.pdf`.
The code of the analysis is available in `main.R`.
The code for the regularised horseshoe model is available in `regularised_horseshoe.stan`.
