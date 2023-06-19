# npchoice
An open-source Python package for nonparametric discrete choice estimation. 

Key features:
* Nonparametric estimation of panel discrete choice models
    - Employs latent-class approximation 
* Powered by [JAX](jax.readthedocs.io), Google's GPU-accelerated scientific computing suite

Future additions:
* Fixed-grid approximation, both with (conditional) logit and (multinomial) probit link functions
    - To the best of my knowledge, this will be the first nonparametric implementation of multinomial probit
    - In the case of cross-sectional data, <i>npchoice</i> will support three estimators:
        * Elastic net, à la [Heiss, Hetzenecker, and Osterhaus (2022, JoE)](https://doi.org/10.1016/j.jeconom.2020.11.010)
        * Lasso, à la [Fox, Kim, Ryan and Bajari (2011, QE)](https://doi.org/10.3982/QE49)
        * EM algorithm, à la [Train (2008, JOCM)](https://doi.org/10.1016/S1755-5345(13)70022-8)
* Hyperparameter tuning utilities 
    - Computation of Aikake Information Criterion and Bayesian Information Criterion
    - Automated cross validation
* Automated data pre-processing 
    - At present, code assumes data are pre-sorted by agent 
* Systematic user documentation

My code for the LBFGS algorithm is based on code from the open-source Python package <i>xlogit</i>:

> Arteaga, C., Park, J., Beeramoole, P. B., & Paz, A. (2022). xlogit: An open-source Python package for GPU-accelerated estimation of Mixed Logit models. Journal of Choice Modelling, 42, 100339. https://doi.org/10.1016/j.jocm.2021.100339
