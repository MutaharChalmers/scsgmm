# scsgmm
Sequential Conditional Sampling from Gaussian Mixed Models (SCS-GMM) is a method to fit simple parametric models to univariate and multivariate time series and generate synthetic realisations. The method was inspired by Sharma et al (1997) Streamflow simulation: A nonparametric approach [https://doi.org/10.1029/96WR02839], and originally implemented using KDEs in the package `scskde`. The next logical step was to try an analogous parametric approach, the simplest of which is to replace the KDEs with GMMs. The approach supports
- lag orders >= 1
- arbitrary seasonality
- vector-valued processes
- exogenous forcing
- arbitrary dependence structure
