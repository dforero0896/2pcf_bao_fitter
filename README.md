# 2pcf BAO fitter

A class to manage the BAO fit to 2pcf of galaxies and DT voids (with templates, the support for fitting with functions of k instead of templates may be added later). 
Pk are extrapolated by default in log-space which shows good results. 
The code uses two backends for sampling the posteriors: `zeus` and `pymultinest`. Here some quich timings:
+  Zeus backend: 4m52s 500 steps
+ Zeus backend: 17m31s 2000 steps
+ Multinest backend 3m22s

The marginal posteriors can be fount in the `output` and `test` directories for `pymultinest` and `zeus` respectively. In short `Multinest` is much faster and results in tighter bounds.

