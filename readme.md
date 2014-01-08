 # It is all in the noise: Efficient multi-task Gaussian process inference with structured residuals

Multi-task prediction methods are widely used to couple regressors by sharing information across related tasks. We propose a multi-task Gaussian process approach for modeling both the relatedness between regressors and the task correlations in the resuiduals, in order to more accuraetly identify true sharing between regressors.
The resulting Gaussian model has a covariance term in form of a sum of Kronecker products, for which efficient parameter inference and out of sample predict are feasible.

Please see the following paper for more informations:

* Barbara Rakitsch, Christoph Lippert, Karsten Borgwardt, Oliver Stegle: It is all in the noise: **Efficient multi-task Gaussian process inference with structured residuals**, *Advances in Neural Information Processing Systems 26(NIPS 2013)*,1466--1474.

The implementation is in Python and builds on the Gaussian process toolbox in pygp ([link](https://github.com/PMBio/pygp)). 

The code is organized as follows:

* core: efficient implementation of Gaussian processes
* experiments: code for re-running the NIPS experiments on simulated data
* demo: sample code to get started

For running the demo, go to the folder pygp_kronsum/demo and type in:
python small_demo.py

For running the NIPS simulation experiments, go to the folder pygp_kronsum/experiments and do the following:

* python generate_data.py (data is simulated)
* python simulations.py common (common effect is varying)
* python simulations.py causal (signal strength is varying)
* python simulations.py hidden (hidden signal strength is varying)


For running the NIPS runtime experiments, go to the folder pygp_kronsum/experiments
and do the following:
* python generate_data.py
* python runtime.py n (n is the number of samples and we used the values 16,32,64,128,256)

For plotting the results, use the following commands:
* python plot_runtime.py 
* python plot_simulations.py 


Please contact me, [Barbara Rakitsch](barbara.rakitsch@tuebingen.mpg.de), if you have any further questions.
