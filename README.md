# 2D Parametric Fit Using Least Square

The program uses the least square method to fit a 2D histogram with a 2D parametric curve, i.e. a curve (x(t), y(t)) defined by a single parameter t. The distances used for the residual is the shorted distance from each data point to the curve. For this reason, the code requires applying an opimize.minimize (for finding the shortest distance) on top of another opimize.minimize (for finding the minimal residual sum of square); the fit requires significant processing power.

The fit function can be used by importing parametricFit2D.py. However, an example code runs on python3 with the following:

    pip3 install scipy
    pip3 install tqdm
    python3 parametricFit2D.py
The code outputs the following images:

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/paraFitCurve2D_Display.png" width="600" height="1350">

- Top: the 20,000 data points comes from the gaussian broaden given curve (blue curve on the bottom plot). The red curve is then fitted using a 6th order polynomial on both x(t) and y(t).
- Middle: the given curve (blue) versus 100 fitted curves (red) sampled from the fit parameters and their standard errors.
- Bottom: similar to the middle plot, but the 100 fitted curves (red) are sampled from the fit parameters and their covariant matrix.
- Note: by default, the range of t is [-1, 1]. Having a 0 at the range boundary would make polynomial fit depend too much on the constant parameter.
- Note: the success of the fit depends greatly on the initial values given to the parameters. A mathamtica solver systemOfLinearEquation.nb is provided to find these values more easily, whose required inputs are the approximate points on the curve. These approximate points are best taken uniformly across the curve. 
- Note: however that the paramatric fit does NOT give a constant speed curve, which would be unnecessarily difficult for polynomials.

Other then the plot from the example code, the main code also output progress plots and save the progress in .pickle file such that the fit can be stopped and continued:

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/progressPlot_Boot_Display.png" width="600" height="450">

- The plot shows the "normalized averaged residual sum of square" as the optimization iteration increases. The smallest value is labeled.
- Note: the distance residual is normalized by the given range of x and y to give their distances comparative weights
- Note: since each of the data points do not come with an error bar, the (normalized averaged residual sum of square) do not necessarily follow the &chi;-square distribution, so can only accounted for a relative goodness-of-fit.
- Note: to increase the fit speed, each data point is actually "down-sampled" to 1000 data points from the original 20,000 data with replacement. Each "down-sampling" point shows as a blue dot in the plot.
- Note: the orange points are the bootstrapped "down-sampling" to get the standard error for the parameters. As shown in the plot, these bootstrap samples are collected when the "normalized averaged residual sum of square" reaches more or less the equilibium. On critical points to point out is that these bootstrap standard error does NOT take into account of how good your models are in fitting the curve. The code also has an option to evaluate the standard error based on the inverse of the Hessian function. Nonetheless, the Hessian can only be numerically calculated and the resulting standard errors are not stable.

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/paraRangeLocalMinDegen_Display" width="600" height="450">

Future tasks: the bounds and constraints don't see to work for the general optimization method for scipy.optimize. Using language multiplier may be a solution.

References:
- stats.stackexchange question 285023 (<a href="https://stats.stackexchange.com/questions/285023/compute-standard-errors-of-nonlinear-regression-parameters-with-maximum-likeliho">StackExchange</a>)
