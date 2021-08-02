# 2D Parametric Fit Using Least Square

The program using the least square method to fit a 2D histogram with a 2D parametric curve, i.e. a curve (x(t), y(t)) defined by a single parameter t. The distances used for the residual is the shorted distance from each data point to the curve. The code therefore used a opimize.minimize (for finding the shortest distance) on top of anoother opimize.minimize (for finding the minimal residual sum of square); the fit requires significant processing power.

The fit function can be used by importing parametricFit2D.py. However, an example code runs on python3 with the following:
    pip3 install scipy
    pip3 install tqdm
    python3 parametricFit2D.py
The code outputs the following image:

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/paraFitCurve2D.png" width="600" height="900">


<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/progressPlot_Boot.png" width="600" height="450">
