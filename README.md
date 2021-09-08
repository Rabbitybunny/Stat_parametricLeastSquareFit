# 2D Parametric Fit Using Least Square

| Function  | Output | Description |
| - | - | - |
| `paraLeastSquare` | `parXYOpt`:[list,list], `parXYErr`:[list,list,matrix], `res2AveVal`:float | least square fit to a 2D-histogram with a parametric function defined on both of the [x-axis, y-axis]. The `parXYOpt` is the optimal results of the fit parameter on the [x-axis, y-axis]. The `parXYErr` is the standard errors of the  optimized fit parameters, while the matrix is the covariant matrix (standard errors with correlations). The `res2AveVal` is the average residual square that is used for the least square optimization |
| `getDataPtOfPara` | `ptXY`:[float,float] | get the point `ptXY` on the [x-axis, y-axis] on the parametric curve given a parametric value |
| `getParaOfDataPt` | `paraVal`:float | get the parametric value `paraVal` that correspond to a point on the parametric curve that is closest to a given data point |
| `getInstantPlotDownSampling` | `downSampling` | generate a `downSampling` parameter for `paraLeastSquare` that halt the optimization but give the latest optimization results of the optimization process. See `example_parametricFit2D` for the usage |
| `printSavedProgress` | | print out the keys and values of a pickle file saved by `paraLeastSquare` |
| `example_parametricFit2D` | | example code that runs the `paraLeastSquare`. It produces all the figures on this page |

## Function parameters:

| Function  | parameter | Description |
| - | - | - |
| `paraLeastSquare`  | `parXYinit`: [list,list] | initial parameter values for the fit functions on the [x-axis, y-axis] |
| | `funcXY`: [def,def] | fit functions in format of `def func(t, par_list):` on the <br/> [x-axis, y-axis] |
| | `dataXY`: [list,list] | input data on the [x-axis, y-axis]; each list has a length of N samples and the order of the two list must match |
| | `dataRangeXY`: [list,list] | each 2D-list is a range placed on the [x-axis, y-axis]. Data pair outside the range will be ignored. The ranges are also used to "normalize" residual square in case data have values on the two axis differ by several orders of magnitude. The ranges subtling influence the optimization result |
| | `optMethod`<br/>`="Nelder-Mead"` | fit method for `scipy.optimize.minimize` used in the code |
| | `paraRange=[-1.0,1.0]` | range of the parametric variable. If the length of the list is greater than 2, then partitions would be made to break local minimum degeneracy (see the decription of the 3rd figure) |
| | `ratioHeadTail`<br/>`=[0.01,0.01]` | adding weights to the the residual square to maintain `paraRange[0]` being closed to the head of the curve and `paraRange[-1]` to the tail |
| | `randSeed=None` | random seed used for bootstraping |
| | `downSampling`<br/>`="DEFAULT"` | the parameter is a list of lists that have the following format: <br/>[operation-type, sampling-size, maxiter, bounds, constraints]. There are 3 operation-types, "Opt" for optimization, "Boot" for bootstraping, and "Hess" for inverse Hessian error. The sampling-size refined to resampling from the existing data with replacement. The maxiter is the maximum iteration as am input for `scipy.opimize.minimize`. The bounds and constraints are also inputs for `scipy.opimize.minimize`, but they work only for certain optimization methods (<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html">Scipy doc</a>). The default value is `downSampling = [`<br/> `["Opt",  1000,   0, None, None],`<br/>`*[["Opt", 1000, 1000, None, None]]*2`,<br/>`["Opt", np.inf, np.inf, None, None]`,<br/>`*[["Boot", np.inf, 200, None, None]]*30]`,<br/> where the first line is for plotting out the curve using the initial parameters, the second line is for quickly converging to near optimimum results, the third line is the main optimization, and the forth line is for bootstraping to get the standard error (and covariant matrx) for the fit parameters|
| | `verbosity=3` | verbosity for the output message, up to 4 |
| | `progressPlot=False` | True to save plots in `savePath` at each downSampling iteration |
| | `saveProgress=False` | True to save pickle files in `savePath` at each downSampling iteration. Each pickle file contains a dictionary with keys:<br/>`downSampling, parXYinit, funcXY, dataRangeXY, paraRange, optMethod, ratioHeadTail, randseed, downSamplingIterN, optimizationN, bootstrapN, optIdx, bootIdx, parXOpt, parXErr, parXBoot, parXBootErr, parXHessErr, parYOpt, parYErr, parYBoot, parYBootErr, parYHessErr, parErrCovMatrix, parBootErrCovMatrix, parHessErrCovMatrix, iterErr2`|
| | `readProgress=True` | True to read from pickle files in `savePath` (if exist) to continue the optimization |
| | `savePath="."` | default is `str(pathlib.Path().absolute()))` to define the directory where the files are saved to. The name of the save directory is defined by the variable `_PARAMETRICFIT2D_SAVEDIRNAME` in the code |
| `getDataPtOfPara` | `parXY`: [list,list] | fixed parameter values for the fit functions on the [x-axis, y-axis] |
| | `funcXY`: [def,def] | fit functions in format of `def func(t, par_list):` on the <br/> [x-axis, y-axis] |
| | `paraVal`: float | fixed value for the parametric variable |
| `getParaOfDataPt` | 'parXY': [list,list] | fixed parameter values for the fit functions on the [x-axis, y-axis] |
| | 'funcXY': [def,def] | fit functions in format of `def func(t, par_list):` on the <br/> [x-axis, y-axis] |
| | 'dataPtXY': [float,float] | data point value on the [x-axis, y-axis] |
| | 'dataRangeXY': [list,list] | each 2D-list is a range placed on the [x-axis, y-axis]. The ranges are also used to "normalize" residual square in case data have values on the two axis differ by several orders of magnitude. The ranges subtling influence the optimization result |
| | `paraRange=[-1.0,1.0]` | 2D-list as the range of the parametric variable |
| | 'iterCounter=[]' | pass-by-reference list to record the number of optimization iteration |
| `getInstantPlotDownSampling` | 'downSampling=None' | downSampling parameter readied for `paraLeastSquare` |
| | `savePath="."` | default is `str(pathlib.Path().absolute()))` to define the directory where the files are saved to. The name of the save directory is defined by the variable `_PARAMETRICFIT2D_SAVEDIRNAME` in the code |
| `printSavedProgress` | 'fullPicklePath' | path for the pickle file saved by `paraLeastSquare` to be print out by this function |
| | 'verbosity=2' | verbosity for the output message, up to 2 |

## Example:

The program uses the least square method to fit a 2D histogram with a 2D parametric curve, i.e. a curve (x(t), y(t)) defined by a single parameter t. The distances used for the residual is the shorted distance from each data point to the curve. For this reason, the code requires applying a `scipy.opimize.minimize` (for finding the shortest distance) on top of another `scipy.opimize.minimize` (for finding the minimal residual sum of square); the fit requires significant processing power.

The fit function can be used by importing `parametricFit2D.py`. However, an example code runs on python3 with the following:

    pip3 install scipy
    pip3 install tqdm
    python3 parametricFit2D.py
The code outputs the following images:

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/paraFitCurve2D_Display.png" width="600" height="1350">

- Top: the 20,000 data points comes from the gaussian broaden given curve (blue curve on the bottom plot). The red curve is then fitted using a 6th order polynomial on both x(t) and y(t).
- Middle: the given curve (blue) versus 100 fitted curves (red) sampled from the fit parameters and their standard errors.
- Bottom: similar to the middle plot, but the 100 fitted curves (red) are sampled from the fit parameters and their covariant matrix.
- Note: by default, the range of t is [-1, 1]. Having a 0 at the range boundary would make polynomial fit depend too much on the constant parameter.
- Note: the success of the fit depends greatly on the initial values given to the parameters. A mathamtica solver `systemOfLinearEquation.nb` is provided to find these values more easily, whose required inputs are the approximate points on the curve. These approximate points are best taken uniformly across the curve. 
- Note: however that the paramatric fit does NOT give a constant speed curve, which would be unnecessarily difficult for polynomials.

------------------------------------------------------------------

Other then the plot from the example code, the main code also output progress plots and save the progress in .pickle file such that the fit can be stopped and continued:

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/progressPlot_Boot_Display.png" width="600" height="450">

- The plot shows the "normalized averaged residual sum of square" as the optimization iteration increases. The smallest value is labeled.
- Note: the distance residual is normalized by the given range of x and y to give their distances comparative weights
- Note: since each of the data points do not come with an error bar, the (normalized averaged residual sum of square) do not necessarily follow the &chi;-square distribution, so can only accounted for a relative goodness-of-fit.
- Note: to increase the fit speed, each data point is actually "down-sampled" to 1000 data points from the original 20,000 data with replacement. Each "down-sampling" point shows as a blue dot in the plot.
- Note: the orange points are the bootstrapped "down-sampling" to get the standard error for the parameters. As shown in the plot, these bootstrap samples are collected when the "normalized averaged residual sum of square" reaches more or less the equilibium. On critical points to point out is that these bootstrap standard error does NOT take into account of how good your models are in fitting the curve. The code also has an option to evaluate the standard error based on the inverse of the Hessian function. Nonetheless, the Hessian can only be numerically calculated and the resulting standard errors are not stable.

------------------------------------------------------------------

Moreover, the code could output the following plot if local minima are spotted when applying opimize.minimize to find the shortest distance to the curve for some point:

<img src="https://github.com/Rabbitybunny/Stat_parametricLeastSquareFit/blob/main/paraRangeLocalMinDegen_Display.png" width="600" height="450">

The idea is that for points around (x,y)=(1, -1) of the example curve, the shortest distance can be on either to the "left-hand-side" or the "right-hand-side" of the point, which corresponds to the paramatric variable around -1 (blue data in the plot) and -0.3 (red data in the plot). The `opimize.minimize` cannot handle this degeneracy so easily and a global optimization is expansive and not always reliable to do on some many sample data points. So the solution in this code is to modify the `paraRange` parameter from [-1, 1] to [-1, -0.6, 1], and the code would then evaluate a minimum distance in both [-1, -0.6] and [-0.6, 1] partition and compare the results to determine the global minimal distance. For more complicated curve, finner partitions can also be placed in the similar regard.

Future tasks: the bounds and constraints don't see to work for the general optimization method for scipy.optimize. Using language multiplier may be a solution.

## References:
- K.K. Gan, Ohio State University, lecture on least chi-square fit (2004) (<a href="https://www.asc.ohio-state.edu/gan.1/teaching/spring04/Chapter6.pdf">PPT</a>)
- N. Börlin, Umeå University, lecture on nonlinear optimization (2007) (<a href="https://stats.stackexchange.com/questions/285023/compute-standard-errors-of-nonlinear-regression-parameters-with-maximum-likeliho">StackExchange</a>, <a href="https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf">PPT</a>)
- stats.stackexchange: bootstrap standard error (<a href="https://stats.stackexchange.com/questions/272098/bootstrap-estimate-for-standard-error-in-linear-regression">StackExchange</a>)
