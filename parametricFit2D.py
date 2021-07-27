import os, sys, pathlib, math
import numpy as np
import random as rd
import numdifftools as nd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import optimize
from scipy import linalg
from scipy.misc import derivative
from tqdm import tqdm
import pickle


SAVE_DIR=str(pathlib.Path().absolute())
######################################################################################################
#downSampling[i] = [replacible sampling size, maxiter, bounds, constraints]
#constraints only for optMethod = "COBYLA", "SLSQP", "trust-constr":
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
ERR_EPS=1e-12   #np.finfo(float).eps
def paraLeastSquare(parXYinit, funcXY, dataXY, dataRangeXY, paraRange=[0.0, 1.0],\
                    optMethod="Nelder-Mead", bounds=None, constraints=None, ratioHeadTail=0.01,\
                    verbosity=1, progressPlot=False, saveProgress=False, randSeed=None,\
                    downSampling="DEFAULT"): 
    #drop out-of-range data
    dataXInput = []
    dataYInput = []
    for x, y in np.array(dataXY).T:
        if (dataRangeXY[0][0] < x) and (x < dataRangeXY[0][1]) and\
           (dataRangeXY[1][0] < y) and (y < dataRangeXY[1][1]):
            dataXInput.append(x)
            dataYInput.append(y)
    dataN = len(dataXInput)
    parXN, parYN = len(parXYinit[0]), len(parXYinit[1])
    normXYRatio = [1.0/(dataRangeXY[0][1]-dataRangeXY[0][0]),\
                   1.0/(dataRangeXY[1][1]-dataRangeXY[1][0])]
    if len(dataXInput) != len(dataYInput):
        print("ERROR: paraLeastSquare: lengths of dataX and dataY don't match")
        sys.exit(0)   
 
    if (downSampling is None) or (len(downSampling) == 0):
        downSampling = [[np.inf, np.inf, bounds, constraints]]
    elif downSampling == "DEFAULT":
        downSampling=[*[[100,   1000, None, None]]*3,\
                      *[[1000,  1000, None, None]]*(parXN+parYN),\
                        [np.inf, 200, None, None]] 
    if verbosity >= 1:
        print("\n---------------------------------------------------------------Begin Parametric Fit")
    parXOpt, parYOpt = parXYinit[0].copy(), parXYinit[1].copy()
    #recover saved parameters for the next optimization
    downSamplingProgressN = -1
    pickleName = SAVE_DIR+"/zSavedProgress/savedProgress.pickle"
    iterErr2s = []
    if saveProgress == True:
        if os.path.isdir(SAVE_DIR) is False:
            print("ERROR: paraLeastSquare: the directory for SAVE_DIR does not exist:")
            print("   ", SAVE_DIR)
            sys.exit(0)
        try:
            progressDict = {}
            with open(pickleName, "rb") as handle:
                progressDict = pickle.load(handle)
            downSamplingProgressN = progressDict["downSamplingN"]
            #parXOpt = progressDict["parXOpt"]
            #parYOpt = progressDict["parYOpt"]
            parXOpt = progressDict["parX"]
            parYOpt = progressDict["parY"]
            iterErr2s = progressDict["iterErr2"]
        except OSError or FileNotFoundError:
            pathlib.Path(SAVE_DIR+"/zSavedProgress/").mkdir(exist_ok=True)
    #loop through downSampling
    rd.seed(randSeed)
    for s, sampStat in enumerate(downSampling):
        if s <= downSamplingProgressN:
            continue
        #down-sample data to speed up the inital progress
        dataXforOpt, dataYforOpt = dataXInput.copy(), dataYInput.copy()
        if dataN > sampStat[0]:
            sampledData = rd.choices(list(zip(dataXInput, dataYInput)), k=int(sampStat[0]))
            dataXforOpt = [d[0] for d in sampledData]
            dataYforOpt = [d[1] for d in sampledData]
        if len(dataXforOpt) < (parXN + parYN):
            print("ERROR: paraLeastSquare: the number of samples ("+str(len(dataXforOpt))+")"+\
                  "is fewer than the number of parameters("+str(parXN + parYN)+")")
            sys.exit(0)
        #main optimization
        iterErr2s.append([])
        res2Ave = lambda par : paraSquareResidualAve([par[:parXN], par[parXN:]], funcXY,\
                                                     [dataXforOpt, dataYforOpt],\
                                                     normXYRatio=normXYRatio, paraRange=paraRange,\
                                                     ratioHeadTail=ratioHeadTail,verbosity=verbosity,\
                                                     iterErr2=iterErr2s[s],downSamp=[s, sampStat[:2]])
        paraFitResult = optimize.minimize(res2Ave, [*parXOpt, *parYOpt], method=optMethod,\
                                          options={"maxiter":sampStat[1]},\
                                          bounds=sampStat[2], constraints=sampStat[3])
        parXOpt, parYOpt = paraFitResult.x[:parXN], paraFitResult.x[parXN:]
        if verbosity >= 4:
            print("Optimization Result:")
            print(paraFitResult)
        #error evaluation
        if verbosity >= 1:
            print("\nEvaluating Standard Errors:")
        iterErr2Err = []
        res2 = lambda par : len(dataXforOpt)\
                           *paraSquareResidualAve([par[:parXN], par[parXN:]], funcXY,\
                                                  [dataXforOpt, dataYforOpt],normXYRatio=normXYRatio,\
                                                  paraRange=paraRange, ratioHeadTail=0.0,\
                                                  verbosity=min(2,verbosity), iterErr2=iterErr2Err)
        errResult = optimize.minimize(res2, [*parXOpt, *parYOpt],method="BFGS",options={"maxiter":1},\
                                      bounds=sampStat[2], constraints=sampStat[3])
        hessInv = errResult.hess_inv
        sigma2 = errResult.fun/(len(dataXforOpt) - parXN - parYN)

        parXErr, parYErr = [-1 for _ in range(parXN)], [-1 for _ in range(parYN)]
        for i in range(parXN):
            parXErr[i] = math.sqrt(sigma2*errResult.hess_inv[i][i])
        for i in range(parYN):
            parYErr[i] = math.sqrt(sigma2*errResult.hess_inv[parXN+i][parXN+i]) 
        if verbosity >= 4:
            print("Error Evaluation Result:")
            print(errResult)
        #progress plot
        if progressPlot == True:
            progressPlot_paraLeastSquare([parXOpt, parYOpt], funcXY, [dataXforOpt, dataYforOpt],\
                                         dataRangeXY, iterErr2s=iterErr2s, downSamp=[s, sampStat[:2]])
        #save the progress
        if saveProgress == True:
            progressDict = {}
            progressDict["downSamplingN"] = s
            progressDict["parXOpt"] = parXOpt
            progressDict["parXErr"] = parXErr
            progressDict["parYOpt"] = parYOpt
            progressDict["parYErr"] = parYErr
            progressDict["iterErr2"] = [[iterErr2[-1]] for iterErr2 in iterErr2s[:-1]]
            progressDict["iterErr2"] += [iterErr2s[-1]]
            with open(pickleName, "wb") as handle:
                pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickleDSName = pickleName.replace(".pickle", "DS["+str(s)+"].pickle")
            with open(pickleDSName, "wb") as handle:
                pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity >= 1:
                print("Saving progress:")
                print("    downSamplingN = ", s)
                print("    parXOpt = ", [scientificStr_paraLeastSquare(par) for par in parXOpt])
                print("    parXerr = ", [scientificStr_paraLeastSquare(err) for err in parXErr])
                print("    parYOpt = ", [scientificStr_paraLeastSquare(par) for par in parYOpt])
                print("    parYErr = ", [scientificStr_paraLeastSquare(err) for err in parYErr])
                print("    iterErr2[-1] = ", iterErr2s[-1][-1])
                print("  with files:\n   ", pickleName, "\n   ", pickleDSName)
        if (verbosity >= 1) and ((len(downSampling) > 0)):
            print("----------------------------------------------downSampling["+str(s)+"] Complete\n")

   






#####testing 
    iterErr2Err = []
    dataXforOpt, dataYforOpt = dataXInput[:100], dataYInput[:100]
    res2 = lambda par : len(dataXforOpt)\
                       *paraSquareResidualAve([par[:parXN], par[parXN:]], funcXY,\
                                              [dataXforOpt, dataYforOpt],normXYRatio=normXYRatio,\
                                              paraRange=paraRange, ratioHeadTail=0.0,\
                                              verbosity=2, iterErr2=iterErr2Err)
    res20 = lambda par : len(dataXforOpt)\
                        *paraSquareResidualAve([[par, *parXOpt[1:]], parYOpt], funcXY,\
                                               [dataXforOpt, dataYforOpt],normXYRatio=normXYRatio,\
                                               paraRange=paraRange, ratioHeadTail=0.0,\
                                               verbosity=1, iterErr2=iterErr2Err)

 
    #print([parXOpt[0], *parXOpt[1:], *parYOpt])


    #def funcTest(x):
    #    return x*x
    #print(derivative(funcTest, 1, n=2))
    #print(optimize.approx_fprime(1, funcTest, [ERR_EPS]))
    #print([*parXOpt, *parYOpt])
    #print(ERR_EPS)
    #print(optimize.approx_fprime([*parXOpt, *parYOpt], res2, [1e-12]))
    #print(derivative(res20, parXOpt[0], dx=1e-12))
    #print(nd.Gradient(res2)([*parXOpt, *parYOpt]))
    #print(nd.Hessian(res2)([*parXOpt, *parYOpt]))
    #sys.exit(0)
#####




    if verbosity >= 1:
        print("-----------------------------------------------------------Parametric Fit Complete\n")
    return parXOpt, parYOpt
def paraSquareResidualAve(parXY, funcXY, dataXY, normXYRatio=[1.0, 1.0], paraRange=[0.0, 1.0],\
                          ratioHeadTail=0.0, verbosity=1, iterErr2=None, downSamp=None):
    if len(dataXY[0]) != len(dataXY[1]):
        print("ERROR: paraSquareErrorAve: lengths of dataX and dataY don't match")
        sys.exit(0)
    outputStr = ""
    if iterErr2 is not None:
        if len(iterErr2) == 0:
            iterErr2.append([0, None, None])
        else:
            iterErr2.append([iterErr2[-1][0]+1, None])
        outputStr += "opt iter "+str(iterErr2[-1][0])
    if downSamp is not None:
        if outputStr != "":
            outputStr += ", "
        outputStr += "downSampling["+str(downSamp[0])+"]=" +\
                     str([(int(ds) if (ds < np.inf) else np.inf) for ds in downSamp[1]])
    if verbosity >= 2:    
        print(outputStr)

    lambdaX = lambda t : funcXY[0](t, parXY[0])
    lambdaY = lambda t : funcXY[1](t, parXY[1])
    
    (res2Sum, opt_ts) = (0, [])
    for x, y in tqdm(np.array(dataXY).T, disable=(verbosity < 2)):
        distSquare = lambda t : paraSquareDist(t, [lambdaX, lambdaY], [x, y], normXYRatio=normXYRatio)
        opt_t = optimize.minimize_scalar(distSquare, method="bounded", bounds=tuple(paraRange))
        res2Sum += distSquare(opt_t.x)
        opt_ts.append(opt_t.x)

    err2HeadTail = 0
    if ratioHeadTail > 0:
        countHeadTail = max(1, int(len(dataXY[0])*ratioHeadTail))
        opt_ts = sorted(opt_ts)
        for i in range(countHeadTail):
            err2HeadTail += pow(lambdaX(opt_ts[i])   -lambdaX(paraRange[0]+ratioHeadTail), 2) 
            err2HeadTail += pow(lambdaY(opt_ts[i])   -lambdaY(paraRange[0]+ratioHeadTail), 2)
            err2HeadTail += pow(lambdaX(opt_ts[-i-1])-lambdaX(paraRange[1]-ratioHeadTail), 2)
            err2HeadTail += pow(lambdaY(opt_ts[-i-1])-lambdaY(paraRange[1]-ratioHeadTail), 2)

    res2Ave      = res2Sum/len(dataXY[0])
    err2HeadTail = err2HeadTail/len(dataXY[0])
    if iterErr2 is not None:
        iterErr2[-1] = [iterErr2[-1][0], res2Ave, err2HeadTail]

    if verbosity >= 3:
        print("                                                average normalized square residual =",\
              scientificStr_paraLeastSquare(res2Ave, 10))
        print("  sample size =", len(dataXY[0]))
        print("  parX =", [scientificStr_paraLeastSquare(par) for par in parXY[0]])
        print("  parY =", [scientificStr_paraLeastSquare(par) for par in parXY[1]])
        print("  [min_t, max_t] =", [scientificStr_paraLeastSquare(min(opt_ts)),\
                                     scientificStr_paraLeastSquare(max(opt_ts))])
        print("  head_tail normalized square error =", scientificStr_paraLeastSquare(err2HeadTail,10))
        print("")
    return res2Ave + err2HeadTail
def paraSquareDist(t, funcXY, dataXY, normXYRatio=[1.0, 1.0]):
    return pow(normXYRatio[0]*(funcXY[0](t) - dataXY[0]), 2) +\
           pow(normXYRatio[1]*(funcXY[1](t) - dataXY[1]), 2)
def progressPlot_paraLeastSquare(parXYFit, funcXY, dataXY, dataRangeXY,\
                                 verbosity=1, iterErr2s=None, downSamp=[-1, [-1, -1]]):
    pathlib.Path(SAVE_DIR+"/zSavedProgress/").mkdir(exist_ok=True)
    figName = SAVE_DIR+"/zSavedProgress/progressPlot.png"
    def truncateColorMap(cmap, lowR, highR):
        cmapNew = matplotlib.colors.LinearSegmentedColormap.from_list(\
            "trunc({n}, {l:.2f}, {h:.2f})".format(n=cmap.name, l=lowR, h=highR),\
            cmap(np.linspace(lowR, highR, 1000)))
        return cmapNew

    iterations, res2Aves = [], []
    totIter = 0
    for s, iterErr2 in enumerate(iterErr2s):
        totIter += iterErr2[-1][0]
        iterations.append(totIter)
        res2Aves.append(iterErr2[-1][1])

    binN = int(max(10, min(1000, 10*math.sqrt(downSamp[1][0]))))
    fitT = np.linspace(0.0, 1.0, binN+1)[:-1]
    fitFuncX = funcXY[0](fitT, parXYFit[0])
    fitFuncY = funcXY[1](fitT, parXYFit[1])

    fig = plt.figure(figsize=(12, 9))
    matplotlib.rc("xtick", labelsize=16)
    matplotlib.rc("ytick", labelsize=16)
    gs = gridspec.GridSpec(1, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols):
        ax.append(fig.add_subplot(gs[i])); 

    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.12, right=0.98)
    ax[0].plot(iterations, res2Aves, linewidth=2, color="blue", marker="o", markersize=6)
    ax[0].set_title("Normalized Residual Square Average at Each DownSampling", fontsize=24, y=1.03)
    ax[0].set_xlabel("iterations", fontsize=20)
    ax[0].set_ylabel("residual", fontsize=20)
    ax[0].set_xlim(left=0)
    plt.savefig(figName)

    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=1.0)
    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[0].hist2d(*dataXY, bins=binN, cmin=1, cmap=cmap, range=dataRangeXY)
    cb = fig.colorbar(hist[3], ax=ax[0]).mappable
    ax[0].plot(fitFuncX, fitFuncY, linewidth=3, color="red")
    plotTile =  "DownSampling[" + str(downSamp[0]) + "]="
    plotTile += str([(int(ds) if (ds < np.inf) else np.inf) for ds in downSamp[1]]) + ", "
    plotTile += "Iter=" + str(iterErr2s[-1][-1][0]) + ", NormSqErr="
    plotTile += scientificStr_paraLeastSquare(iterErr2s[-1][-1][1])
    ax[0].set_title(plotTile, fontsize=20, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_xlim(*dataRangeXY[0])
    ax[0].set_ylim(*dataRangeXY[1])

    figDSName = figName.replace(".png", "DS["+str(s)+"]"+".png") 
    plt.savefig(figDSName)
    if verbosity >= 1:
        print("Saving plots:\n   ", figName, "\n   ", figDSName)
def roundSig_paraLeastSquare(val, sigFig=3):
    if val == 0:
        return val;
    return round(val, sigFig-int(np.floor(np.log10(abs(val))))-1);
def scientificStr_paraLeastSquare(val, sigFig=3):
    valStr = ""
    if val == 0:
        valStr = "0.0"
    elif abs(np.floor(np.log10(abs(val)))) < sigFig:
        valStr = str(roundSig_paraLeastSquare(val, sigFig=sigFig))
    else:
        valStr = "{:." + str(sigFig-1) + "e}"
        valStr = valStr.format(val)
        valStr = valStr.replace("e+0", "e+")
        valStr = valStr.replace("e+", "e")
        valStr = valStr.replace("e0", "")
        valStr = valStr.replace("e-0", "e-")
    return valStr
















def exampleFunc_parametricFit2D(t):
    x = 2*math.sin(t + math.pi/5) + 0.5*t
    y = 1.2*math.cos(t + math.pi/5) + 0.8*math.sin(t + math.pi/5)
    return x, y
def example_parametricFit2D():
    def curve(t):
        x = 2*math.sin(t + math.pi/5) + 0.5*t
        y = 1.2*math.cos(t + math.pi/5) + 0.8*math.sin(t + math.pi/5)
        return x, y
    def polyFunc(x, coefs):
        if np.isscalar(coefs) == True:
            print("ERROR: polyFunc: coefs must be a 1D array/list")
            sys.exit(0)
        result = 0
        for i, c in enumerate(coefs):
            result += c*np.power(x, i)
        return result
    def truncateColorMap(cmap, lowR, highR):
        cmapNew = matplotlib.colors.LinearSegmentedColormap.from_list(\
                  "trunc({n}, {l:.2f}, {h:.2f})".format(n=cmap.name, l=lowR, h=highR),\
                  cmap(np.linspace(lowR, highR, 1000)))
        return cmapNew

    #sample setup
    binN    = 1000
    sampleN = 20000
    rd.seed(0)
    noiseSig = 0.2
    rangeXY = [[-1.5, 3.5], [-2.2, 2.2]]

    paraRange = [-math.pi/4, 3*math.pi/2]
    paraT = np.linspace(*paraRange, binN+1)[:-1]
    curveX = [curve(t)[0] for t in paraT]
    curveY = [curve(t)[1] for t in paraT]
    data = [[], []]
    for i in range(sampleN):
        x, y = curve(rd.uniform(*paraRange))
        x, y = rd.gauss(x, noiseSig), rd.gauss(y, noiseSig)
        data[0].append(x), data[1].append(y)

    #parametric fit
    #heavily depends on initial conditions, can test with downSampling=[*[[100, 1]]*1] first
    funcX = polyFunc
    funcY = polyFunc
    initX = [1.0, -15.0, 43.0, -10.0, -20.0, 0.0, 0.0, 0.0]
    initY = [0.0, -8.0,  12.0, -4.0,   1.0,  0.0, 0.0, 0.0]
    optMethod = "BFGS"#"Nelder-Mead"       #"BFGS"

    downSampling = [*[[1000, 1, None, None]]*5]
    #noBnd = (None, None)
    #bounds = ((0.9, 1.1),  noBnd, noBnd, noBnd, noBnd,\
    #          (-0.1, 0.1), noBnd, noBnd, noBnd, noBnd)
    #downSampling = [[100, 1000, bounds, None]]
    #downSampling = [*[[100,   1000, None, None]]*3,\
    #                *[[1000,  1000, None, None]]*14,\
    #                  [np.inf, 200, None, None]]
    
    saveBool=True
    parXFit, parYFit = paraLeastSquare([initX, initY], [funcX, funcY], data, rangeXY,\
                                       optMethod=optMethod, ratioHeadTail=0.01,\
                                       verbosity=4, progressPlot=saveBool, saveProgress=saveBool,\
                                       randSeed=0)#, downSampling=downSampling)

    fitT = np.linspace(0.0, 1.0, binN+1)[:-1]
    fitFuncX = funcX(fitT, parXFit)
    fitFuncY = funcY(fitT, parYFit)
    print("parXOpt =", [scientificStr_paraLeastSquare(par) for par in parXFit])
    print("parYOpt =", [scientificStr_paraLeastSquare(par) for par in parYFit])
    #plot
    fig = plt.figure(figsize=(12, 18))
    matplotlib.rc("xtick", labelsize=16)
    matplotlib.rc("ytick", labelsize=16)
    gs = gridspec.GridSpec(2, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols):
        ax.append(fig.add_subplot(gs[i]));

    ax[0].plot(curveX, curveY, linewidth=3, color="blue")
    ax[0].set_title("Given Parametric Curve", fontsize=28, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(rangeXY[0][0], rangeXY[0][1]+1.18)
    ax[0].set_ylim(*rangeXY[1])

    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[1].hist2d(*data, bins=int(binN/10.0), cmin=1, cmap=cmap, range=rangeXY)
    cb = fig.colorbar(hist[3], ax=ax[1]).mappable
    ax[1].plot(fitFuncX, fitFuncY, linewidth=3, color="red") 
    ax[1].set_title("Parametric Fitting the Curve", fontsize=28, y=1.03)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_aspect("equal")
    ax[1].set_xlim(*rangeXY[0])
    ax[1].set_ylim(*rangeXY[1])

    figName = "paraFitCurve2D.png"
    gs.tight_layout(fig)
    plt.savefig(figName)
    print("Saving the following file:\n   ", figName)


if __name__ == "__main__":
    print("##################################################################################Begin\n")
    example_parametricFit2D()
    print("\n##################################################################################End\n")



