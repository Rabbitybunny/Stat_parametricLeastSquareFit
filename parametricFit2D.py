import os, sys, math
import numpy as np
import random as rd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import optimize
from tqdm import tqdm
import pickle

#####################################################################################################
#downSampling[i] = [replacible sampling size, maxiter, bounds, constraints]
#constraints only for optMethod = "COBYLA", "SLSQP", "trust-constr":
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
def paraLeastSquare(parXYinit, funcXY, dataXY, dataRangeXY, paraRange=[0.0, 1.0],\
                    optMethod="Nelder-Mead", bounds=None, constraints=None, ratioHeadTail=0.01,\
                    randSeed=None, iterRef=[], progressPlot=False, saveProgress=False,\
                    downSampling=[*[[100,  1000, None, None]]*3,\
                                  *[[1000, 1000, None, None]]*5,\
                                  [  1e12, 200,  None, None]]): 
    #drop out-of-range data
    dataXInput = []
    dataYInput = []
    for x, y in np.array(dataXY).T:
        if (dataRangeXY[0][0] < x) and (x < dataRangeXY[0][1]) and\
           (dataRangeXY[1][0] < y) and (y < dataRangeXY[1][1]):
            dataXInput.append(x)
            dataYInput.append(y)
    if len(dataXInput) != len(dataYInput):
        print("ERROR: paraLeastSquare: lengths of dataX and dataY don't match")
        sys.exit(0)
    dataN = len(dataXInput)
    parXN = len(parXYinit[0])

    normXYRatio = [1.0/(dataRangeXY[0][1]-dataRangeXY[0][0]),\
                   1.0/(dataRangeXY[1][1]-dataRangeXY[1][0])]
    parXOpt, parYOpt = None, None
    if len(downSampling) != 0:
        print("\n--------------------------------------------------------------Begin Parametric Fit")
        #recover saved parameters for the next optimization
        pickleName = "zSavedProgress.pickle"
        downSamplingProgressN = -1
        parXforOpt, parYforOpt = parXYinit[0].copy(), parXYinit[1].copy()
        if saveProgress == True:
            try:
                progressDict = {}
                with open(pickleName, "rb") as handle:
                    progressDict = pickle.load(handle)
                downSamplingProgressN = progressDict["downSamplingN"]
                parXforOpt = progressDict["parX"]
                parYforOpt = progressDict["parY"]
            except OSError or FileNotFoundError:
                print("Saving the following file:\n   ", pickleName)
        #loop through downSampling
        rd.seed(randSeed)
        for s, sampStat in enumerate(downSampling):
            if s <= downSamplingProgressN:
                continue
            #down-sample data to speed up the inital progress
            dataNRatio = 1.0
            dataXforOpt, dataYforOpt = dataXInput.copy(), dataYInput.copy()
            if dataN > sampStat[0]:
                sampledData = rd.choices(list(zip(dataXInput, dataYInput)), k=int(sampStat[0]))
                dataXforOpt = [d[0] for d in sampledData]
                dataYforOpt = [d[1] for d in sampledData]
                dataNRatio = dataN/sampStat[0]
            #main optimization
            err2Sum = lambda par : dataNRatio*paraErrorSquareSum([par[:parXN], par[parXN:]], funcXY,\
                                                                 [dataXforOpt, dataYforOpt],\
                                                                 normXYRatio=normXYRatio,\
                                                                 ratioHeadTail=ratioHeadTail,\
                                                                 iterRef=iterRef,\
                                                                 downSamp=[s, sampStat[:2]],\
                                                                 dataNRatioDisp=dataNRatio)
            paraFitResult = optimize.minimize(err2Sum, [*parXforOpt, *parYforOpt], method=optMethod,\
                                              options={"maxiter":sampStat[1]},\
                                              bounds=sampStat[2], constraints=sampStat[3])
            parXforOpt, parYforOpt = paraFitResult.x[:parXN], paraFitResult.x[parXN:]
            #progress plot
            if progressPlot == True:
                progressPlot_paraErrorSquareSum([parXforOpt, parYforOpt], funcXY,\
                                                [dataXforOpt, dataYforOpt], dataRangeXY, err2Sum,\
                                                iterRef=iterRef, downSamp=[s, sampStat[:2]])
            #save the progress
            if saveProgress == True:
                progressDict = {}
                progressDict["downSamplingN"] = s
                progressDict["parX"] = parXforOpt
                progressDict["parY"] = parYforOpt
                with open(pickleName, "wb") as handle:
                    pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Saving the progress to:\n   ", pickleName, "\n   ", progressDict)
        parXOpt, parYOpt = parXforOpt, parYforOpt
        print("-----------------------------------------------------------Parametric Fit Complete\n")
    else:
        #main optimization without downSampling
        err2Sum = lambda par : paraErrorSquareSum([par[:parXN], par[parXN:]], funcXY,\
                                                  [dataXInput, dataYInput], normXYRatio=normXYRatio,\
                                                  ratioHeadTail=ratioHeadTail, iterRef=iterRef)
        paraFitResult = optimize.minimize(err2Sum, [*parXYinit[0], *parXYinit[1]],\
                                          method=optMethod, bounds=bounds, constraints=constraints)
        parXOpt, parYOpt = paraFitResult.x[:parXN], paraFitResult.x[parXN:]
    return parXOpt, parYOpt
def paraDistSquare(t, funcXY, dataXY, normXYRatio=[1.0, 1.0]):
    curveX, curveY = funcXY[0](t), funcXY[1](t)
    return pow(normXYRatio[0]*(curveX - dataXY[0]), 2) +\
           pow(normXYRatio[1]*(curveY - dataXY[1]), 2)
def paraErrorSquareSum(parXY, funcXY, dataXY, normXYRatio=[1.0, 1.0], paraRange=[0.0, 1.0],\
                       ratioHeadTail=0.0, iterRef=[], downSamp=[], dataNRatioDisp=1.0):
    lambdaX = lambda t : funcXY[0](t, parXY[0])
    lambdaY = lambda t : funcXY[1](t, parXY[1])

    if len(dataXY[0]) != len(dataXY[1]):
        print("ERROR: paraErrorSquareSum: lengths of dataX and dataY don't match")
        sys.exit(0)

    (err2Sum, opt_ts) = (0, [])
    for x, y in tqdm(np.array(dataXY).T, disable=(len(iterRef) == 0)):
        distSquare = lambda t : math.sqrt(paraDistSquare(t, [lambdaX, lambdaY], [x, y],\
                                                         normXYRatio=normXYRatio))
        opt_t = optimize.minimize_scalar(distSquare, method="bounded", bounds=tuple(paraRange))
        err2Sum += distSquare(opt_t.x)
        opt_ts.append(opt_t.x)

    err2HeadTail = 0
    if ratioHeadTail != 0:
        countHeadTail = max(1, int(len(dataXY[0])*ratioHeadTail))
        opt_ts = sorted(opt_ts)
        for i in range(countHeadTail):
            err2HeadTail += pow(lambdaX(opt_ts[i])   -lambdaX(paraRange[0]+ratioHeadTail), 2) 
            err2HeadTail += pow(lambdaY(opt_ts[i])   -lambdaY(paraRange[0]+ratioHeadTail), 2)
            err2HeadTail += pow(lambdaX(opt_ts[-i-1])-lambdaX(paraRange[1]-ratioHeadTail), 2)
            err2HeadTail += pow(lambdaY(opt_ts[-i-1])-lambdaY(paraRange[1]-ratioHeadTail), 2)

    if len(iterRef) != 0:
        iterRef[0] += 1
        print("opt iter "+str(iterRef[0]))
        if len(downSamp) != 0:
            print("downSampling["+str(downSamp[0])+"]="+str(downSamp[1]))
        print("                                                                norm square error =",\
              scientificStr_paraErrorSquareSum(err2Sum*dataNRatioDisp))
        print("  parX =", [scientificStr_paraErrorSquareSum(par) for par in parXY[0]])
        print("  parY =", [scientificStr_paraErrorSquareSum(par) for par in parXY[1]])
        print("  data number =", len(dataXY[0]))
        print("  [min_t, max_t] =", [scientificStr_paraErrorSquareSum(min(opt_ts)),\
                                     scientificStr_paraErrorSquareSum(max(opt_ts))])
        print("  norm head_tail square error =", scientificStr_paraErrorSquareSum(err2HeadTail))
    return err2Sum + err2HeadTail
def progressPlot_paraErrorSquareSum(parXYFit, funcXY, dataXY, dataRangeXY, err2SumFunc,\
                                    iterRef=[], downSamp=[-1, [-1, -1]]):
    def truncateColorMap(cmap, lowR, highR):
        cmapNew = matplotlib.colors.LinearSegmentedColormap.from_list(\
              "trunc({n}, {l:.2f}, {h:.2f})".format(n=cmap.name, l=lowR, h=highR),\
              cmap(np.linspace(lowR, highR, 1000)))
        return cmapNew
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

    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[0].hist2d(*dataXY, bins=binN, cmin=1, cmap=cmap, range=dataRangeXY)
    cb = fig.colorbar(hist[3], ax=ax[0]).mappable
    ax[0].plot(fitFuncX, fitFuncY, linewidth=3, color="red")
    plotTile =  "DownSampling[" + str(downSamp[0]) + "]=" + str(downSamp[1]) + ", "
    plotTile += "Iter=" + str(iterRef[0]) + ", NormSqErr="
    plotTile += scientificStr_paraErrorSquareSum(err2SumFunc([*parXYFit[0], *parXYFit[1]]))
    ax[0].set_title(plotTile, fontsize=20, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_xlim(*dataRangeXY[0])
    ax[0].set_ylim(*dataRangeXY[1])

    figName = "zProgressPlot" + str(downSamp[0]) + ".png"
    gs.tight_layout(fig)
    plt.savefig(figName)
    print("Saving the following file:\n   ", figName)
def roundSig_paraErrorSquareSum(val, sigFig=3):
    if val == 0:
        return val;
    return round(val, sigFig-int(np.floor(np.log10(abs(val))))-1);
def scientificStr_paraErrorSquareSum(val, sigFig=3):
    valStr = ""
    if val == 0:
        valStr = "0.0"
    elif abs(np.floor(np.log10(abs(val)))) < sigFig:
        valStr = str(roundSig_paraErrorSquareSum(val, sigFig=sigFig))
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
    initX = [1.0, -15.0, 43.0, -10.0, -20.0]
    initY = [0.0, -8.0,  12.0, -4.0,   1.0]
    optMethod = "Nelder-Mead"

    noBnd = (None, None)
    bounds = ((0.9, 1.1),  noBnd, noBnd, noBnd, noBnd,\
              (-0.1, 0.1), noBnd, noBnd, noBnd, noBnd)
    downSampling = [[100, 1000, bounds, None]]
    #downSampling = [*[[200, 20, None, None]]*10]
    #downSampling = [*[[1000, 1000, None, None]]*10, [1e6, 1000, None, None]]
    
    saveProg=False
    iterRef = [0]
    parXFit, parYFit = paraLeastSquare([initX, initY], [funcX, funcY], data, rangeXY,\
                                       optMethod=optMethod, ratioHeadTail=0.01,\
                                       iterRef=iterRef, progressPlot=True, saveProgress=saveProg,\
                                       downSampling=downSampling)
    fitT = np.linspace(0.0, 1.0, binN+1)[:-1]
    fitFuncX = funcX(fitT, parXFit)
    fitFuncY = funcY(fitT, parYFit)
    print("parXFit =", [scientificStr_paraErrorSquareSum(par) for par in parXFit])
    print("parYFit =", [scientificStr_paraErrorSquareSum(par) for par in parYFit])

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
    print("#################################################################################Begin\n")
    example_parametricFit2D()
    print("\n#################################################################################End\n")



