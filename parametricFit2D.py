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
#downSampling[i] = [operation type ("Opt": optimization, 
#                                   "Boot": error from bootstrap, recommending 30 iterations,
#                                   "Hess": error from inverse Hessian using numdifftools),
#                   sampling size with replacement, 
#                   maxiter for least square optimization,
#                   bounds, constraints]
#bounds/constraints only for optMethod = "COBYLA", "SLSQP", "trust-constr":
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#
#parXYinit     = [[initial parameter value for x-axis], [initial parameter value for y-axis]]
#funcXY        = [parametric fit function for x-axis,   parametric fit function for y-axis]
#dataXY        = [[data values for x-axis],             [data values for y-axis]]
#dataRangeXY   = [data range for x-axis,                data range for y-axis]
#   Data that fall outside the range will be removed
#   The range goes into normalizing the residual square
#optMethod     = optimization method for the least square (minimizing the the residual square)
#paraRange     = range allowed for the parametric variable t, whose fit function is (x(t), y(t))
#ratioHeadTail = adding weight to the the residual square to maintain paraRange
#randSeed      = random seed for downSampling
#downSampling  = (see above)
#verbosity:    controlling the amount of output messages, up to 4
#progressPlot: saving plots in SAVE_DIR at each downSampling iteration
#saveProgress: saving pickle files in SAVE_DIR at each downSampling iteration
#readProgress: reading from pickle files (if exist) to continue the optimization
def paraLeastSquare(parXYinit, funcXY, dataXY, dataRangeXY, optMethod="Nelder-Mead",\
                    paraRange=[0.0, 1.0], ratioHeadTail=0.01, randSeed=None, downSampling="DEFAULT",\
                    verbosity=3, progressPlot=False, saveProgress=False, readProgress=True): 
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
        downSampling = [["Opt",  np.inf, np.inf, None, None],\
                        ["Hess", np.inf, np.inf, None, None]]
    elif downSampling == "DEFAULT":
        downSampling=[*[["Opt",  100,    1000, None, None]]*3,\
                      *[["Opt",  1000,   1000, None, None]]*(parXN+parYN),\
                        ["Opt",  np.inf, 200,  None, None],\
                      *[["Boot", 1000,   1000, None, None]]*30] 
    for s, sampStat in enumerate(downSampling):
        if sampStat[0] not in ["Opt", "Boot", "Hess"]:
            print("ERROR: paraLeastSquare: the options are " +\
                  "(\"Opt\", \"Boot\", \"Hess\"), but the following is found:")
            print("    downSampling["+str(s)+"][0] = \""+str(sampStat[0])+"\"")
            sys.exit(0)
    if verbosity >= 1:
        print("\n---------------------------------------------------------------Begin Parametric Fit")
   
    parXOpt, parYOpt = parXYinit[0].copy(), parXYinit[1].copy()
    parXErr, parYErr = [-1 for _ in range(parXN)], [-1 for _ in range(parYN)]
    parXBootErr, parYBootErr = [-1 for _ in range(parXN)], [-1 for _ in range(parYN)]
    parXHessErr, parYHessErr = [-1 for _ in range(parXN)], [-1 for _ in range(parYN)]
    #recover saved parameters for the next optimization
    pickleName = SAVE_DIR+"/zSavedProgress/savedProgress.pickle"
    downSamplingIterN, optIdx, bootIdx = -1, [], []
    parXBoot, parYBoot                 = [[] for _ in range(parXN)], [[] for _ in range(parYN)]
    iterErr2s                          = []
    if readProgress == True:
        if os.path.isdir(SAVE_DIR) is False:
            print("ERROR: paraLeastSquare: the directory for SAVE_DIR does not exist:")
            print("   ", SAVE_DIR)
            sys.exit(0)
        try:
            progressDict = {}
            with open(pickleName, "rb") as handle:
                progressDict = pickle.load(handle)
            downSamplingIterN = progressDict["downSamplingIterN"] + 0
            optIdx            = progressDict["optIdx"].copy()
            bootIdx           = progressDict["bootIdx"].copy()
            parXOpt = progressDict["parXOpt"].copy()
            parYOpt = progressDict["parYOpt"].copy()
            parXErr     = progressDict["parXErr"].copy()
            parYErr     = progressDict["parYErr"].copy() 
            parXBootErr = progressDict["parXBootErr"].copy()
            parYBootErr = progressDict["parYBootErr"].copy()
            parXHessErr = progressDict["parXHessErr"].copy()
            parYHessErr = progressDict["parYHessErr"].copy()
            parXBoot = progressDict["parXBoot"].copy()
            parYBoot = progressDict["parYBoot"].copy()
            iterErr2s  = progressDict["iterErr2"].copy()
        except OSError or FileNotFoundError:
            pathlib.Path(SAVE_DIR+"/zSavedProgress/").mkdir(exist_ok=True)
        #allow change in input parameters
        parXOpt = parXOpt[:parXN]
        for xn in range(parXN):
            if xn >= len(parXOpt):
                parXOpt.append(parXYinit[0][xn])
                parXBoot.append([])
                parXErr.append(-1)
                parXBootErr.append(-1)
                parXHessErr.append(-1)
        parYOpt = parYOpt[:parYN]
        for yn in range(parYN):
            if yn >= len(parYOpt):
                parYOpt.append(parXYinit[1][yn])
                parYBoot.append([])
                parYErr.append(-1)
                parYBootErr.append(-1)
                parYHessErr.append(-1)
    #loop through downSampling
    rd.seed(randSeed)
    for s, sampStat in enumerate(downSampling):
        if s <= downSamplingIterN:
            continue
        if sampStat[0] == "Opt":
            optIdx.append(s)
        elif sampStat[0] == "Boot":
            bootIdx.append(s)
        parXforOpt, parYforOpt = parXOpt.copy(), parYOpt.copy()
        #down-sample data to speed up the inital progress
        dataXforOpt, dataYforOpt = dataXInput.copy(), dataYInput.copy()
        if dataN > sampStat[1]:
            sampledData = rd.choices(list(zip(dataXInput, dataYInput)), k=int(sampStat[1]))
            dataXforOpt = [d[0] for d in sampledData]
            dataYforOpt = [d[1] for d in sampledData]
        elif sampStat[0] == "Boot":
            sampledData = rd.choices(list(zip(dataXInput, dataYInput)), k=dataN)
            dataXforOpt = [d[0] for d in sampledData]
            dataYforOpt = [d[1] for d in sampledData]
        if len(dataXforOpt) < (parXN + parYN):
            print("ERROR: paraLeastSquare: the number of samples ("+str(len(dataXforOpt))+")"+\
                  "is fewer than the number of parameters("+str(parXN + parYN)+")")
            sys.exit(0)
        #main optimization
        if sampStat[0] in ["Opt", "Boot"]: 
            iterErr2s.append([])
            res2Ave = lambda par : \
                paraSquareResidualAve([par[:parXN], par[parXN:]], funcXY, [dataXforOpt, dataYforOpt],\
                                      normXYRatio=normXYRatio, paraRange=paraRange,\
                                      ratioHeadTail=ratioHeadTail, verbosity=verbosity,\
                                      iterErr2=iterErr2s[s], downSamp=[s, sampStat])
            if (s != 0) and (sampStat[2] != 0):
                paraFitResult = optimize.minimize(res2Ave, [*parXforOpt, *parYforOpt],\
                                                  method=optMethod,\
                                                  options={"maxiter":sampStat[2]},\
                                                  bounds=sampStat[3], constraints=sampStat[4])
                parXforOpt = paraFitResult.x[:parXN].tolist()
                parYforOpt = paraFitResult.x[parXN:].tolist()
            else:
                iterErr2s[-1].append([0, iterErr2s[s-1][-1][1], iterErr2s[s-1][-1][2]])
########################################################################NOTE
            if sampStat[0] == "Opt":
                parXOpt, parYOpt = parXforOpt.copy(), parYforOpt.copy()
############################################################################
            if verbosity >= 4:
                if sampStat[0] == "Opt":
                    print("Optimization Result:")
                elif sampStat[0] == "Boot":
                    print("Bootstrap Result:")
                print(paraFitResult)
        #error evaluation
        if (sampStat[0] in ["Boot", "Hess"]) and (s != 0) and (sampStat[2] != 0):
            if verbosity >= 1:
                print("\nEvaluating Standard Errors:")
                print("downSampling["+str(s)+"]=[\""+sampStat[0]+"\", "+
                      str(int(sampStat[1]) if (sampStat[1] < np.inf) else np.inf)+", "+
                      str(int(sampStat[2]) if (sampStat[2] < np.inf) else np.inf)+"]")
            if sampStat[0] == "Boot":
                for xn in range(parXN):
                    parXBoot[xn].append(parXforOpt[xn])
                    parXBootErr[xn] = np.std(np.array(parXBoot[xn]))
                for yn in range(parYN):
                    parYBoot[yn].append(parYforOpt[yn])
                    parYBootErr[yn] = np.std(np.array(parYBoot[yn])) 
                parXErr, parYErr = parXBootErr, parYBootErr
            elif sampStat[0] == "Hess":
                iterErr2Err = []
                res2 = lambda par : len(dataXforOpt)\
                   *paraSquareResidualAve([par[:parXN],par[parXN:]],funcXY,[dataXforOpt,dataYforOpt],\
                                          normXYRatio=normXYRatio, paraRange=paraRange,\
                                          ratioHeadTail=0.0, verbosity=(verbosity-1),\
                                          iterErr2=iterErr2Err, downSamp=[s, sampStat])
                sigma2 = res2([*parXOpt, *parYOpt])/(len(dataXforOpt) - parXN - parYN)
                hessFunc = nd.Hessian(res2)
                hessInv  = linalg.inv(hessFunc([*parXOpt, *parYOpt]))
    
                iterErr2s.append([[0, sigma2, -1]])

                for nx in range(parXN):
                    parXHessErr[nx]=math.sqrt(sigma2*abs(hessInv[nx][nx]))/normXYRatio[0]
                for ny in range(parYN):
                    parYHessErr[ny]=math.sqrt(sigma2*abs(hessInv[parXN+ny][parXN+ny]))/normXYRatio[1]
                parXErr, parYErr = parXHessErr, parYHessErr
        if verbosity >= 1:
            print("Parameter results:")
            print("  parXOpt =", str([scientificStr_paraLeastSquare(par) for par in parXOpt])\
                                 .replace("'",""))
            print("  parYOpt =", str([scientificStr_paraLeastSquare(par) for par in parYOpt])\
                                 .replace("'",""))
            print("  parXErr =", str([scientificStr_paraLeastSquare(par) for par in parXErr])\
                                 .replace("'",""))
            print("  parYErr =", str([scientificStr_paraLeastSquare(par) for par in parYErr])\
                                 .replace("'",""))
        #progress plot
        if progressPlot == True:
            progressPlot_paraLeastSquare([parXforOpt, parYforOpt], funcXY, [dataXforOpt,dataYforOpt],\
                                         dataRangeXY, optIdx=optIdx, bootIdx=bootIdx,\
                                         iterErr2s=iterErr2s, downSamp=[s, sampStat])
        #save the progress
        if saveProgress == True:
            progressDict = {}
            progressDict["downSampling"] = sampStat
            progressDict["parXYinit"]     = parXYinit
            progressDict["funcXY"]        = [str(funcXY[0]), str(funcXY[1])] 
            progressDict["dataRangeXY"]   = dataRangeXY
            progressDict["paraRange"]     = paraRange
            progressDict["optMethod"]     = optMethod
            progressDict["ratioHeadTail"] = ratioHeadTail
            progressDict["randSeed"]      = randSeed
            progressDict["downSamplingIterN"] = s
            progressDict["optimizationN"]     = len(optIdx)
            progressDict["bootstrapN"]        = len(bootIdx)
            progressDict["optIdx"]            = optIdx
            progressDict["bootIdx"]           = bootIdx
            progressDict["parXOpt"]     = parXOpt
            progressDict["parXErr"]     = parXErr
            progressDict["parXBoot"]    = parXBoot
            progressDict["parXBootErr"] = parXBootErr
            progressDict["parXHessErr"] = parXHessErr
            progressDict["parYOpt"]     = parYOpt
            progressDict["parYBoot"]    = parYBoot
            progressDict["parYErr"]     = parYErr
            progressDict["parYBootErr"] = parYBootErr
            progressDict["parYHessErr"] = parYHessErr
            progressDict["iterErr2"]   = [[iterErr2[-1]] for iterErr2 in iterErr2s[:-1]]
            progressDict["iterErr2"]  += [iterErr2s[-1]]
            with open(pickleName, "wb") as handle:
                pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickleDSName = pickleName.replace(".pickle", "DS["+str(s)+"].pickle")
            with open(pickleDSName, "wb") as handle:
                pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity >= 1:
                print("Saving progress:")
                print("   ", pickleName, "\n   ", pickleDSName)
                if verbosity >= 2:
                    print("   ", [({key: str(progressDict[key])} if key != "iterErr2" else\
                                   {key: str(progressDict[key][-1][-1])}) for key in progressDict])
        if (verbosity >= 1) and ((len(downSampling) > 0)):
            print("----------------------------------------------downSampling["+str(s)+"] Complete\n")
    if verbosity >= 1:
        print("-----------------------------------------------------------Parametric Fit Complete\n")
    res2AveVal = iterErr2s[-1][-1][1]
    return parXOpt, parYOpt, parXErr, parYErr, res2AveVal
def paraSquareResidualAve(parXY, funcXY, dataXY, normXYRatio=[1.0, 1.0], paraRange=[0.0, 1.0],\
                          ratioHeadTail=0.0, iterErr2=None, downSamp=None, verbosity=1):
    if len(dataXY[0]) != len(dataXY[1]):
        print("ERROR: paraSquareErrorAve: lengths of dataX and dataY don't match")
        sys.exit(0)
    outputStr = ""
    if iterErr2 is not None:
        if len(iterErr2) == 0:
            iterErr2.append([0, None, None])
        else:
            iterErr2.append([iterErr2[-1][0]+1, None])
        outputStr += "iter "+str(iterErr2[-1][0])
    if downSamp is not None:
        if outputStr != "":
            outputStr += ", "
        outputStr += "downSampling["+str(downSamp[0])+"]="
        outputStr += "[\"" + downSamp[1][0] + "\", "
        outputStr += str(int(downSamp[1][1]) if (downSamp[1][1] < np.inf) else np.inf) + ", "
        outputStr += str(int(downSamp[1][2]) if (downSamp[1][2] < np.inf) else np.inf) + "]"
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
        print("  parX =", str([scientificStr_paraLeastSquare(par) for par in parXY[0]])\
                          .replace("'",""))
        print("  parY =", str([scientificStr_paraLeastSquare(par) for par in parXY[1]])\
                          .replace("'",""))
        print("  [min_t, max_t] =",str([scientificStr_paraLeastSquare(min(opt_ts)),\
                                        scientificStr_paraLeastSquare(max(opt_ts))]).replace("'", ""))
        print("  head_tail normalized square error =", scientificStr_paraLeastSquare(err2HeadTail,10))
        print("")
    return res2Ave + err2HeadTail
def paraSquareDist(t, funcXY, dataXY, normXYRatio=[1.0, 1.0]):
    return pow(normXYRatio[0]*(funcXY[0](t) - dataXY[0]), 2) +\
           pow(normXYRatio[1]*(funcXY[1](t) - dataXY[1]), 2)
def progressPlot_paraLeastSquare(parXYFit, funcXY, dataXY, dataRangeXY, verbosity=1,\
                                 optIdx=None, bootIdx=None, iterErr2s=None, downSamp=None):
    if downSamp[1][0] == "Hess":
        return
    pathlib.Path(SAVE_DIR+"/zSavedProgress/").mkdir(exist_ok=True)
    figName = SAVE_DIR+"/zSavedProgress/progressPlot_Opt.png"
    if downSamp[1][0] == "Boot":
        figName = figName.replace("Opt", "Boot")
    def truncateColorMap(cmap, lowR, highR):
        cmapNew = matplotlib.colors.LinearSegmentedColormap.from_list(\
            "trunc({n}, {l:.2f}, {h:.2f})".format(n=cmap.name, l=lowR, h=highR),\
            cmap(np.linspace(lowR, highR, 1000)))
        return cmapNew

    iterations, res2Aves    = [], []
    bootIters, bootRes2Aves = [], []
    totIter, minIter, res2Min = 0, 0, 1e12
    if downSamp[1][0] == "Opt":
        for s, iterErr2 in enumerate(iterErr2s):
            if s in optIdx:
                totIter += iterErr2[-1][0]
                iterations.append(totIter)
                res2Aves.append(iterErr2[-1][1])
                if iterErr2[-1][1] < res2Min:
                    minIter = totIter + 0
                    res2Min = iterErr2[-1][1] + 0.0
    elif downSamp[1][0] == "Boot":
        for s, iterErr2 in enumerate(iterErr2s):
            if (s in optIdx) or (s in bootIdx):
                totIter += iterErr2[-1][0]
                iterations.append(totIter)
                res2Aves.append(iterErr2[-1][1])
                if iterErr2[-1][1] < res2Min:
                    minIter = totIter + 0
                    res2Min = iterErr2[-1][1] + 0.0
                if s in bootIdx:
                    bootIters.append(totIter)
                    bootRes2Aves.append(iterErr2[-1][1])

    binN = int(max(10, min(1000, 10*math.sqrt(downSamp[1][1]))))
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
    ax[0].plot(iterations, res2Aves, color="blue", linewidth=2, marker="o", markersize=5)
    scatter = ax[0].scatter(bootIters, bootRes2Aves, s=120, color="orange")
    ax[0].set_title("Normalized Residual Square Average at Each DownSampling", fontsize=24, y=1.03)
    ax[0].set_xlabel("iterations", fontsize=20)
    ax[0].set_ylabel("residual", fontsize=20)
    xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
    ax[0].set_xlim(0, 1.04*xlim[1])
    ax[0].text(minIter, ylim[0]+0.015*(ylim[1]-ylim[0]), scientificStr_paraLeastSquare(res2Min, 3),\
               color="blue", fontsize=14, weight="bold")

    if downSamp[1][0] == "Boot":
        ax[0].legend([scatter], ["bootstrap"], scatterpoints=1, loc="upper right", fontsize=16)
    plt.savefig(figName)
    plt.cla()

    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=1.0)
    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[0].hist2d(*dataXY, bins=binN, cmin=1, cmap=cmap, range=dataRangeXY)
    cb = fig.colorbar(hist[3], ax=ax[0]).mappable
    ax[0].plot(fitFuncX, fitFuncY, linewidth=3, color="red")
    plotTile =  "DownSampling[" + str(downSamp[0]) + "]="
    plotTile += "[\"" + downSamp[1][0] + "\","
    plotTile += str(int(downSamp[1][1]) if (downSamp[1][1] < np.inf) else np.inf) + ","
    plotTile += str(int(downSamp[1][2]) if (downSamp[1][2] < np.inf) else np.inf) + "], "
    plotTile += "Iter=" + str(iterErr2s[-1][-1][0]) + ", NormSqErr="
    plotTile += scientificStr_paraLeastSquare(iterErr2s[-1][-1][1])
    ax[0].set_title(plotTile, fontsize=20, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_xlim(*dataRangeXY[0])
    ax[0].set_ylim(*dataRangeXY[1])
    
    if downSamp[1][2] != 0:
        figDSName = figName.replace("progressPlot", "progressPlotDS["+str(s)+"]") 
    else:
        figDSName = figName.replace("progressPlot", "progressPlotDS[]")
    plt.savefig(figDSName)
    plt.close(fig)
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
def printSavedProgress(fullPicklePath, verbosity=2):
    progressDict = {}
    try:
        with open(fullPicklePath, "rb") as handle:
            progressDict = pickle.load(handle)
    except OSError or FileNotFoundError:
        print("The following file does not exist:\n ", fullPath)
        sys.exit(0)

    print("Print results from:\n ", fullPicklePath)
    print("downSampling["+str(progressDict["downSamplingIterN"])+"]="+\
          str(progressDict["downSampling"][:3]))
    print("iterErr2[-1] =", progressDict["iterErr2"][-1][-1])
    print("  parXOpt =", str([scientificStr_paraLeastSquare(par) for par in progressDict["parXOpt"]])\
                              .replace("'",""))
    print("  parYOpt =", str([scientificStr_paraLeastSquare(par) for par in progressDict["parYOpt"]])\
                              .replace("'",""))
    print("  parXErr =", str([scientificStr_paraLeastSquare(err) for err in progressDict["parXErr"]])\
                              .replace("'",""))
    print("  parYErr =", str([scientificStr_paraLeastSquare(err) for err in progressDict["parYErr"]])\
                              .replace("'",""))
    if verbosity >= 1:
        print("")
        print("  optimizationN =", progressDict["optimizationN"])
        print("  bootstrapN    =", progressDict["bootstrapN"])
        print("  parXBoot    =", str([scientificStr_paraLeastSquare(par[-1]) if (par != []) else -1\
                                     for par in progressDict["parXBoot"]]).replace("'",""))
        print("  parYBoot    =", str([scientificStr_paraLeastSquare(par[-1]) if (par != []) else -1\
                                     for par in progressDict["parYBoot"]]).replace("'",""))
        print("  parXBootErr =", str([scientificStr_paraLeastSquare(err)\
                                     for err in progressDict["parXBootErr"]]).replace("'","")) 
        print("  parYBootErr =", str([scientificStr_paraLeastSquare(err)\
                                     for err in progressDict["parYBootErr"]]).replace("'","")) 
        print("  parXHessErr =", str([scientificStr_paraLeastSquare(err)\
                                     for err in progressDict["parXHessErr"]]).replace("'","")) 
        print("  parYHessErr =", str([scientificStr_paraLeastSquare(err)\
                                     for err in progressDict["parYHessErr"]]).replace("'","")) 
    if verbosity >= 2:
        print("")
        print("  parXInit =", str([scientificStr_paraLeastSquare(par)\
                                  for par in progressDict["parXYinit"][0]]).replace("'",""))
        print("  parYInit =", str([scientificStr_paraLeastSquare(par)\
                                  for par in progressDict["parXYinit"][1]]).replace("'",""))
        print("  funcX =", progressDict["funcXY"][0])
        print("  funcY =", progressDict["funcXY"][1])
        print("  dataRangeXY   =", progressDict["dataRangeXY"])
        print("  paraRange     =", progressDict["paraRange"])
        print("  optMethod     =", progressDict["optMethod"])
        print("  ratioHeadTail =", progressDict["ratioHeadTail"])
        print("  randSeed      =", progressDict["randSeed"])




















#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
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

    optMethod = "Nelder-Mead"
    savePlot, saveProg, readProg = True, True, True

    ###################################################################################################
    #parametric fit
    ###heavily depends on initial conditions, can test with downSampling=[*[[100, 1]]*1] first
    funcX = polyFunc
    funcY = polyFunc    
    ###round1
    #initX = [1.0, -15.0, 43.0, -10.0, -20.0, 0.0, 0.0]
    #initY = [0.0, -8.0,  12.0, -4.0,   1.0,  0.0, 0.0]
    #downSampling=[*[["Opt",  100,    1000, None, None]]*3,\
    #              *[["Opt",  1000,   1000, None, None]]*14,\
    #                ["Opt",  np.inf, 200,  None, None],\
    #              *[["Boot", 1000,   1000, None, None]]*30]
    ###round2 using results of round1
    ###[ 1.53e-4, -0.238, 15.6, -10.8, -5.76, -5.76e-5, -3.04e-4]
    ###[-6.73e-4, -11.3,  23.8, -3.5,  -8.04, -0.0163,   0.0573]
    initX = [0.8,      -8.238, 23.6, -10.8, -5.76, 20.0, -20.0]
    initY = [-6.73e-4, -11.3,  23.8, -3.5,  -8.04, -2.0, 2.3]
    downSampling=[*[["Opt",  1000,   1000, None, None]]*14,\
                    ["Opt",  np.inf, 200,  None, None],\
                  *[["Boot", 1000,   1000, None, None]]*30]
    ###set maxiter to 0 with (progressPlot=True, saveProgress=False) prints the plot with initXY
    #downSampling = [*[["Opt", 1000, 0, None, None]]*1]
    ###plotting the last entry indefinitely
    #downSampling.append([downSampling[-1][0], np.inf, 0,  None, None])
    #saveProg=False
    ###################################################################################################
    
    parXOpt, parYOpt, parXErr, parYErr, res2AveVal = \
        paraLeastSquare([initX, initY], [funcX, funcY], data, rangeXY, optMethod=optMethod,\
                        ratioHeadTail=0.01, randSeed=0, downSampling=downSampling,\
                        verbosity=3, progressPlot=savePlot, saveProgress=saveProg,readProgress=readProg)

    fitT = np.linspace(0.0, 1.0, binN+1)[:-1]
    fitFuncX = funcX(fitT, parXOpt)
    fitFuncY = funcY(fitT, parYOpt)
    print("Results from paraLeastSquare():")
    print("  average normalized square residual =", scientificStr_paraLeastSquare(res2AveVal))
    print("  parXOpt =", str([scientificStr_paraLeastSquare(par) for par in parXOpt]).replace("'", ""))
    print("  parYOpt =", str([scientificStr_paraLeastSquare(par) for par in parYOpt]).replace("'", ""))
    print("  parXErr =", str([scientificStr_paraLeastSquare(err) for err in parXErr]).replace("'", ""))
    print("  parYErr =", str([scientificStr_paraLeastSquare(err) for err in parYErr]).replace("'", ""))
    #plot
    fig = plt.figure(figsize=(12, 18))
    matplotlib.rc("xtick", labelsize=16)
    matplotlib.rc("ytick", labelsize=16)
    gs = gridspec.GridSpec(2, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols):
        ax.append(fig.add_subplot(gs[i]));
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.095, right=0.98)

    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[0].hist2d(*data, bins=int(binN/10.0), cmin=1, cmap=cmap, range=rangeXY)
    cb = fig.colorbar(hist[3], ax=ax[0]).mappable
    ax[0].plot(fitFuncX, fitFuncY, linewidth=3, color="red") 
    ax[0].set_title("Parametric Fitting the Curve", fontsize=28, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(*rangeXY[0])
    ax[0].set_ylim(*rangeXY[1])

    errCurveN = 1000
    print("Sampling the fit curve from the parameter standard errors...")
    for i in tqdm(range(errCurveN)):
        parXOptSamp = [rd.gauss(par, parXErr[nx]) for nx, par in enumerate(parXOpt)]
        parYOptSamp = [rd.gauss(par, parYErr[ny]) for ny, par in enumerate(parYOpt)]
        sampFitFuncX = funcX(fitT, parXOptSamp)
        sampFitFuncY = funcY(fitT, parYOptSamp) 
        plotFitted, = ax[1].plot(sampFitFuncX, sampFitFuncY, linewidth=3, color="red", alpha=0.5)
    plotGiven, = ax[1].plot(curveX, curveY, linewidth=5, color="blue")
    ax[1].set_title("Parametric Curve: Given vs Fitted (sampled from fit err)", fontsize=24, y=1.03)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_aspect("equal")
    ax[1].set_xlim(rangeXY[0][0], rangeXY[0][1]+1.18)
    ax[1].set_ylim(*rangeXY[1])
    ax[1].legend([plotGiven, plotFitted], ["give", "fitted"], loc="upper right", fontsize=20)

    figName = "paraFitCurve2D.png"
    plt.savefig(figName)
    plt.close(fig)
    print("Saving the following file:\n ", figName)


if __name__ == "__main__":
    print("##################################################################################Begin\n")
    #printSavedProgress("zSavedProgress/savedProgressDS[0].pickle")
    example_parametricFit2D()
    print("\n##################################################################################End\n")



