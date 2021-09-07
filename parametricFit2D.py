import os, sys, pathlib, time, math
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




#######################################################################################################
##########################################   INPUT   ##################################################
#downSampling[i] = [operation type ("Opt": optimization, 
#                                   "Boot": error from bootstrap, recommending 30 iterations,
#                                   "Hess": error from inverse Hessian using numdifftools),
#                   sampling size with replacement, 
#                   maxiter for least square optimization,
#                   bounds, constraints]
#bounds/constraints only for optMethod = "COBYLA", "SLSQP", "trust-constr":
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
#parXYinit     = [[initial parameter values for x-axis], [initial parameter values for y-axis]]
#funcXY        = [parametric fit function for x-axis,    parametric fit function for y-axis]
#dataXY        = [[data values for x-axis],              [data values for y-axis]]
#dataRangeXY   = [[data range for x-axis],               [data range for y-axis]]
#  Data that fall outside the range will be removed
#  The range also goes into normalizing the residual square
#optMethod     = optimization method for the least square (minimizing the the residual square)
#paraRange     = range allowed for the parametric variable t, whose fit function is (x(t), y(t))
#ratioHeadTail = adding weights to the the residual square to maintain paraRange
#randSeed      = random seed for downSampling
#downSampling  = (see above)
#verbosity:    controlling the amount of output messages, up to 4
#progressPlot: saving plots in savePath at each downSampling iteration
#saveProgress: saving pickle files in savePath at each downSampling iteration
#readProgress: reading from pickle files (if exist) to continue the optimization
#savePath    : the directory where the files are saved to
##########################################   OUTPUT   #################################################
#parXYOpt, parXYErr, res2AveVal
#parXYOpt   = [[optimized parameter values for x-axis], [optimized parameter values for y-axis]]
#parXYErr   = [[parameter standard errors for x-axis],  [parameter standard errors for y-axis],
#              [[parameter estimate covariant matrix for xy-axis]]]
#res2AveVal = average residual square of the optimized result 

_PARAMETRICFIT2D_SAVEDIRNAME = "zSavedProgress"
#keys of the dictionary saved in the progress pickle files:
#   downSampling, parXYinit, funcXY, dataRangeXY, paraRange, optMethod, ratioHeadTail, randseed,
#   downSamplingIterN, optimizationN, bootstrapN, optIdx, bootIdx, 
#   parXOpt, parXErr, parXBoot, parXBootErr, parXHessErr,
#   parYOpt, parYErr, parYBoot, parYBootErr, parYHessErr,
#   parErrCovMatrix, parBootErrCovMatrix, parHessErrCovMatrix, iterErr2

def paraLeastSquare(parXYinit, funcXY, dataXY, dataRangeXY, optMethod="Nelder-Mead",\
                    paraRange=[-1.0, 1.0], ratioHeadTail=[0.01, 0.01],\
                    randSeed=None, downSampling="DEFAULT", verbosity=3,\
                    progressPlot=False, saveProgress=False, readProgress=True,\
                    savePath=str(pathlib.Path().absolute())):
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
    if len(dataXInput) != len(dataYInput):
        raise ValueError("paraLeastSquare: lengths of dataX and dataY don't match")
 
    if (downSampling is None) or (len(downSampling) == 0):
        downSampling = [["Opt",  np.inf, np.inf, None, None],\
                        ["Hess", np.inf, np.inf, None, None]]
    elif downSampling == "DEFAULT":
        downSampling=[  ["Opt",  1000,   0,      None, None],\
                      *[["Opt",  1000,   1000,   None, None]]*2,\
                        ["Opt",  np.inf, np.inf, None, None],\
                      *[["Boot", np.inf, 200,    None, None]]*30]
    for s, sampStat in enumerate(downSampling):
        if sampStat[0] not in ["Opt", "Boot", "Hess"]:
            raise ValueError("paraLeastSquare: the options are "+\
                             "(\"Opt\", \"Boot\", \"Hess\"), but the following is found:\n"+\
                             "    downSampling["+str(s)+"][0] = \""+str(sampStat[0])+"\"")
    if verbosity >= 1:
        print("\n----------------------------------------------------------------Begin Parametric Fit")
   
    parXOpt, parYOpt = parXYinit[0].copy(), parXYinit[1].copy()
    parXErr, parYErr = [0 for _ in range(parXN)], [0 for _ in range(parYN)]
    parErrCovMatrix  = [[0 for _ in range(parXN+parYN)] for _ in range(parXN+parYN)]
    parXBootErr, parYBootErr = [0 for _ in range(parXN)], [0 for _ in range(parYN)]
    parBootErrCovMatrix      = [[0 for _ in range(parXN+parYN)] for _ in range(parXN+parYN)]
    parXHessErr, parYHessErr = [0 for _ in range(parXN)], [0 for _ in range(parYN)]
    parHessErrCovMatrix      = [[0 for _ in range(parXN+parYN)] for _ in range(parXN+parYN)]
    #recover saved parameters for the next optimization
    pickleName = "/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME, "savedProgress.pickle") )
    downSamplingIterN, optIdx, bootIdx = -1, [], []
    parXBoot, parYBoot                 = [[] for _ in range(parXN)], [[] for _ in range(parYN)]
    iterParaRanges, iterErr2s          = [], []
    if readProgress == True:
        if os.path.isdir(savePath) is False:
            raise NotADirectoryError("paraLeastSquare: the directory for savePath does not exist:\n"+\
                                     "   ", savePath)
        try:
            progressDict = {}
            with open(pickleName, "rb") as handle: progressDict = pickle.load(handle)
            downSamplingIterN = progressDict["downSamplingIterN"] + 0
            optIdx            = progressDict["optIdx"].copy()
            bootIdx           = progressDict["bootIdx"].copy()
            parXOpt = progressDict["parXOpt"].copy()
            parYOpt = progressDict["parYOpt"].copy()
            parXErr         = progressDict["parXErr"].copy()
            parYErr         = progressDict["parYErr"].copy() 
            parErrCovMatrix = progressDict["parErrCovMatrix"].copy()
            parXBootErr         = progressDict["parXBootErr"].copy()
            parYBootErr         = progressDict["parYBootErr"].copy()
            parBootErrCovMatrix = progressDict["parBootErrCovMatrix"].copy()
            parXHessErr         = progressDict["parXHessErr"].copy()
            parYHessErr         = progressDict["parYHessErr"].copy()
            parHessErrCovMatrix = progressDict["parHessErrCovMatrix"].copy()
            parXBoot = progressDict["parXBoot"].copy()
            parYBoot = progressDict["parYBoot"].copy()
            iterParaRanges = progressDict["iterParaRanges"].copy()
            iterErr2s      = progressDict["iterErr2"].copy()
        except OSError or FileNotFoundError:
            pathlib.Path("/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME) )).mkdir(exist_ok=True)
            readProgress = False
        #allow change in parXYinit dimention on existing progress pickle file
        if (parXN != len(parXOpt)) or (parYN != len(parYOpt)):
            parErrCovMatrixNew     = [[0 for _ in range(parXN+parYN)] for _ in range(parXN+parYN)]
            parBootErrCovMatrixNew = [[0 for _ in range(parXN+parYN)] for _ in range(parXN+parYN)]
            parHessErrCovMatrixNew = [[0 for _ in range(parXN+parYN)] for _ in range(parXN+parYN)]
            for col in range(parXN+parYN):
                for row in range(parXN+parYN):
                    if col < parXN:
                        if col < len(parXOpt):
                            if row < parXN:
                                if row < len(parXOpt):
                                    parErrCovMatrixNew[col][row]     = parErrCovMatrix[col][row]
                                    parBootErrCovMatrixNew[col][row] = parBootErrCovMatrix[col][row]
                                    parHessErrCovMatrixNew[col][row] = parHessErrCovMatrix[col][row]
                            else:
                                if row < (parXN + len(parYOpt)):
                                    rowOld = row - parXN + len(parXOpt)
                                    parErrCovMatrixNew[col][row]     = parErrCovMatrix[col][rowOld]
                                    parBootErrCovMatrixNew[col][row] = parBootErrCovMatrix[col][rowOld]
                                    parHessErrCovMatrixNew[col][row] = parHessErrCovMatrix[col][rowOld]
                    else:
                        if col < (parXN + len(parYOpt)):
                            colOld = col - parXN + len(parXOpt)
                            if row < parXN:
                                if row < len(parXOpt):
                                    parErrCovMatrixNew[col][row]     = parErrCovMatrix[colOld][row]
                                    parBootErrCovMatrixNew[col][row] = parBootErrCovMatrix[colOld][row]
                                    parHessErrCovMatrixNew[col][row] = parHessErrCovMatrix[colOld][row]
                            else:   
                                if row < (parXN + len(parYOpt)):
                                    rowOld = row - parXN + len(parXOpt)
                                    parErrCovMatrixNew[col][row]     = parErrCovMatrix[colOld][rowOld]
                                    parBootErrCovMatrixNew[col][row] = \
                                        parBootErrCovMatrix[colOld][rowOld]
                                    parHessErrCovMatrixNew[col][row] = \
                                        parHessErrCovMatrix[colOld][rowOld]
            parErrCovMatrix     = parErrCovMatrixNew
            parBootErrCovMatrix = parBootErrCovMatrixNew
            parHessErrCovMatrix = parHessErrCovMatrixNew
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
        if s <= downSamplingIterN: continue
        if   sampStat[0] == "Opt":  optIdx.append(s)
        elif sampStat[0] == "Boot": bootIdx.append(s)
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
            raise AssertionError("paraLeastSquare: the number of samples("+str(len(dataXforOpt))+") "+\
                                 "is fewer than the number of parameters("+str(parXN + parYN)+")")
        #updating the paraRange
        iterParaRanges.append(paraRange)
        if len(iterParaRanges) > 1:
            preParaRange = iterParaRanges[-2]
            for paraIdx, localMiniCut in enumerate(preParaRange):
                if (paraIdx != 0) and (paraIdx != len(preParaRange)-1):            
                    localMiniDateXY  = getDataPtOfPara([parXforOpt, parYforOpt], funcXY, localMiniCut)
                    localMiniPara, _ = getParaOfDataPt([parXforOpt, parYforOpt], funcXY,\
                                                       localMiniDateXY, dataRangeXY, preParaRange)
                    iterParaRanges[-1][paraIdx] = localMiniPara + 0.0
        #main optimization
        if sampStat[0] in ["Opt", "Boot"]: 
            iterErr2s.append([])
            res2Ave = lambda par : \
                _paraSquareResidualAve([par[:parXN], par[parXN:]], funcXY, [dataXforOpt, dataYforOpt],\
                                       dataRangeXY, paraRange=iterParaRanges[-1],\
                                       ratioHeadTail=ratioHeadTail, verbosity=verbosity,\
                                       iterErr2=iterErr2s[s], downSamp=[s, sampStat])
            if sampStat[2] != 0:
                paraFitResult = optimize.minimize(res2Ave, [*parXforOpt, *parYforOpt],\
                                                  method=optMethod, options={"maxiter":sampStat[2]},\
                                                  bounds=sampStat[3], constraints=sampStat[4])
                parXforOpt = paraFitResult.x[:parXN].tolist()
                parYforOpt = paraFitResult.x[parXN:].tolist()
            else: 
                res2Ave([*parXforOpt, *parYforOpt])
########################################################################NOTE
            if sampStat[0] == "Opt": parXOpt, parYOpt = parXforOpt.copy(), parYforOpt.copy()
############################################################################
            if verbosity >= 4:
                if   sampStat[0] == "Opt":  print("Optimization Result:")
                elif sampStat[0] == "Boot": print("Bootstrap Result:")
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
                    parXBootErr[xn] = np.std(np.array(parXBoot[xn]), ddof=1)
                for yn in range(parYN):
                    parYBoot[yn].append(parYforOpt[yn])
                    parYBootErr[yn] = np.std(np.array(parYBoot[yn]), ddof=1)
                parBootErrCovMatrix = np.cov(np.array(parXBoot+parYBoot)).tolist()
                parXErr, parYErr, parErrCovMatrix = parXBootErr, parYBootErr, parBootErrCovMatrix
            elif sampStat[0] == "Hess":
                iterErr2Err = []
                res2 = lambda par : len(dataXforOpt)\
                   *_paraSquareResidualAve([par[:parXN],par[parXN:]],funcXY,[dataXforOpt,dataYforOpt],\
                                           dataRangeXY, paraRange=iterParaRanges[-1],\
                                           ratioHeadTail=[0.0, 0.0], verbosity=verbosity,\
                                           iterErr2=iterErr2Err, downSamp=[s, sampStat],\
                                           progressPlot=progressPlot)
                sigma2 = res2([*parXOpt, *parYOpt])/(len(dataXforOpt) - parXN - parYN)
                iterErr2s.append([[0, sigma2, -1]])

                hessFunc = nd.Hessian(res2)
                hessInv  = linalg.inv(hessFunc([*parXOpt, *parYOpt]))
                normXYRatio = [1.0/(dataRangeXY[0][1]-dataRangeXY[0][0]),\
                               1.0/(dataRangeXY[1][1]-dataRangeXY[1][0])]
                for nx in range(parXN):
                    parXHessErr[nx] = math.sqrt(sigma2*abs(hessInv[nx][nx]))/normXYRatio[0]
                for ny in range(parYN):
                    parYHessErr[ny] = math.sqrt(sigma2*abs(hessInv[parXN+ny][parXN+ny]))/normXYRatio[1]
                for col in range(parXN+parYN):
                    for row in range(parXN+parYN):
                        parHessErrCovMatrix[col][row] = sigma2*abs(hessInv[col][row])
                        if col < parXN: parHessErrCovMatrix[col][row] /= normXYRatio[0]
                        else:           parHessErrCovMatrix[col][row] /= normXYRatio[1]
                        if row < parXN: parHessErrCovMatrix[col][row] /= normXYRatio[0]
                        else:           parHessErrCovMatrix[col][row] /= normXYRatio[1]
                    parHessErrCovMatrix[col][row] = math.sqrt(parHessErrCovMatrix[col][row])
                parXErr, parYErr, parErrCovMatrix = parXHessErr, parYHessErr, parHessErrCovMatrix
        if verbosity >= 1:
            print("Parameter results:")
            print("  parXOpt =", str([_parametricFit2D_scientificStr(par) for par in parXOpt])\
                                 .replace("'",""))
            print("  parYOpt =", str([_parametricFit2D_scientificStr(par) for par in parYOpt])\
                                 .replace("'",""))
            print("  parXErr =", str([_parametricFit2D_scientificStr(par) for par in parXErr])\
                                 .replace("'",""))
            print("  parYErr =", str([_parametricFit2D_scientificStr(par) for par in parYErr])\
                                 .replace("'",""))
        #progress plot
        if progressPlot == True:
            _parametricFit2D_progressPlot([parXforOpt, parYforOpt], funcXY, [dataXforOpt,dataYforOpt],\
                                          dataRangeXY, paraRange=paraRange, verbosity=verbosity,\
                                          optIdx=optIdx, bootIdx=bootIdx, iterErr2s=iterErr2s,\
                                          downSamp=[s, sampStat], saveProgress=saveProgress,\
                                          savePath=savePath)
        #save progress
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
            progressDict["parErrCovMatrix"]     = parErrCovMatrix
            progressDict["parBootErrCovMatrix"] = parBootErrCovMatrix
            progressDict["parHessErrCovMatrix"] = parHessErrCovMatrix
            progressDict["iterParaRanges"] = iterParaRanges
            progressDict["iterErr2"]       = [[iterErr2[-1]] for iterErr2 in iterErr2s[:-1]]
            progressDict["iterErr2"]      += [iterErr2s[-1]]
            with open(pickleName, "wb") as handle:
                pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickleDSName = pickleName.replace(".pickle", "DS["+str(s)+"].pickle")
            with open(pickleDSName, "wb") as handle:
                pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if verbosity >= 1:
                print("Saving progress:")
                print(" ", pickleName, "\n ", pickleDSName)
        if readProgress == True:     
            if verbosity >= 2:
                np.set_printoptions(precision=2, linewidth=200)
                print("Lastest progress entry:")
                for key in progressDict:
                    if key == "iterErr2": 
                        print(" ", key+":", end=" ")
                        if verbosity >= 3: 
                            iterErr2List = progressDict[key][:-1].copy()
                            iterErr2List.append(progressDict[key][-1][-1])
                            print(iterErr2List)
                        else:
                            print(progressDict[key][-1][-1])
                    else:
                        if np.array(progressDict[key]).ndim <= 1:
                            print(" ", key+":", np.array(progressDict[key]))
                        else:
                            print(" ", key+":\n   ",\
                                  str(np.array(progressDict[key])).replace("\n", "\n    "))
                np.set_printoptions(precision=8, linewidth=75)  #default
        if (verbosity >= 1) and ((len(downSampling) > 0)):
            print("-----------------------------------------------downSampling["+str(s)+"] Complete\n")
    if verbosity >= 1:
        print("-------------------------------------------------------------Parametric Fit Complete\n")
    res2AveVal = iterErr2s[-1][-1][1]
    return [parXOpt, parYOpt], [parXErr, parYErr, parErrCovMatrix], res2AveVal
def getDataPtOfPara(parXY, funcXY, paraVal):
    lambdaX = lambda t : funcXY[0](t, parXY[0])
    lambdaY = lambda t : funcXY[1](t, parXY[1])
    return [lambdaX(paraVal), lambdaY(paraVal)]
def getParaOfDataPt(parXY, funcXY, dataPtXY, dataRangeXY, paraRange, iterCounter=[]):
    lambdaX = lambda t : funcXY[0](t, parXY[0])
    lambdaY = lambda t : funcXY[1](t, parXY[1])
    distSquare = lambda t : _paraSquareDist(t, [lambdaX, lambdaY], dataPtXY, dataRangeXY, iterCounter)
    opt_t = paraRange[0]
    for paraIdx in range(len(paraRange)-1):
        bounds = (paraRange[paraIdx], paraRange[paraIdx+1])
        optResult = optimize.minimize_scalar(distSquare, method="bounded", bounds=bounds)
        if distSquare(optResult.x) < distSquare(opt_t): opt_t = optResult.x
    return opt_t, distSquare(opt_t)
def getInstantPlotDownSampling(downSampling=None, savePath=str(pathlib.Path().absolute())):
    if downSampling == None: return [["Opt", np.inf, 0, None, None]]
    pickleName = "/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME, "savedProgress.pickle") )
    if os.path.isfile(pickleName) == False: return [["Opt", downSampling[0][1], 0, None, None]]
    progressDict = {}
    with open(pickleName, "rb") as handle: progressDict = pickle.load(handle)
    downSampN = len(progressDict["iterErr2"])
    downSamplingO = downSampling.copy()
    if downSampN < len(downSamplingO): downSamplingO = downSamplingO[:downSampN]
    downSampNtoAdd = downSampN - len(downSamplingO) + 1
    lastDS = downSamplingO[-1].copy()
    downSamplingO = [*downSamplingO, *[["Opt", lastDS[1], 0, *lastDS[3:]]]*downSampNtoAdd]
    return downSamplingO
def printSavedProgress(fullPicklePath, verbosity=2):
    progressDict = {}
    try:
        with open(fullPicklePath, "rb") as handle: progressDict = pickle.load(handle)
    except OSError or FileNotFoundError:
        print("The following file does not exist:\n ", fullPath)
        sys.exit(0)

    print("Print results from:\n ", fullPicklePath)
    print("downSampling["+str(progressDict["downSamplingIterN"])+"]="+\
          str(progressDict["downSampling"][:3]))
    print("iterErr2[-1] =", progressDict["iterErr2"][-1][-1])
    print("  parXOpt =", str([_parametricFit2D_scientificStr(par) for par in progressDict["parXOpt"]])\
                              .replace("'",""))
    print("  parYOpt =", str([_parametricFit2D_scientificStr(par) for par in progressDict["parYOpt"]])\
                              .replace("'",""))
    print("  parXErr =", str([_parametricFit2D_scientificStr(err) for err in progressDict["parXErr"]])\
                              .replace("'",""))
    print("  parYErr =", str([_parametricFit2D_scientificStr(err) for err in progressDict["parYErr"]])\
                              .replace("'",""))
    if verbosity >= 1:
        print("")
        print("  optimizationN =", progressDict["optimizationN"])
        print("  bootstrapN    =", progressDict["bootstrapN"])
        print("  parXBoot    =", str([_parametricFit2D_scientificStr(par[-1]) if (par != []) else -1\
                                     for par in progressDict["parXBoot"]]).replace("'",""))
        print("  parYBoot    =", str([_parametricFit2D_scientificStr(par[-1]) if (par != []) else -1\
                                     for par in progressDict["parYBoot"]]).replace("'",""))
        print("  parXBootErr =", str([_parametricFit2D_scientificStr(err)\
                                     for err in progressDict["parXBootErr"]]).replace("'","")) 
        print("  parYBootErr =", str([_parametricFit2D_scientificStr(err)\
                                     for err in progressDict["parYBootErr"]]).replace("'","")) 
        print("  parXHessErr =", str([_parametricFit2D_scientificStr(err)\
                                     for err in progressDict["parXHessErr"]]).replace("'","")) 
        print("  parYHessErr =", str([_parametricFit2D_scientificStr(err)\
                                     for err in progressDict["parYHessErr"]]).replace("'","")) 
    if verbosity >= 2:
        print("")
        print("  parXInit =", str([_parametricFit2D_scientificStr(par)\
                                  for par in progressDict["parXYinit"][0]]).replace("'",""))
        print("  parYInit =", str([_parametricFit2D_scientificStr(par)\
                                  for par in progressDict["parXYinit"][1]]).replace("'",""))
        print("  funcX =", progressDict["funcXY"][0])
        print("  funcY =", progressDict["funcXY"][1])
        print("  dataRangeXY   =", progressDict["dataRangeXY"])
        print("  paraRange     =", progressDict["paraRange"])
        print("  optMethod     =", progressDict["optMethod"])
        print("  ratioHeadTail =", progressDict["ratioHeadTail"])
        print("  randSeed      =", progressDict["randSeed"])
#######################################################################################################
#helper functions
def _paraSquareResidualAve(parXY, funcXY, dataXY, dataRangeXY, paraRange=[-1.0, 1.0],\
                           ratioHeadTail=[0.0, 0.0], iterErr2=None, downSamp=None,\
                           verbosity=1, progressPlot=True, savePath="."):
    if len(dataXY[0]) != len(dataXY[1]):
        raise ValueError("paraSquareErrorAve: lengths of dataX and dataY don't match")
    checkParaRange = False
    outputStr = ""
    if iterErr2 is not None:
        if len(iterErr2) == 0: iterErr2.append([0, None, None])
        else: iterErr2.append([iterErr2[-1][0]+1, None])
        outputStr += "iter "+str(iterErr2[-1][0])
    if downSamp is not None:
        if outputStr != "": outputStr += ", "
        outputStr += "downSampling["+str(downSamp[0])+"]="
        outputStr += "[\"" + downSamp[1][0] + "\", "
        outputStr += str(int(downSamp[1][1]) if (downSamp[1][1] < np.inf) else np.inf) + ", "
        outputStr += str(int(downSamp[1][2]) if (downSamp[1][2] < np.inf) else np.inf) + "]"
        if (downSamp[0] == 0) and (len(iterErr2) == 1): checkParaRange = True
    if verbosity >= 3: print(outputStr)

    lambdaX = lambda t : funcXY[0](t, parXY[0])
    lambdaY = lambda t : funcXY[1](t, parXY[1])
    #checking if futher partition in paraRange is needed
    if checkParaRange == True:
        if verbosity >= 2: print("Checking local minima inside each section of paraRange:", paraRange)
        localOpt_tList = []
        for x, y in tqdm(np.array(dataXY).T, disable=(verbosity < 3)):
            distSquare = lambda t : _paraSquareDist(t, [lambdaX, lambdaY], [x, y], dataRangeXY)
            for paraIdx in range(len(paraRange)-1):
                bounds = (paraRange[paraIdx], paraRange[paraIdx+1])
                shgoResult = optimize.shgo(distSquare, [bounds], n=64, iters=3,sampling_method="sobol")
                localOpt_ts = []
                for shgo_t in shgoResult.xl:
                    if (abs(shgo_t[0] - bounds[0]) > 1E-6) and (abs(shgo_t[0] - bounds[1]) > 1E-6):
                        localOpt_ts.append(shgo_t[0])
                localOpt_ts.sort()
                if len(localOpt_ts) > 1: 
                    for localOpt_idx in range(len(localOpt_ts)-1):
                        localOpt_tList.append([localOpt_ts[localOpt_idx], localOpt_ts[localOpt_idx+1]])
        if len(localOpt_tList) > 0:
            if progressPlot == True: 
                _parametricFit2D_paraRangePlot(localOpt_tList, paraRange, len(dataXY[0]),\
                                               verbosity=verbosity, savePath=savePath)
            if verbosity >= 2: 
                print("WARNING: input paraRange can be affected local minima degeneracies\n")
                print("Please consider checking out the output plot (needs progressPlot == True), "
                      "and include an additional paraRange seperation point from in between "
                      "the left(blue) and right(red) local minima")
            if verbosity >= 3:
                print("The degenerate pairs in parametric variable are:")
                for localPair in localOpt_tList: print(" ", localPair)
                print("")
            if len(localOpt_tList) > len(dataXY[0])/1000.0: _parametricFit2_ContinueYesNo()
        else:
            if verbosity >= 2: print("Input paraRange has no local minima degeneracies. Good to go\n")
    #finding parametric variable t on the curve that has the shortest distance to the point
    (res2Sum, opt_ts, fullStat) = (0, [], [])
    for x, y in tqdm(np.array(dataXY).T, disable=(verbosity < 3)):
        opt_t, resSquare = getParaOfDataPt(parXY, funcXY, [x, y], dataRangeXY, paraRange)
        res2Sum += resSquare
        opt_ts.append(opt_t)
        fullStat.append([x, y, opt_t, lambdaX(opt_t), lambdaY(opt_t), resSquare])
    
    opt_ts = sorted(opt_ts)
    err2HeadTail = 0
    if ratioHeadTail[0] > 0:
        for i in range(max(1, int(len(dataXY[0])*ratioHeadTail[0]))):
            err2HeadTail += pow(lambdaX(opt_ts[i])\
                              - lambdaX(paraRange[0] +ratioHeadTail[0]*(paraRange[-1]-paraRange[0])),2)
            err2HeadTail += pow(lambdaY(opt_ts[i])\
                              - lambdaY(paraRange[0] +ratioHeadTail[0]*(paraRange[-1]-paraRange[0])),2)
    if ratioHeadTail[1] > 0:
        for i in range(max(1, int(len(dataXY[0])*ratioHeadTail[1]))):
            err2HeadTail += pow(lambdaX(opt_ts[-i-1])\
                              - lambdaX(paraRange[-1]-ratioHeadTail[1]*(paraRange[-1]-paraRange[0])),2)
            err2HeadTail += pow(lambdaY(opt_ts[-i-1])\
                              - lambdaY(paraRange[-1]-ratioHeadTail[1]*(paraRange[-1]-paraRange[0])),2)

    res2Ave      = res2Sum/len(dataXY[0])
    err2HeadTail = err2HeadTail/len(dataXY[0])
    if iterErr2 is not None: iterErr2[-1] = [iterErr2[-1][0], res2Ave, err2HeadTail]

    if verbosity >= 3:
        print("                                                average normalized square residual =",\
              _parametricFit2D_scientificStr(res2Ave, 10))
        print("  sample size =", len(dataXY[0]))
        print("  parX =", str([_parametricFit2D_scientificStr(par) for par in parXY[0]])\
                          .replace("'",""))
        print("  parY =", str([_parametricFit2D_scientificStr(par) for par in parXY[1]])\
                          .replace("'",""))
        print("  [min_t, max_t] =",str([_parametricFit2D_scientificStr(min(opt_ts)),\
                                        _parametricFit2D_scientificStr(max(opt_ts))]).replace("'", ""))
        print("  head_tail normalized square error =", _parametricFit2D_scientificStr(err2HeadTail,10))
        print("")
    return res2Ave + err2HeadTail
def _paraSquareDist(t, lambdaXY, dataPtXY, dataRangeXY, iterCounter=[]):
    if iterCounter != []: iterCounter[0] += 1
    normXYRatio = [1.0/(dataRangeXY[0][1]-dataRangeXY[0][0]),\
                   1.0/(dataRangeXY[1][1]-dataRangeXY[1][0])]
    return pow(normXYRatio[0]*(lambdaXY[0](t) - dataPtXY[0]), 2) +\
           pow(normXYRatio[1]*(lambdaXY[1](t) - dataPtXY[1]), 2)
def _parametricFit2D_paraRangePlot(localOpt_tList, paraRange, dataN, verbosity=1, savePath="."):
    pathlib.Path("/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME) )).mkdir(exist_ok=True)
    figName = "/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME, "paraRangeLocalMinDegen.png") )

    paraBinN = 200
    paraX = np.linspace(paraRange[0], paraRange[-1], paraBinN+1)[:-1]
    paraLeftHist  = np.histogram(np.array(localOpt_tList).T[0], bins=paraBinN,\
                                 range=[paraRange[0], paraRange[-1]])[0]
    paraRightHist = np.histogram(np.array(localOpt_tList).T[1], bins=paraBinN,\
                                 range=[paraRange[0], paraRange[-1]])[0]
 
    fig = plt.figure(figsize=(12, 9))
    matplotlib.rc("xtick", labelsize=16)
    matplotlib.rc("ytick", labelsize=16)
    gs = gridspec.GridSpec(1, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols): 
        ax.append(fig.add_subplot(gs[i]))
        ax[-1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.97)
    
    leftPlot  = ax[0].plot(paraX, paraLeftHist,  linewidth=2, color="blue", drawstyle="steps-mid")[0]
    rightPlot = ax[0].plot(paraX, paraRightHist, linewidth=2, color="red",  drawstyle="steps-mid")[0]
    ax[0].axhline(y=0, linewidth=2, color="black")
    plotTitle = "Local Minima Degeneracies within Parametric Ranges, "
    plotTitle+= str(len(localOpt_tList)) + "/" + str(dataN)
    ax[0].set_title(plotTitle, fontsize=20, y=1.03)
    ax[0].set_xlabel("parametric variable", fontsize=20)
    ax[0].set_ylabel("count", fontsize=20)
    ax[0].set_xlim(paraRange[0], paraRange[-1])
    ax[0].legend([leftPlot, rightPlot], ["left", "right"], loc="upper right", fontsize=20)

    plt.savefig(figName)
    plt.close(fig)
    if verbosity >= 1: print("Saving plot:\n ", figName)
#stackoverflow.com/questions/3041986
def _parametricFit2_ContinueYesNo(default=False):
    validity = {"": default, "yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = ""
    if   default == True:  prompt = "Continue processing? [Y/n] "
    elif default == False: prompt = "Continue processing? [y/N] "
    else: raise ValueError("_parametricFit2_ContinueYesNo: invalid default vale: "+default)
   
    contBool = None
    while True:
        sys.stdout.write(prompt)
        choice = input().lower()
        if choice in validity:
            contBool = validity[choice]
            break
        else: 
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
    if contBool == False: sys.exit(0)
def _parametricFit2D_progressPlot(parXYFit, funcXY, dataXY, dataRangeXY, paraRange=[-1.0, 1.0],\
                                  verbosity=1, optIdx=None, bootIdx=None, iterErr2s=None,\
                                  downSamp=None, saveProgress=False, savePath="."):
    if downSamp[1][0] == "Hess": return
    pathlib.Path("/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME) )).mkdir(exist_ok=True)
    figName = "/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME, "progressPlot_Opt.png") )
    if (len(iterErr2s) == 1) and (downSamp[1][2] == 0): figName = figName.replace("Opt", "Init")
    elif downSamp[1][0] == "Boot": figName = figName.replace("Opt", "Boot")
    
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

    binN = int(max(10, min(500, 3*min( 1.0*len(dataXY[0]), math.sqrt(downSamp[1][1]) ))))
    fitT = np.linspace(paraRange[0], paraRange[-1], binN+1)[:-1]
    fitCurveX = funcXY[0](fitT, parXYFit[0])
    fitCurveY = funcXY[1](fitT, parXYFit[1])

    fig = plt.figure(figsize=(12, 9))
    matplotlib.rc("xtick", labelsize=16)
    matplotlib.rc("ytick", labelsize=16)
    gs = gridspec.GridSpec(1, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols): 
        ax.append(fig.add_subplot(gs[i]))
        ax[-1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")

    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.12, right=0.98)
    ax[0].plot(iterations, res2Aves, color="blue", linewidth=2, marker="o", markersize=5)
    scatter = ax[0].scatter(bootIters, bootRes2Aves, s=120, color="orange")
    ax[0].set_title("Normalized Residual Square Average at Each DownSampling", fontsize=24, y=1.03)
    ax[0].set_xlabel("iterations", fontsize=20)
    ax[0].set_ylabel("residual", fontsize=20)
    xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
    ax[0].set_xlim(0, 1.06*xlim[1])
    ax[0].text(minIter, ylim[0]+0.015*(ylim[1]-ylim[0]), _parametricFit2D_scientificStr(res2Min, 3),\
               color="blue", fontsize=14, weight="bold")

    if downSamp[1][0] == "Boot":
        ax[0].legend([scatter], ["bootstrap"], scatterpoints=1, loc="upper right", fontsize=16)
    plt.savefig(figName)
    plt.cla()

    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=1.0)
    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[0].hist2d(*dataXY, bins=binN, cmin=1, cmap=cmap, range=dataRangeXY)
    cb = fig.colorbar(hist[3], ax=ax[0]).mappable
    ax[0].plot(fitCurveX, fitCurveY, linewidth=3, color="red")
    plotTile =  "DownSampling[" + str(downSamp[0]) + "]="
    plotTile += "[\"" + downSamp[1][0] + "\","
    plotTile += str(int(downSamp[1][1]) if (downSamp[1][1] < np.inf) else np.inf) + ","
    plotTile += str(int(downSamp[1][2]) if (downSamp[1][2] < np.inf) else np.inf) + "], "
    plotTile += "Iter=" + str(iterErr2s[-1][-1][0]) + ", NormSqErr="
    plotTile += _parametricFit2D_scientificStr(iterErr2s[-1][-1][1])
    ax[0].set_title(plotTile, fontsize=20, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_xlim(*dataRangeXY[0])
    ax[0].set_ylim(*dataRangeXY[1])
    cb.set_clim(0.0, max(5.0, cb.get_clim()[1]))
    
    figDSName = figName.replace("progressPlot", "progressPlotDS["+str(s)+"]")
    plt.savefig(figDSName)
    plt.close(fig)
    if verbosity >= 1: print("Saving plots:\n ", figName, "\n ", figDSName)
def _parametricFit2D_roundSig(val, sigFig=3):
    if val == 0: return val;
    return round(val, sigFig-int(np.floor(np.log10(abs(val))))-1);
def _parametricFit2D_scientificStr(val, sigFig=3):
    valStr = ""
    if val == 0:
        valStr = "0.0"
    elif abs(np.floor(np.log10(abs(val)))) < sigFig:
        valStr = str(_parametricFit2D_roundSig(val, sigFig=sigFig))
    else:
        valStr = "{:." + str(sigFig-1) + "e}"
        valStr = valStr.format(val)
        valStr = valStr.replace("e+0", "e+")
        valStr = valStr.replace("e+", "e")
        valStr = valStr.replace("e0", "")
        valStr = valStr.replace("e-0", "e-")
    return valStr








#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
def example_parametricFit2D():
    paraRangeOrig = [-math.pi/4, 3*math.pi/2]
    def example_curve(t):
        x = 2*math.sin(t + math.pi/5) + 0.5*t
        y = 1.2*math.cos(t + math.pi/5) + 0.8*math.sin(t + math.pi/5)
        return x, y
    def polyFunc(x, coefs):
        if np.isscalar(coefs) == True:
            raise TypeError("polyFunc: coefs must be a 1D array/list")
        result = 0
        for i, c in enumerate(coefs): result += c*np.power(x, i)
        return result
    def truncateColorMap(cmap, lowR, highR):
        cmapNew = matplotlib.colors.LinearSegmentedColormap.from_list(\
                  "trunc({n}, {l:.2f}, {h:.2f})".format(n=cmap.name, l=lowR, h=highR),\
                  cmap(np.linspace(lowR, highR, 1000)))
        return cmapNew

    #sample setup
    binN    = 1000
    sampleN = 3000
    rd.seed(1)
    noiseSig = 0.2
    paraT = np.linspace(*paraRangeOrig, binN+1)[:-1]
    curveX = [example_curve(t)[0] for t in paraT]
    curveY = [example_curve(t)[1] for t in paraT]
    data = [[], []]
    for i in range(sampleN):
        paraVal = rd.uniform(*paraRangeOrig)
        if paraVal > (paraRangeOrig[1] - (paraRangeOrig[1]-paraRangeOrig[0])/10.0):
            paraVal = rd.uniform(*paraRangeOrig)
        x, y = example_curve(paraVal)
        x, y = rd.gauss(x, noiseSig), rd.gauss(y, noiseSig)
        data[0].append(x), data[1].append(y)

    ###################################################################################################
    #parametric fit
    ###heavily depends on initial conditions, can test with downSampling=[*[[100, 1]]*1] first
    ###may need to get a systems of linear equation solver for both x&y to get a few points correct
    rangeXY       = [[-1.5, 3.5], [-2.2, 2.2]]
    paraRange     = [-1.0, -0.6, 1.0]
    ratioHeadTail = [0.01, 0.01]
    randSeed      = 0
    savePlot, saveProg, readProg = True, True, True  

    funcXY = [polyFunc, polyFunc]
    initX = [2.3, 1.1233, -5.24659, -1.8233, 2.84659]
    initY = [-0.2, 3.71818, 1.06364, -3.21818, -0.363636]
    downSampling=[  ["Opt",  1000,   0,      None, None],\
                  *[["Opt",  1000,   1000,   None, None]]*2,\
                    ["Opt",  np.inf, np.inf, None, None],\
                  *[["Boot", 1000,   np.inf,    None, None]]*30]
    #downSampling = getInstantPlotDownSampling(downSampling=downSampling); saveProg=False 

    parXYOpt, parXYErr, res2AveVal = paraLeastSquare([initX, initY], funcXY, data, rangeXY,\
                                                     paraRange=paraRange, ratioHeadTail=ratioHeadTail,\
                                                     randSeed=randSeed, downSampling=downSampling,\
                                                     progressPlot=savePlot, saveProgress=saveProg,\
                                                     readProgress=readProg)
    '''
    savePath = str(pathlib.Path().absolute()) 
    pickleName = "/".join( (savePath, _PARAMETRICFIT2D_SAVEDIRNAME, "savedProgress.pickle") )
    progressDict = {}
    with open(pickleName, "rb") as handle: progressDict = pickle.load(handle)
    parXYOpt, parXYErr, res2AveVal = [None, None], [None, None, None], None
    parXYOpt[0] = progressDict["parXOpt"]
    parXYOpt[1] = progressDict["parYOpt"]
    parXYErr[0] = progressDict["parXErr"] 
    parXYErr[1] = progressDict["parYErr"] 
    parXYErr[2] = progressDict["parErrCovMatrix"] 
    res2AveVal  = progressDict["iterErr2"][-1][-1][1] 
    '''

    fitT = np.linspace(paraRange[0], paraRange[-1], binN+1)[:-1]
    fitCurveX = funcXY[0](fitT, parXYOpt[0])
    fitCurveY = funcXY[1](fitT, parXYOpt[1])
    print("Results from paraLeastSquare():")
    print("  average normalized square residual =", _parametricFit2D_scientificStr(res2AveVal))
    print("  parXOpt =", str([_parametricFit2D_scientificStr(par) for par in parXYOpt[0]])\
                         .replace("'", ""))
    print("  parYOpt =", str([_parametricFit2D_scientificStr(par) for par in parXYOpt[1]])\
                         .replace("'", ""))
    print("  parXErr =", str([_parametricFit2D_scientificStr(err) for err in parXYErr[0]])\
                         .replace("'", ""))
    print("  parYErr =", str([_parametricFit2D_scientificStr(err) for err in parXYErr[1]])\
                         .replace("'", ""))
    #plot
    fig = plt.figure(figsize=(12, 18))
    matplotlib.rc("xtick", labelsize=16)
    matplotlib.rc("ytick", labelsize=16)
    gs = gridspec.GridSpec(2, 1)
    ax = []
    for i in range (gs.nrows*gs.ncols):
        ax.append(fig.add_subplot(gs[i]))
        ax[-1].ticklabel_format(style="sci", scilimits=(-2, 2), axis="both")
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.095, right=0.98)

    cmap = truncateColorMap(plt.get_cmap("jet"), 0.0, 0.92)
    hist = ax[0].hist2d(*data, bins=int(binN/10.0), cmin=1, cmap=cmap, range=rangeXY)
    cb = fig.colorbar(hist[3], ax=ax[0]).mappable
    ax[0].plot(fitCurveX, fitCurveY, linewidth=3, color="red") 
    ax[0].set_title("Parametric Fitting the Curve", fontsize=28, y=1.03)
    ax[0].set_xlabel("x", fontsize=20)
    ax[0].set_ylabel("y", fontsize=20)
    ax[0].set_aspect("equal")
    ax[0].set_xlim(*rangeXY[0])
    ax[0].set_ylim(*rangeXY[1])

    errCurveN = 100
    print("Sampling the fit curve from the parameter standard errors...", end="\r" )
    parXN, parYN = len(parXYOpt[0]), len(parXYOpt[1])
    parXYOptSamps = np.random.multivariate_normal(parXYOpt[0]+parXYOpt[1], parXYErr[2], size=errCurveN)
    for parXYOptSamp in parXYOptSamps:
        sampFitCurveX = funcXY[0](fitT, parXYOptSamp[:parXN])
        sampFitCurveY = funcXY[1](fitT, parXYOptSamp[parXN:]) 
        plotFitted = ax[1].plot(sampFitCurveX, sampFitCurveY, linewidth=3, color="red", alpha=0.1)[0]
    plotGiven = ax[1].plot(curveX, curveY, linewidth=3, color="blue")[0]
    ax[1].set_title("Parametric Curve: Given vs Fitted (sampled from fit err)", fontsize=24, y=1.03)
    ax[1].set_xlabel("x", fontsize=20)
    ax[1].set_ylabel("y", fontsize=20)
    ax[1].set_aspect("equal")
    ax[1].set_xlim(rangeXY[0][0], rangeXY[0][1]+1.18)
    ax[1].set_ylim(*rangeXY[1])
    ax[1].legend([plotGiven, plotFitted], ["given", "fitted"], loc="upper right", fontsize=20)

    figName = "paraFitCurve2D.png"
    plt.savefig(figName)
    plt.close(fig)
    print("Saving the following file:                                         \n ", figName)


if __name__ == "__main__":
    print("###################################################################################Begin\n")
    #printSavedProgress("/".join( (_PARAMETRICFIT2D_SAVEDIRNAME, "/savedProgressDS[0].pickle") ))
    example_parametricFit2D()
    print("\n###################################################################################End\n")



