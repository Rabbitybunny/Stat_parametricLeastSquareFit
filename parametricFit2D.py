import os, sys, math
import numpy as np
import random as rd
import pickle
from scipy import optimize
from tqdm import tqdm



def paraLeastSquare(parXYinit, funcXY, dataXY, dataRangeXY, ratioHeadTail=0.01, randSeed=None, iterRef=[],\
                    downSampling=[*[[100, 100]]*3, *[[1000, 100]]*5, [1e12, 100]]): 
    rd.seed(randSeed)

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

    if len(downSampling) != 0:
        dataXList = np.array(dataXInput).tolist()
        dataYList = np.array(dataYInput).tolist()
        parXforOpt = parXYinit[0].copy()
        parYforOpt = parXYinit[1].copy()
        for sampStat in downSampling:
            dataNRatio = 1.0
            if dataN > sampStat[0]:
                sampledData = rd.choices(list(zip(dataXList, dataYList)), k=int(sampStat[0]))
                dataXforOpt = [d[0] for d in sampledData]
                dataYforOpt = [d[1] for d in sampledData]
                dataNRatio = sampStat[0]/dataN
            else:
                dataXforOpt = dataXList.copy()
                dataYforOpt = dataYList.copy()
            err2Sum = lambda par : dataNRatio*paraErrorSquareSum([par[:parXN], par[parXN:]], funcXY,\
                                                                 [dataXforOpt, dataYforOpt],\
                                                                 ratioHeadTail=ratioHeadTail,\
                                                                 iterRef=iterRef, dataNRatioDisp=dataNRatio)    
            paraFitResult = optimize.minimize(err2Sum, [*parXforOpt, *parYforOpt], method="Nelder-Mead",\
                                              options={"maxiter":sampStat[1]})
            parXforOpt, parYforOpt = paraFitResult.x[:parXN], paraFitResult.x[parXN:]
        return parXforOpt, parYforOpt
    else:
        err2Sum = lambda par : paraErrorSquareSum([par[:parXN], par[parXN:]], funcXY, [dataXInput, dataYInput],\
                                                  ratioHeadTail=ratioHeadTail, iterRef=iterRef)
        paraFitResult = optimize.minimize(err2Sum, [*parX, *parY], method="Nelder-Mead")
        return paraFitResult.x[:parXN], paraFitResult.x[parXN:]
def paraDistSquare(t, funcXY, dataXY):
    curveX, curveY = funcXY[0](t), funcXY[1](t)
    return pow(curveX - dataXY[0], 2) + pow(curveY - dataXY[1], 2)
def paraErrorSquareSum(parXY, funcXY, dataXY, ratioHeadTail=0.0, iterRef=[], dataNRatioDisp=1.0):
    lambdaX = lambda t : funcXY[0](t, parXY[0])
    lambdaY = lambda t : funcXY[1](t, parXY[1])

    if len(dataXY[0]) != len(dataXY[1]):
        print("ERROR: paraErrorSquareSum: lengths of dataX and dataY don't match")
        sys.exit(0)

    (err2Sum, opt_ts) = (0, [])
    for x, y in tqdm(np.array(dataXY).T, disable=(len(iterRef) == 0)):
        distSquare = lambda t : math.sqrt(paraDistSquare(t, [lambdaX, lambdaY], [x, y]))
        opt_t = optimize.minimize_scalar(distSquare, method="bounded", bounds=(0.0, 1.0))
        err2Sum += distSquare(opt_t.x)
        opt_ts.append(opt_t.x)

    err2HeadTail = 0
    if ratioHeadTail != 0:
        countHeadTail = max(1, int(len(dataXY[0])*ratioHeadTail))
        opt_ts = sorted(opt_ts)
        for i in range(countHeadTail):
            err2HeadTail += pow(lambdaX(opt_ts[i])   -lambdaX(ratioHeadTail),     2) 
            err2HeadTail += pow(lambdaY(opt_ts[i])   -lambdaY(ratioHeadTail),     2)
            err2HeadTail += pow(lambdaX(opt_ts[-i-1])-lambdaX(1.0-ratioHeadTail), 2)
            err2HeadTail += pow(lambdaY(opt_ts[-i-1])-lambdaY(1.0-ratioHeadTail), 2)

    if len(iterRef) != 0:
        iterRef[0] += 1
        print("opt iter "+str(iterRef[0])+":                                 square error =",err2Sum/dataNRatioDisp)
        print("  parX =", [*parXY[0]])
        print("  parY =", [*parXY[1]])
        print("  data number =", len(dataXY[0]))
        print("  (min_t, max_t) =", (opt_ts[0], opt_ts[-1]))
        print("  head_tail square error =", err2HeadTail)
    return err2Sum + err2HeadTail






















if __name__ == "__main__":
    pass




