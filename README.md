# 2D Parametric Fit Using Least Square

In this example, the parameters are the &mu; and &sigma; sampling from a gaussian with a sample size of 30 and, <br/>
&ensp;&ensp;&mu; = 4.8 and &sigma; = 0.6. <br/>
For this simple test, the profile likelihood isn't quite necessary as the maximum likelihood method can do it just fine.

The code runs on python3 with additional packages:

    pip3 install scipy
    pip3 install tqdm
    python3 parametricFit2D.py
The code outputs the following image:

<img src="https://github.com/Rabbitybunny/Stat_profileLikelihood/blob/main/gausProfileNoNoise.png" width="630" height="490">
