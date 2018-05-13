import numpy as np
import math 
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from sklearn import datasets, linear_model
import matplotlib.animation as anim
import statsmodels.regression.linear_model as lm
import statsmodels.api as sm

# The first two functions are created based on the code from the following link: 
# https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.
    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.
    """
    try:
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N
    
    except:
        print("Invalid covariance")
        return
        
def BVN(meanX, meanY, varX, varY, cov, x):
    meanY = float(meanY)
    meanX = float(meanX)
    varY = float(varY)
    varX = float(varX)
    cov = float(cov)

    print ("Observe how the shape of the sampled points (1st plot) and the 3D Bell curve of the joint density (2nd plot)" + \
         " changes as each paremeter is shifted.")
    
    # Our 2-dimensional distribution will be over variables X and Y
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 3, N)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([meanX, meanY])
    Sigma = np.array([[varX , cov], [cov,  varY]])

    # Randomly sample 200 points for the scatterplot
    samples = np.random.multivariate_normal(mu, Sigma, 100)
    plt.scatter(samples[:,0], samples[:,1])
    plt.plot(meanX, meanY, 'ro', label='mean')
    plt.legend()
    plt.title("200 samples from the distribution")
    plt.xlim(-3, 3)
    plt.xlabel("X")
    plt.ylim(-3, 3)
    plt.ylabel("Y")
    plt.show()
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)

    # Create a surface plot and projected filled contour plot under it.
    fig = plt.figure()
    fig.set_size_inches(9, 7)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                    cmap=cm.viridis)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Joint Density")
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.7, cmap=cm.viridis)

    # Adjust the limits, ticks and view angle
    ax.set_zlim(-0.8,1.0)
    ax.set_zticks(np.linspace(0,1.0,5))
    ax.view_init(27, -85)
    plt.title("Joint Density of Bivariate Normal Distribution")
    plt.show()
    
    corr = np.corrcoef(samples[:, 0], samples[:, 1])[0][1]
    yy = np.linspace(-2, 2, 100)
    yyy = [ 1 / (2 * math.pi * math.sqrt(varX) * math.sqrt(varY) * math.sqrt(1-corr**2)) * \
           np.exp(-((x - meanX)**2 / varX - 2*corr*(x-meanX)*(y-meanY) / (math.sqrt(varX) * math.sqrt(varY)) + (y - meanY)**2 / varY)) for y in yy]
    plt.plot(yy, yyy)
    plt.title("marginal distribution of y, when x=" + str(x) + " is fixed")
    plt.xlabel("y")
    plt.ylabel("P(y | x)")
    plt.show()
    
def CLT(mean, N):
    
    ld = 1.0/mean
    
    print ("First, the data would be sampled from the following exponential distribution.")
    print ("The mean of your choice is shown with the vertical green line.")
    x = np.linspace(0, 7, 100)
    y = ld * np.exp(-ld * x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("exponential distribution with lambda: " + str(round(ld, 2)))
    plt.axvline(x=1 / float(ld), color='g', label='mean(1/lambda) = ' + str(round(1/float(ld), 2)))
    plt.legend()
    plt.show()    
    
    rds = np.random.exponential(mean, size=300)
    avg3s = []
    for i in range(100):
        avg3s.append(np.mean(rds[3*i:3*(i+1)+1]))

    print ("The histogram of the sample averages (by groups of 3) can be compared with the original distribution.")
    print ("300 data points were sampled in total, and each group of 3 produced an average, for example:")
    for i in range(4):
        print ("samples "+str(3*i+1)+"~"+str(3*(i+1))+": ", rds[3*i:3*(i+1)], "-> average:", round(avg3s[i], 2))
    print ("...")
    print ("samples 298~300:", rds[297:], "-> average:", round(avg3s[-1], 2))        

    print ("\nAnd these are the 100 sample averages (of 3) that will be drawn below:")
    print (list(map(lambda x: round(x, 2), avg3s)))
    
    rds = np.random.exponential(mean, size=100*N)
    avgNs = []
    for i in range(100):
        avgNs.append(np.mean(rds[N*i:N*(i+1)]))         

    print ("\nYou can also test with different Ns by using the slidebar. With the current N=" + str(N) + \
          ", these are the 100 sample averages (of " + str(N) + ") that will be drawn below:")
    print (list(map(lambda x: round(x, 2), avgNs)))
    
    plt.subplots(1, 2, figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    samplemean3 = np.mean(avg3s)
    samplevar3 = np.var(avg3s)
    x3 = np.linspace(0, 7, 100)
    y3 = [ 1 / math.sqrt(2 * math.pi * samplevar3) * math.exp( - (xx-samplemean3)**2 / (2 * samplevar3) ) for xx in x3 ]
    plt.plot(x3, y3, label="normal distribution for averages of 3")        
    plt.hist(avg3s, label="averages of 3 samples", normed=True)
    plt.legend()
    plt.xlabel("values of samples/averages")
    plt.ylabel("relative frequency")

    plt.subplot(1, 2, 2)
    samplemeanN = np.mean(avgNs)
    samplevarN = np.var(avgNs)
    xN = np.linspace(0, 7, 100)
    yN = [ 1 / math.sqrt(2 * math.pi * samplevarN) * math.exp( - (xx-samplemeanN)**2 / (2 * samplevarN) ) for xx in xN ]
    plt.plot(xN, yN, label="normal distribution for averages of N")    
    plt.hist(avgNs, label="averages of N=" + str(N) + " samples", normed=True)
    plt.legend()
    plt.xlabel("values of averages")
    plt.ylabel("relative frequency")
    plt.show()
    
    print ("The histograms of 3-averages and N-averages are shown above.")    
    print ("Observe that although the data were sampled from an exponential distribution,")
    print ("the distribution of averages look closer to a normal distribution(bell curve) as N gets larger.")

def HT(mean, hypothesis, pvalue):
    shape = math.sqrt(mean)
    scale = math.sqrt(mean)
    truevar = shape*scale*scale
    N=100
    
    sample1 = np.random.gamma(shape=shape, scale=scale, size=N)
    sample2 = np.random.gamma(shape=scale, scale=scale, size=N)
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    
    print ("Below are the histograms of two sets of samples from a gamma distribution with your chosen mean="+str(mean))
    
    f, axarr = plt.subplots(1, 2, figsize=(20,5), sharey=True)
    axarr[0].hist(sample1)
    axarr[0].axvline(x=mean1, color='b', label='sample mean='+str(mean1))
    axarr[1].hist(sample2)
    axarr[1].axvline(x=mean2, color='b', label='sample mean='+str(mean2))
    axarr[0].legend()
    axarr[1].legend()
    axarr[0].set_xlabel('sample values')
    axarr[0].set_ylabel('sample counts')
    axarr[1].set_xlabel('sample values')
    axarr[1].set_ylabel('sample counts')    
    plt.show()
    
    nmean1 = (mean1 - hypothesis) / (math.sqrt(truevar) / math.sqrt(N))
    nmean2 = (mean2 - hypothesis) / (math.sqrt(truevar) / math.sqrt(N))
    
    p1 = stats.norm(0, 1).ppf(pvalue/2.0)
    p2 = stats.norm(0, 1).ppf(1-pvalue/2.0)  
    
    print ("normalized test statistics for the two sample means (blue line): "+str(nmean1)+", "+str(nmean2))
    print ("p-value threshold for a standard normal distribution (green): "+str(p1)+", "+str(p2))
    
    plt.figure(figsize=(12, 5))
    
    x = np.linspace(-3.5, 3.5, 300)
    y = [ 1 / math.sqrt(2 * math.pi * 1) * math.exp( - (xx-0)**2 / (2 * 1) ) for xx in x ]
    plt.plot(x, y)
    plt.axvline(x=0, color='r', label='mean=0 of standard normal')
    plt.axvline(x=nmean1, color='b', label='test statistics based on sample')
    plt.axvline(x=nmean2, color='b')
    plt.xlabel("x")
    plt.ylabel("P(x)")

    plt.axvline(x=p1, color='g', label='p-value threshold (p='+str(pvalue)+')')
    plt.axvline(x=p2, color='g')       
    plt.legend()
    plt.show()
    
def IV(covXZ=0.7):
    N=500
    alpha = 0.5
    beta = 2.0
    mean = [0, 0, 0]
    cov = [[1, covXZ, 0.7], [covXZ, 1, 0], [0.7, 0, 1]]  # X, Z, eps
    X, Z, eps = np.random.multivariate_normal(mean, cov, N).T

    Y = [alpha + beta * x + e for x, e in zip(X, eps)]
    
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    regr = linear_model.LinearRegression()
    xx = np.linspace(-3, 3, 300)
    
    
    # new addition
    print ('While fitting on Y~X would produce the best fit for the scatterplot, the line misses the true beta=2, ' + \
          'as variable Z that is correlated with X is ignored')
    regr.fit(X.reshape(N, 1), Y.reshape(N, 1))
    Yhat = regr.predict(X.reshape(N, 1))
    xy = [regr.intercept_ + x * regr.coef_[0] for x in xx] 
    truexy = [alpha + x * beta for x in xx] 
    plt.scatter(X, Y)
    plt.plot(xx, xy, 'b', label='slope: ' + str(round(regr.coef_[0][0], 3)))
    plt.plot(xx, truexy, 'r', label='true slope beta=2')
    plt.title('fit Y against X, ignoring Z')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim((-3.5, 3.5))
    plt.legend()
    plt.show()
    
    print ("One method of estimating the true beta with the Instrumental Variable Z is to first fit X~Z to get Xhat, " + \
           "and then fit Y~Xhat to get Yhat. The resulting slope would correctly estimate the true beta=2")
    
    f, axarr = plt.subplots(1, 2, figsize=(18,6), sharey=True)

    regr.fit(Z.reshape(N, 1), X.reshape(N, 1))
    Xhat = regr.predict(Z.reshape(N, 1))
    zx = [regr.intercept_ + x * regr.coef_[0] for x in xx] 
    ZXbeta = regr.coef_[0][0]
    axarr[0].scatter(Z, X)
    axarr[0].plot(xx, zx, 'C1', label='slope: ' + str(round(regr.coef_[0][0], 3)))
    axarr[0].set_xlabel("Z")
    axarr[0].set_ylabel("X")    
    axarr[0].set_title("stage 1: fit X against Z")
    axarr[0].legend()
    #plt.show()    
    
    regr.fit(Xhat.reshape(N, 1), Y.reshape(N, 1))
    Yhat = regr.predict(Xhat.reshape(N, 1))
    yx = [regr.intercept_ + x * regr.coef_[0] for x in xx] 
    axarr[1].scatter(Xhat, Y)
    axarr[1].plot(xx, yx, 'C1', label='slope: ' + str(round(regr.coef_[0][0], 3)))
    axarr[1].plot(xx, truexy, 'r', label='true slope beta=2')
    axarr[1].set_xlabel("Xhat")
    axarr[1].set_ylabel("Y")    
    axarr[1].set_title("stage 2: fit Y against Xhat from stage 1")
    axarr[1].legend()
    plt.show()        
    print ("Observe that the resulting fitted line from stage 2 closely estimates the true beta=2.\n")

def LLN(mean, N):
    
    ld = 1.0/mean
    
    print ("First, the data would be sampled from the following exponential distribution.")
    print ("The mean of your choice is shown with the vertical green line.")
    x = np.linspace(0, 7, 100)
    y = ld * np.exp(-ld * x)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("P(x)")
    plt.title("exponential distribution with lambda: " + str(round(ld, 2)))
    plt.axvline(x=1 / float(ld), color='g', label='mean(1/lambda) = ' + str(round(1/float(ld), 2)))
    plt.legend()
    plt.show()

    avgs = []
    rds = []
    for i in range(1, N+1):
        rd = round(np.random.exponential(1/float(ld), 1)[0], 2)
        rds.append(rd)
        avgs.append(np.mean(rds))
    
    Ns = []
    sampleavgs = []
    
    print ("These values below illustrate how the average changes as more samples are added")
    if N >= 1:
        Ns.append(1)
        sampleavgs.append(np.mean([rds[0]]))
        print ("first 1 sample:", [rds[0]], "-> average:", round(np.mean([rds[0]]), 2))    
        if N >= 5:
            Ns.append(5)
            sampleavgs.append(np.mean(rds[:5]))    
            print ("first 5 samples:", rds[:5], "-> average:", round(np.mean(rds[:5]), 2))
            if N >= 10:
                Ns.append(10)
                sampleavgs.append(np.mean(rds[:10])) 
                print ("first 10 samples:", rds[:10], "-> average:", round(np.mean(rds[:10]), 2))
                if N >= 100:
                    Ns.append(100)
                    sampleavgs.append(np.mean(rds[:100])) 
                    print ("first 100 samples: average:", round(np.mean(rds[:100]), 2))
                    if N >= 200: 
                        Ns.append(200)
                        sampleavgs.append(np.mean(rds[:200])) 
                        print ("first 200 samples: average:", round(np.mean(rds[:200]), 2))                        
    
    print ("\nThe two plots below display the changing trend of the averages as N gets larger.")
    print ("The example averages shown above are marked with red dots.")
    print ("The left plot is a zoomed-in version to display the red dots more clearly, the right plot shows the full trend.")
    print ("Observe that as N gets very large, the average nearly converges to the true mean indicated by the green line.")
    f, axarr = plt.subplots(1, 2, figsize=(20,5), sharey=True)
    
    axarr[0].plot(Ns, sampleavgs, 'ro')
    axarr[0].plot(range(1, max(Ns)+1), avgs[:max(Ns)])
    axarr[0].set_xlabel("N")
    axarr[0].set_ylabel("average")    
    
    axarr[1].plot(Ns, sampleavgs, 'ro')
    axarr[1].plot(range(1, N+1), avgs)
    axarr[1].set_xlim(0, 1500)
    axarr[1].axhline(y=round(1/float(ld), 2), color='g', label='mean(1/lambda) = ' + str(round(1/float(ld), 2)))
    axarr[1].legend()
    axarr[1].set_xlabel("N")
    axarr[1].set_ylabel("average")
    plt.show()

def OVB(beta2, cov, angle=280, height=30):
    A = beta2
    alpha = 0.5
    beta1 = 2.0
    N = 500
    mean = [0, 0]
    cov = [[1, cov], [cov, 1]]
    np.random.seed(123)
    sampleX, sampleZ = np.random.multivariate_normal(mean, cov, N).T

    sampleY = np.array([alpha + x * beta1 + z * beta2 + np.random.normal(0, 1)*4 for x, z in zip(sampleX, sampleZ)])

    fullregr = linear_model.LinearRegression()
    fullregr.fit(np.transpose(np.vstack((sampleX, sampleZ))), sampleY.reshape(N, 1))   
    
    smallregr = linear_model.LinearRegression()
    smallregr.fit(sampleX.reshape(N, 1), sampleY.reshape(N, 1))     
    
    fullxx = np.linspace(-3, 3, 300)
    fullzz = np.linspace(-3, 3, 300)
    fullyy = [fullregr.intercept_ + x * fullregr.coef_[0][0] for x in fullxx]

    smallxx = np.linspace(-3, 3, 300)
    smallzz = np.linspace(-3, 3, 300)
    smallyy = [smallregr.intercept_ + x * smallregr.coef_[0][0] for x in smallxx]        
    
    fig = plt.figure(figsize=(20, 7))    
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot3D(fullxx, fullzz, zs=np.array(fullyy).flatten(), c='r', label="full model")
    ax1.scatter3D(sampleX, sampleZ, sampleY, label="sample points")
    ax1.view_init(height, azim=angle)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y")
    ax1.set_title("3D plot of the full model")
    ax1.legend()    
    
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.plot(fullxx, fullyy, 'r', label="full model")
    ax2.scatter(sampleX, sampleY)
    ax2.set_title("2D plot (Y~X) of the full model")
    ax2.legend()
    plt.show()

    
    fig = plt.figure(figsize=(20, 7))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.view_init(height, azim=angle)
    ax1.plot3D(smallxx, smallzz, zs=np.array(smallyy).flatten(), c='gray', label="reduced model")
    ax1.plot3D(fullxx, fullzz, zs=np.array(fullyy).flatten(), c='r', label="full model")
    ax1.scatter3D(sampleX, sampleZ, sampleY, label="sample points")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("Y")
    ax1.set_title("3D plot of the reduced model")
    ax1.legend()

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.plot(smallxx, smallyy, 'gray', label="reduced model")
    ax2.plot(fullxx, fullyy, 'r', label="full model")
    ax2.scatter(sampleX, sampleY)
    ax2.set_title("2D plot (Y~X) of the reduced model")
    ax2.legend()    
    plt.show()

    print ("\nTrue beta1 of the full model : 2")
    print ("beta1hat from the full model : "+str(fullregr.coef_[0][0]))
    print ("beta1hat from the reduced model, with Z omitted (N=500, graphed above): " + str(smallregr.coef_[0][0]))
    
    N=5000
    sampleX, sampleZ = np.random.multivariate_normal(mean, cov, N).T
    sampleY = np.array([alpha + x * beta1 + z * beta2 + np.random.normal(0, 1)*4 for x, z in zip(sampleX, sampleZ)])  
    smallregr = linear_model.LinearRegression()
    smallregr.fit(sampleX.reshape(N, 1), sampleY.reshape(N, 1))  
    print ("beta1hat from the reduced model, with Z omitted (N=5000, not graphed): " + str(smallregr.coef_[0][0]))
    print ("-> Thus, the bias is maintained even with very large Ns")

def SLR(N):

    # generate random 2d data
    beta0 = 0.5
    beta1 = 2.0

    print ("The set of N=" +str(N)+ " sampled points shown below are sampled based on Y=2X+0.5 with noise")
    print ("The orange line indicates a fitted regression line based on the sample. It may not exactly be equal to Y=2X+0.5")
    
    sampleX = np.random.rand(N)
    sampleY = np.array([beta0 + x * beta1 + -2 + np.random.rand(1)*4 for x in sampleX])
    sampleX = sampleX.reshape(N, 1)
    sampleY = sampleY.reshape(N, 1)
    plt.plot(sampleX, sampleY, 'o')
    
    regr = linear_model.LinearRegression()
    regr.fit(sampleX.reshape(N, 1), sampleY.reshape(N, 1))

    xx = np.linspace(0, 1, 300)
    yy = [regr.intercept_ + x * regr.coef_[0] for x in xx]
    plt.plot(xx, yy)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    betahats = []
    meanbetahats = []
    for i in range(2, N):
        sX = sampleX[:i+1]
        sY = np.array([beta0 + x * beta1 + -1 + np.random.rand(1)*2 for x in sX])
        xbar = np.mean(sX)
        ybar = np.mean(sY)

        cov = np.cov(np.vstack((sX.reshape(1, i+1), sY.reshape(1, i+1))))
        betahat = cov[0][1] / cov[0][0]
        betahats.append(betahat)
        meanbetahats.append(np.mean(betahats))

        
    print ("The plot below displays how betahat converges to the true beta=2,")
    print ("as more and more samples are used to fit a regression line.")
    print ("Thus, the estimator betahat is consistent.")
        
    plt.plot(meanbetahats)
    plt.xlabel("N (number of samples used)")
    plt.ylabel("betahat of fitted line")
    plt.axhline(y=2, color='r')
    plt.show()

    
    betahats = []
    for i in range(100):
    
        sampleX = np.random.rand(N)
        sampleY = np.array([beta0 + x * beta1 + -2 + np.random.rand(1)*4 for x in sampleX])
        regr = linear_model.LinearRegression()
        regr.fit(sampleX.reshape(N, 1), sampleY.reshape(N, 1))
        betahats.append(regr.coef_[0][0])
    
    var = np.var(betahats)
    xx = np.linspace(min(betahats), max(betahats), 300)
    yy = [ 1 / math.sqrt(2 * math.pi * var) * math.exp( - (x-2.0)**2 / (2 * var) ) for x in xx ]
    
    print ("Now, N="+str(N) + " points are sampled 100 times, creating 100 fitted lines with betahats")
    print ("The distribution of betahats would resemble a normal distribution with a large enough N (by CLT)")
    
    plt.plot(xx, yy)
    plt.hist(betahats, normed=True)
    plt.xlabel("value of betahats")
    plt.ylabel("relative frequency")
    plt.show()

def SE(option):
    alpha = 0.5
    beta = 2
    N = 300
    regr = linear_model.LinearRegression()
    
    X = np.linspace(-3, 3, N)
    Y1 = [alpha + beta * xx + np.random.normal(0, 2) for xx in X]
    Y2 = [alpha + beta * xx + np.random.normal(0, 2) * (xx+3)/3 for xx in X]    
    X = np.array(X)
    Y1 = np.array(Y1)
    Y2 = np.array(Y2)

    fig = plt.figure(figsize=(20, 7))    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)     

    cov = np.zeros((N, N))
    for i in range(N):
        cov[i][i] = 1
    Yhomo = np.random.multivariate_normal(np.zeros(N), cov)

    xx = np.random.normal(0, 1, N)
    yy = [alpha + x * beta + y for x, y in zip(xx, Yhomo)]
    regr.fit(np.array(xx).reshape(N, 1), np.array(yy).reshape(N, 1))
    yyy = [regr.intercept_ + x * regr.coef_[0] for x in xx]
    residuals = [y - (regr.intercept_ + x * regr.coef_[0]) for x, y in zip(xx, yy)]
    betahomo = regr.coef_[0][0]
    sehomo = 1 / sum([(x - np.mean(xx))**2 for x in xx])

    if option == 'Homoscedastic':    
        ax1.plot(xx, yy, 'o')
        ax1.plot(xx, yyy, label='fitted line')
       
        ax2.plot(residuals, 'o')
        ax2.axhline(y=0, c='C1', label='residual=0')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Scatterplot of homoscedastic data')
        ax2.set_xlabel('X')
        ax2.set_ylabel('residual')
        ax2.set_title('Residual plot of homoscedastic data')
        ax1.legend()
        ax2.legend()
        
        plt.show()

        xx = sm.add_constant(xx)
        model = lm.OLS(yy, xx)
        results = model.fit()
        results2 = model.fit(cov_type='HC0')
        mygroups = [0]*(int(N/3)) + [1]*(int(N/3)) + [2]*(int(N/3))
        results3 = model.fit(cov_type='cluster', cov_kwds={'groups': mygroups})
        
    cov = np.zeros((N, N))
    for i in range(N):
        cov[i][i] = (i+1)/(int(N/3))
    Yhetero = np.random.multivariate_normal(np.zeros(N), cov) #increasing variance by index

    xx = sorted(np.random.normal(0, 1, N))
    yy = [alpha + x * beta + y for x, y in zip(xx, Yhetero)]
    regr.fit(np.array(xx).reshape(N, 1), np.array(yy).reshape(N, 1))
    yyy = [regr.intercept_ + x * regr.coef_[0] for x in xx]        
    residuals = [y - (regr.intercept_ + x * regr.coef_[0]) for x, y in zip(xx, yy)]
    betahetero = regr.coef_[0][0]
    sehetero = 1 / np.dot(xx, xx) * sum([resid**2 * x**2 for resid, x in zip(residuals, xx)]) * 1 / np.dot(xx, xx)
    sehetero = sehetero[0]

    if option == 'Heteroscedastic': 
        ax1.plot(xx, yy, 'o')
        ax1.plot(xx, yyy, label='fitted line')
        ax2.plot(residuals, 'o')
        ax2.axhline(y=0, c='C1', label='residual=0')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Scatterplot of heteroscedastic data')
        ax2.set_xlabel('X')
        ax2.set_ylabel('residual')
        ax2.set_title('Residual plot of heteroscedastic data')
        ax1.legend()
        ax2.legend()
        plt.show()

        xx = sm.add_constant(xx)
        model = lm.OLS(yy, xx)
        results = model.fit()
        results2 = model.fit(cov_type='HC0')
        mygroups = [0]*(int(N/3)) + [1]*(int(N/3)) + [2]*(int(N/3))
        results3 = model.fit(cov_type='cluster', cov_kwds={'groups': mygroups})

    xx = np.random.normal(0, 1, N)
    eps1 = np.random.normal(0, np.random.rand()*1, int(N/3))
    eps2 = np.random.normal(0, np.random.rand()*3, int(N/3))
    eps3 = np.random.normal(0, np.random.rand()*5, int(N/3))
    Y1 = [alpha + beta*x + e for x, e in zip(xx[:int(N/3)], eps1)]
    Y2 = [alpha + beta*x + e for x, e in zip(xx[int(N/3):int(N*2/3)], eps2)]        
    Y3 = [alpha + beta*x + e for x, e in zip(xx[:int(N*2/3)], eps3)]
    yy = np.hstack((Y1, Y2, Y3))
    regr.fit(np.array(xx).reshape(N, 1), yy.reshape(N, 1))
    yyy = [regr.intercept_ + x * regr.coef_[0] for x in xx]  
    r1 = [y - (regr.intercept_ + x * regr.coef_[0][0]) for x, y in zip(xx, Y1)]
    r2 = [y - (regr.intercept_ + x * regr.coef_[0][0]) for x, y in zip(xx, Y2)]
    r3 = [y - (regr.intercept_ + x * regr.coef_[0][0]) for x, y in zip(xx, Y3)]
    residuals = np.vstack((r1, r2, r3))

    middleterm = 0
    for i in range(3):
        middleterm += np.vdot(xx[int(N/3)*i:int(N/3)*(i+1)], residuals[int(N/3)*i:int(N/3)*(i+1)])**2

    betacluster = regr.coef_[0][0]
    secluster = 1 / np.dot(xx, xx) * middleterm * 1 / np.dot(xx, xx)

    if option == "Clustered":
        ax1.plot(xx[:int(N/3)], Y1, 'bo', label='cluster1')
        ax1.plot(xx[int(N/3):int(N*2/3)], Y2, 'ro', label='cluster2')
        ax1.plot(xx[int(N*2/3):], Y3, 'go', label='cluster3')
        ax1.plot(xx, yyy, 'C1', label='fitted line')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Scatterplot of clustered data')
        ax1.legend()
        
        ax2.plot(r1, 'bo', label='cluster1')
        ax2.plot(r2, 'ro', label='cluster2')
        ax2.plot(r3, 'go', label='cluster3')
        ax2.axhline(y=0, c='C1', label='residual=0')
        ax2.set_xlabel('X')
        ax2.set_ylabel('residual')
        ax2.set_title('Residual plot of clustered data')
        ax2.legend()
        plt.plot()
        plt.show()

        xx = sm.add_constant(xx)
        model = lm.OLS(yy, xx)
        results = model.fit()
        results2 = model.fit(cov_type='HC0')
        mygroups = [0]*(int(N/3)) + [1]*(int(N/3)) + [2]*(int(N/3))
        results3 = model.fit(cov_type='cluster', cov_kwds={'groups': mygroups})      

    print ("estimated betahat (same for all 3 models): " + str(results.params[1]))
    print ("standard error for betahat (OLS): " + str(results.bse[1]))
    print ("standard error for betahat (heteroscedasticity-consistent): " + str(results2.bse[1]))
    print ("standard error for betahat (cluster-robust): " + str(results3.bse[1]))            