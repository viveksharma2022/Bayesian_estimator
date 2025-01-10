import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt

def Proposal(prevTheta, searchWidth = 0.5):
    # this function generates the proposal for the new theta
    # we assume that the distribution of the random variables 
    # is normal for the first two and gamma for the third.
    # conditional on the previous value of the accepted parameters (prec_theta)
    outTheta = np.zeros(3)
    outTheta[:2] = sc.multivariate_normal(prevTheta[:2], cov = np.eye(2)*searchWidth**2).rvs(1)
    outTheta[2] = sc.gamma(a=prevTheta[2]*searchWidth*500, scale=1/(500*searchWidth)).rvs()
    return outTheta

def LogLikelihood(x, theta):
    # x is the data matrix, first column for input and second column for output.
    # theta is a vector containing the parameters for the evaluation
    # remember theta[0] is a, theta[1] is b and theta[2] is sigma
    xs = x[:,0]
    ys = x[:,1]
    lhdOut = sc.norm.logpdf(ys, loc = theta[0]*xs + theta[1], scale = theta[2])
    lhdOut = np.sum(lhdOut)
    return lhdOut

def Prior(theta):
    # evaluate the prior for the parameters on a multivariate gaussian. 
    # 1. Mean of 0 Reflects Neutral Belief
    # Setting the mean to 0 indicates that, before observing any data, we do not have a strong belief about whether the parameter is positive or negative.
    # It serves as a neutral starting point when we lack specific domain knowledge or want to remain unbiased.
    # A high variance in the normal distribution means the prior is diffuse, spreading probability mass over a wide range of values.
    # This minimizes the influence of the prior on the posterior, allowing the data to dominate and play a larger role in shaping the posterior distribution.
    # A large 
    # sigma flattens the prior, effectively making it "non-informative" while still being proper (integrates to 1).
    priorOut = sc.multivariate_normal.logpdf(theta[:2], mean = np.array([0,0]), cov = np.eye(2)*100)
    priorOut += sc.gamma.logpdf(theta[2], a=1, scale=1) # for noise parameter
    return priorOut

def ProposalRatio(thetaOld, thetaNew, searchWidth = 10):
    # this is the proposal distribution ratio
    # first, we calculate of the pdf of the proposal distribution at the old value of theta with respect to the new 
    # value of theta. And then we do the exact opposite.
    propRatioOut = sc.multivariate_normal.logpdf(thetaOld[:2], mean = thetaNew[:2], cov = np.eye(2)*searchWidth**2)
    propRatioOut += sc.gamma.logpdf(thetaOld[2], a = thetaNew[2]*searchWidth*500, scale=1/(500*searchWidth))
    propRatioOut -= sc.multivariate_normal.logpdf(thetaNew[:2], mean = thetaOld[:2], cov = np.eye(2)*searchWidth**2)
    propRatioOut -= sc.gamma.logpdf(thetaNew[2], a = thetaOld[2]*searchWidth*500, scale=1/(500*searchWidth))
    return propRatioOut

if __name__=="__main__":

    #Generate data
    np.random.seed(100)
    numPoints = 100

    x = np.linspace(0,30,numPoints)

    # parameters
    a = 3
    b = 20
    sigma = 10

    # data and noise
    noise = np.random.randn(numPoints)*sigma
    y = a*x + b + noise

    data = np.vstack((x,y)).T

    # Actual calculation
    np.random.seed(101)
    width = 0.2

    thetas = np.random.rand(3).reshape(1,-1)
    accepted = 0
    rejected = 0
    N = 20000

    pdfsAccepted = []
    for i in range(N):
        if(i%200 == 0):
            print(f"Iteration: {i}")

        # Step1: Proposal for theta
        thetaNew = Proposal(thetas[-1,:], width)

        # Step2: calculate the likelihood of this proposal and the likelihood
        # for the old value of theta
        logLikeOldTheta = LogLikelihood(data, thetas[-1,:])
        logLikeNewTheta = LogLikelihood(data, thetaNew)

        # Step3: Evaluate the prior at the new and old theta
        thetaOldPrior = Prior(thetas[-1,:])
        thetaNewPrior = Prior(thetaNew)


        # Step4: finally, we need the proposal distribution ratio
        propRatio = ProposalRatio(thetas[-1,:], thetaNew, searchWidth=width)

        # Step5: assemble likelihood, priors and proposal distributions
        likelihoodPriorProposalRatio = logLikeNewTheta - logLikeOldTheta + \
                             thetaNewPrior - thetaOldPrior + propRatio

        # Step6: throw a - possibly infinitely weigthed - coin. The move for Metropolis-Hastings is
        # not deterministic. Here, we exponentiate the likelihood_prior_proposal ratio in order
        # to obtain the probability in linear scale
        if(np.exp(likelihoodPriorProposalRatio) > sc.uniform.rvs()):
            pdfsAccepted.append(logLikeNewTheta)
            thetas = np.vstack((thetas, thetaNew))
            accepted += 1
        else:
            rejected += 1

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    xdata = np.linspace(0,35)
    for thetaVals in thetas:
        plt.plot(xdata, thetaVals[0]*xdata + thetaVals[1])
    plt.show()

    plt.figure()
    plt.subplot(1,3,1)
    plt.hist(thetas[:,0],200)
    plt.subplot(1,3,2)
    plt.hist(thetas[:,1],200)
    plt.subplot(1,3,3)
    plt.hist(thetas[:,2],200)
    plt.show()


    # Maximum a posteriori
    maxIndex = pdfsAccepted.index(max(pdfsAccepted))
    thetaMax = thetas[maxIndex]

    print(f"MAP: {thetaMax}")

    plt.figure()
    plt.scatter(data[:,0], data[:,1])
    xdata = np.linspace(0,35)
    plt.plot(xdata, thetaVals[0]*xdata + thetaVals[1])
    plt.show()
