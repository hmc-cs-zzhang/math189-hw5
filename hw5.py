import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

df = pd.read_csv('classification.csv', sep=',', engine='python').as_matrix()
X = df[:, :2]
ones = np.ones((X.shape[0], 1))
X = np.matrix(np.hstack((ones, X)))
Y = np.matrix(df[:, 2])
V0 = 100 * np.matrix(np.identity(X.shape[1]))

# problem 2

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def find_opt_theta(X, Y, V, alpha=1.9, max_iter=10000, freq=100):
    theta = np.matrix(np.random.rand(X.shape[0], 1) * np.mean(X))
    vinv = np.linalg.inv(V)
    grad = 999
    itr = 0
    while itr < max_iter:        
        mu = np.matrix(sigmoid(theta.transpose() * X)).transpose()
        cur = X * (mu - Y) + (vinv + vinv.transpose()) * theta
        cur_grad = np.linalg.norm(cur)
        diff = cur_grad - grad
        if diff > 0:
            break
        theta = theta - (alpha / X.shape[1]) * cur
        if itr % freq == 0:
            print "[iter = " + str(itr) + "] with gradient norm: " + str(cur_grad)
        itr += 1
        grad = cur_grad

    return theta

def find_hessian(theta, X, V):
    mu = sigmoid(theta.transpose() * X)
    S = np.zeros((X.shape[1], X.shape[1]))
    for i in range(S.shape[0]):
        S.itemset((i, i), mu.item(i) * (1 - mu.item(i)))
    del mu
    S = np.matrix(S)

    vinv = np.linalg.inv(V)
    return X * S * X.transpose() + (vinv + vinv.transpose())


def find_posterior(theta, theta_opt, H_opt):
    n = H_opt.shape[0]
    denom = (2 * np.pi) ** (n * 0.5) * np.linalg.det(H_opt) ** 0.5
    numer = np.exp(-0.5 * ((theta - theta_opt).transpose() * H_opt * (theta - theta_opt)).item(0))
    return 1.0 * numer / denom

def generate_samples(theta, cov, sampleSize):
    samples = np.random.multivariate_normal(theta, cov, sampleSize)
    ans = [np.matrix(sample).transpose() for sample in samples]
    return ans

def generate_samples_mat(theta, cov, sampleSize):
    samples = np.random.multivariate_normal(theta, cov, sampleSize)
    return samples.transpose()

def gen_pred_density(x, y, theta):
    predX = np.matrix([1, x, y]).transpose()
    return sigmoid(theta.transpose() * predX).item(0)

def predict(predX, theta):
    return (sigmoid(theta.transpose() * predX)).mean(axis=0)

def gen_problem_2_graph(theta_opt, hessian_opt, X, Y):
    plt.style.use('bmh')
    feature1, feature2 = X[:, 1], X[:, 2]

    feature1Neg = [feature1.item(i) for i in range(Y.shape[1]) if Y.item(i) == 0]
    feature1Pos = [feature1.item(i) for i in range(Y.shape[1]) if Y.item(i) == 1]
    feature2Neg = [feature2.item(i) for i in range(Y.shape[1]) if Y.item(i) == 0]
    feature2Pos = [feature2.item(i) for i in range(Y.shape[1]) if Y.item(i) == 1]

    negPlot, = plt.plot(feature1Neg, feature2Neg, 'bo')
    posPlot, = plt.plot(feature1Pos, feature2Pos, 'm.')

    xAxis = np.linspace(np.min(feature1) - 1, np.max(feature1) + 1, num=11)

    sampleSize = 20
    thetas = generate_samples([theta_opt.item(i) for i in range(theta_opt.shape[0])], np.linalg.inv(hessian_opt), sampleSize)

    for theta in thetas:
        bias, k = -theta.item(0) / theta.item(2), -theta.item(1) / theta.item(2)
        yAxis = [k * x + bias for x in xAxis]
        boundaryPlot, = plt.plot(xAxis, yAxis, 'g')

    bias, k = -theta_opt.item(0) / theta_opt.item(2), -theta_opt.item(1) / theta_opt.item(2)
    yAxis = [k * x + bias for x in xAxis]
    optPlot, = plt.plot(xAxis, yAxis, color='purple')
                    
    plt.axis([1.1 * np.min(feature1), 1.1 * np.max(feature1), 1.1 * np.min(feature2), 1.1 * np.max(feature2)])
    plt.title('Posterior Predictive Distribution')
    plt.xlabel('Hours')
    plt.ylabel('Grade')
    plt.legend((negPlot, posPlot, optPlot), \
        ('Failed', 'Passed', 'MAP boundary'), loc=1)

    # Contour
    w1, w2 = np.mgrid[3: 7: .01, 3: 7: .01]
    pos = np.empty(w1.shape + (3,))
    pos[:, :, 0] = theta_opt.item(0)
    pos[:, :, 1] = w1
    pos[:, :, 2] = w2
    rv = stats.multivariate_normal([theta_opt.item(i) for i in range(theta_opt.shape[0])], np.linalg.inv(hessian_opt.tolist()))
    plt.title('Marginal Posterior Fixing w0 = w0*')
    plt.contour(w1, w2, rv.pdf(pos))

    plt.show()

def gen_problem_2():
    theta_opt = find_opt_theta(X.transpose(), Y.transpose(), V0)
    hessian_opt = find_hessian(theta_opt, X.transpose(), V0)
    print 'optimal theta:\n' + str(theta_opt)
    print 'optimal hessian:\n' + str(np.linalg.inv(hessian_opt))

    gen_problem_2_graph(theta_opt, hessian_opt, X, Y)

gen_problem_2()

# problem 3

def genPredPlot(X, Y):
    theta_opt = find_opt_theta(X.transpose(), Y.transpose(), V0)
    hessian_opt = find_hessian(theta_opt, X.transpose(), V0)

    plt.style.use('bmh')
    feature1, feature2 = X[:, 1], X[:, 2]

    feature1Neg = [feature1.item(i) for i in range(Y.shape[1]) if Y.item(i) == 0]
    feature1Pos = [feature1.item(i) for i in range(Y.shape[1]) if Y.item(i) == 1]
    feature2Neg = [feature2.item(i) for i in range(Y.shape[1]) if Y.item(i) == 0]
    feature2Pos = [feature2.item(i) for i in range(Y.shape[1]) if Y.item(i) == 1]

    # plot scatters of original data
    plt.subplot(2,1,1)
    negPlot, = plt.plot(feature1Neg, feature2Neg, 'bo')
    posPlot, = plt.plot(feature1Pos, feature2Pos, 'ro')

    xAxis = np.linspace(np.min(feature1) - 1, np.max(feature1) + 1, num=11)

    # generate samples of theta
    sampleSize = 8000
    thetas = generate_samples_mat([theta_opt.item(i) for i in range(theta_opt.shape[0])], np.linalg.inv(hessian_opt), sampleSize)

    XPredMat = 7 * np.random.rand(10000,2) - 3
    X_bias = np.hstack((np.ones((XPredMat.shape[0],1)),XPredMat)).transpose()
    yPred = predict(X_bias, np.matrix(thetas))
    n_levels = 20
    levels = np.linspace(1.1 * np.min(yPred), 1.1 * np.max(yPred), n_levels)
    plt.tricontour(X_bias[1, :], X_bias[2, :], [yPred.item(i) for i in range(yPred.shape[1])], levels=levels)

    plt.axis([1.1 * np.min(feature1), 1.1 * np.max(feature1), 1.1 * np.min(feature2), 1.1 * np.max(feature2)])
    plt.xlabel('Hours')
    plt.ylabel('Grade')
    plt.legend((negPlot, posPlot), ('failed', 'passed'), loc=3)
    plt.title('Posterior Predictive Distribution')

    plt.subplot(2,1,2)
    value = sigmoid(theta_opt.transpose() * X_bias) - yPred
    plt.hist([value.item(i) for i in range(value.shape[1])], bins=100)
    plt.title(r"$\sigma(\mathbf{X}\mathbf{w}^\star) - \mathbf{y}_{MCMC}$")    
    
    plt.tight_layout()
    plt.show()

# genPredPlot(X, Y)