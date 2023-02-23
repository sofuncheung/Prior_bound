import GPy
import numpy as np
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
GP_prob_folder = os.path.join(ROOT_DIR, 'GP_prob')
sys.path.append(GP_prob_folder)
from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix

class CustomMean(GPy.core.Mapping):
    def __init__(self,X,means):
        GPy.core.Mapping.__init__(self, input_dim=X.shape[1], output_dim=1, name="custom_means")
        self.X = X
        self.means = means
        # self.link_parameter(GPy.core.parameterization.Param('means', means))
    def f(self,X):
        indices = np.concatenate([np.nonzero(np.prod(self.X == x,1))[0] for x in X])
        if np.all(np.isin(X,self.X)):
            if len(indices) != X.shape[0]:
                raise NotImplementedError("Some elements of X appear more than once in self.X")
            else:
                return self.means[indices]
        else:
            raise NotImplementedError("Some elements of X are not in self.X")

    def update_gradients(self, dL_dF, X):
        # self.means.gradient = dL_dF.sum(0)
        pass

    def gradients_X(self, dL_dF, X):
        return np.zeros_like(X)

def nngp_mse_heaviside_posteror_params(Xtrain,Ytrain,Xtest,Kfull,noise=True):
    #find out the analytical posterior
    # NOTE: here the noise=True/False is whether to add on the test set or not.
    # If you want noise-free observations on training set, you need to set the variance of 
    # Gaussian likelihood to be zero.
    Xfull =  np.concatenate([Xtrain,Xtest])
    inference_method = GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference()
    lik = GPy.likelihoods.gaussian.Gaussian(variance=0)
    kernel = CustomMatrix(Xfull.shape[1],Xfull,Kfull)
    gp_model = GPy.core.GP(X=Xtrain,Y=Ytrain,kernel=kernel,inference_method=inference_method, likelihood=lik)
    if noise == True:
        mean, cov = gp_model.predict(Xtest,full_cov=True)
    else:
        mean, cov = gp_model.predict_noiseless(Xtest,full_cov=False)
    return mean, cov

def nngp_mse_heaviside_posteror_logp(Xtest,Ytest,mean,cov):

    #use EP approximation to estimate probability of binary labelling on test set.
    linkfun = GPy.likelihoods.link_functions.Heaviside()
    lik = GPy.likelihoods.Bernoulli(linkfun)
    inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=False)
    m = GPy.core.GP(X=Xtest,
                    Y=Ytest,
                    kernel=CustomMatrix(Xtest.shape[1],Xtest,cov),
                    inference_method=inference_method,
                    mean_function=CustomMean(Xtest,mean),
                    likelihood=lik)

    # return m.log_prior()
    return m.log_likelihood()









