import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from scipy import integrate
from math  import ceil

def np_mode(x, axis=None):
    if type(x) is list:
        x = np.array(x)
    
    if (axis != None) and (axis != 0) and (axis != 1):
        print(f"axis = {axis}")
        print("エラー：：次元数が一致しません。")
        raise

    if axis == None:
        x = x.ravel()
        bins  = 1 + np.log2(len(x))
        bins  = bins * 2
        width = (np.max(x) - np.min(x)) / ceil(bins) / 2
        hist, idex = np.histogram(x, bins=ceil(bins))
        mode = idex[np.argmax(hist)]
        return np.float64(mode + width)
    
    elif axis == 0:
        return np.array([np_mode(x[:, idx], axis=None) for idx in range(0, x.shape[1])])
    
    elif axis == 1:
        return np.array([np_mode(x[idx, :], axis=None) for idx in range(0, x.shape[0])])
    
    else:
        raise

def poisson_distribution(k, λ):    
    return stats.poisson.pmf(round(k), λ)

def binomial_distribution(k, n, p):
    return stats.binom.pmf(round(k), round(n), p)

def gamma_distribution(x, shape, scale):
    return stats.gamma.pdf(x, a=shape, scale=scale)

def normal_distribution(x, loc=0, scale=1):
    return stats.norm.pdf(x, loc=loc, scale=scale)

def uniform_distribution(x, loc=0, scale=1):
    return stats.uniform.pdf(x, loc=loc, scale=scale)

class PoissonRegression_On_Bayes:
    def __init__(self, isGLMM=False, gauss_quadrature_dim=3200, random_state=None):
        self.alpha     = np.array([], dtype=np.float64)
        self.alpha0    = np.float64(0.0)
        self.gauss_r_σ = np.float64(0.0)
        self.isGLMM  = isGLMM
        self.restrict_alpha  = np.float64(0.0)
        self.restrict_alpha0 = np.float64(0.0)
        self.standardization = np.empty([2, 1])
        self.quadrature_dim = gauss_quadrature_dim
        self.sampling_alpha     = np.array([], dtype=np.float64)
        self.sampling_alpha0    = np.array([], dtype=np.float64)
        self.sampling_gauss_r_σ = np.array([], dtype=np.float64)

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random

    def sampling(self, x_train, y_train, scale=2, iter_num=1e6):
        if type(x_train) is pd.core.frame.DataFrame:
            x_train = x_train.to_numpy()

        if type(y_train) is pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        if type(x_train) is list:
            x_train = np.array(x_train)
        
        if type(y_train) is list:
            y_train = np.array(y_train)
        
        if (x_train.ndim != 2) or (y_train.ndim != 1):
            print(f"x_train dims = {x_train.ndim}")
            print(f"y_train dims = {y_train.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        num, s = x_train.shape
        self.standardization = np.empty([2, s])
        self.standardization[0] = np.mean(x_train, axis=0)
        self.standardization[1] = np.std( x_train, axis=0)
        tmp_x_train = (x_train - self.standardization[0]) / self.standardization[1]
        tmp_y_train = y_train.reshape([num, 1])

        #正規方程式
        A = np.hstack([tmp_x_train, np.ones([num, 1])])
        b = np.log(tmp_y_train).reshape([num])

        x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        self.alpha, self.alpha0 = x[0:s].reshape([1, s]), x[s]

        self.restrict_alpha  = self.alpha
        self.restrict_alpha0 = self.alpha0

        tmp_alpha, tmp_alpha0 = self.alpha, self.alpha0
        tmp_likelihood        = self.log_likelihood(x_train, y_train)

        tmp_GLMM_flg = self.isGLMM
        self.isGLMM  = False
        
        now_ite  = 0
        while now_ite < iter_num:
            rnd_alpha = self.random.normal(loc=0, scale=scale, size=(1, s + 1))

            self.alpha, self.alpha0 = tmp_alpha + rnd_alpha[0, 0:s], tmp_alpha0 + rnd_alpha[0, s]
            now_likelihood          = self.log_likelihood(x_train, y_train)

            now_rand = self.random.random()
            now_rate = tmp_likelihood / (now_likelihood + 1e-16)
            if now_rand < now_rate:

                if now_ite % 100 == 0:
                    print(f"現在ite:{now_ite}  保存サンプリング数:{len(self.sampling_alpha0)}", flush=True)
                
                tmp_alpha, tmp_alpha0 = self.alpha, self.alpha0
                tmp_likelihood        = now_likelihood

                if now_ite == 0:
                    self.sampling_alpha  = np.array(tmp_alpha).reshape(  [1, s])
                    self.sampling_alpha0 = np.array(tmp_alpha0).reshape( [1, 1])
                else:
                    self.sampling_alpha  = np.vstack((self.sampling_alpha,     tmp_alpha))
                    self.sampling_alpha0 = np.vstack((self.sampling_alpha0,    tmp_alpha0))
                now_ite  = now_ite  + 1

        if tmp_GLMM_flg:
            self.isGLMM    = True
            self.alpha     = np_mode(self.sampling_alpha,  axis=0).reshape(self.alpha.shape)
            self.alpha0    = np_mode(self.sampling_alpha0, axis=0).reshape(self.alpha0.shape)
            self.gauss_r_σ = -1

            tmp_gauss_r_σ  = self.gauss_r_σ
            tmp_likelihood = self.log_likelihood(x_train, y_train)

            now_ite  = 0
            while now_ite < iter_num:
                rnd_gauss_r_σ  = self.random.normal(loc=0, scale=scale)
                self.gauss_r_σ = tmp_gauss_r_σ + rnd_gauss_r_σ
                now_likelihood = self.log_likelihood(x_train, y_train)

                now_rand = self.random.random()
                now_rate = tmp_likelihood / (now_likelihood + 1e-16)
                if now_rand < now_rate:

                    if now_ite % 100 == 0:
                        print(f"現在ite:{now_ite}  保存サンプリング数:{len(self.sampling_gauss_r_σ)}", flush=True)
                
                    tmp_gauss_r_σ  = self.gauss_r_σ
                    tmp_likelihood = now_likelihood

                    if now_ite == 0:
                        self.sampling_gauss_r_σ = np.array(tmp_gauss_r_σ).reshape([1, 1])
                    else:
                        self.sampling_gauss_r_σ = np.vstack((self.sampling_gauss_r_σ, tmp_gauss_r_σ))
                    now_ite  = now_ite  + 1

        return True
    
    def log_likelihood(self, x_train, y_train):
        if type(x_train) is pd.core.frame.DataFrame:
            x_train = x_train.to_numpy()

        if type(y_train) is pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        if type(x_train) is list:
            x_train = np.array(x_train)
        
        if type(y_train) is list:
            y_train = np.array(y_train)
        
        if (x_train.ndim != 2) or (y_train.ndim != 1):
            print(f"x_train dims = {x_train.ndim}")
            print(f"y_train dims = {y_train.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        num, _  = x_train.shape
        y_train = y_train.reshape([num, 1])
        
        x_train = (x_train - self.standardization[0]) / self.standardization[1]
        lambda_vec = np.exp(np.sum(self.alpha * x_train, axis=1) + self.alpha0)
        lambda_vec = lambda_vec.reshape([num, 1])

        restrict_α  = self.restrict_alpha
        restrict_α0 = self.restrict_alpha0

        if self.isGLMM == True:
            gauss_r_σ = np.exp(self.gauss_r_σ)

            restrict_α       = normal_distribution(self.alpha,     restrict_α,  10)
            restrict_α0      = normal_distribution(self.alpha0,    restrict_α0, 10)
            restrict_exp_σ   = normal_distribution(self.gauss_r_σ, 0,           0.1)
            restrict_gauss_σ = normal_distribution(gauss_r_σ,      1,           0.1)
            
            restrict_prob = restrict_α * restrict_α0 * restrict_exp_σ * restrict_gauss_σ

            def vectorlize_quad(y_indiv, λ, gauss_r_σ, quadrature_dim):
                def mixture(r, k, λ, σ):
                    λ_r = λ * np.exp(r)
                    poisson    = poisson_distribution(k, λ_r)
                    restrict_r = normal_distribution(r, 0, σ)
                    return np.nan_to_num(poisson * restrict_r)

                quadrature_σ = max(10 * gauss_r_σ, 1)
                likelihood   = integrate.fixed_quad(mixture, -quadrature_σ, quadrature_σ, args=(y_indiv, λ, gauss_r_σ), n=quadrature_dim)[0]
                return likelihood
            
            prob           = np.frompyfunc(vectorlize_quad, 4, 1)(y_train, lambda_vec, gauss_r_σ, self.quadrature_dim)
            prob           = prob.astype(float).reshape([num, 1])
            log_prob       = np.sum(np.log(prob)) + np.log(restrict_prob)
            log_likelihood = log_prob / num
            return log_likelihood

        else:
            restrict_α    = normal_distribution(self.alpha,  restrict_α,  10)
            restrict_α0   = normal_distribution(self.alpha0, restrict_α0, 10)

            restrict_prob = restrict_α * restrict_α0

            def vectorlize_non_quad(y_indiv, λ):
                likelihood = poisson_distribution(y_indiv, λ)
                return likelihood

            prob           = np.frompyfunc(vectorlize_non_quad, 2, 1)(y_train, lambda_vec)
            prob           = prob.astype(float).reshape([num, 1])
            log_prob       = np.sum(np.log(prob)) + np.log(restrict_prob)
            log_likelihood = log_prob / num
            return log_likelihood


    def predict(self, x_test, sample=100, step=1):
        if type(x_test) is pd.core.frame.DataFrame:
            x_test = x_test.to_numpy()
        
        if type(x_test) is list:
            x_test = np.array(x_test)
        
        if x_test.ndim != 2:
            print(f"x_train dims = {x_test.ndim}")
            print("エラー：：次元数が一致しません。")
            return False
        
        self.alpha     = np_mode(self.sampling_alpha,     axis=0).reshape(self.alpha.shape)
        self.alpha0    = np_mode(self.sampling_alpha0,    axis=0).reshape(self.alpha0.shape)
        self.gauss_r_σ = np_mode(self.sampling_gauss_r_σ, axis=0).reshape(self.gauss_r_σ.shape)

        x_test = (x_test - self.standardization[0]) / self.standardization[1]
        lambda_vec = np.exp(np.sum(self.alpha * x_test, axis=1) + self.alpha0)
        gauss_r_σ  = np.exp(self.gauss_r_σ)

        if self.isGLMM == True:
            def vectorlize_quad(y_indiv, λ, gauss_r_σ, quadrature_dim):
                def mixture(r, k, λ, s):
                    λ_r  = λ * np.exp(r)
                    poisson   = poisson_distribution(k, λ_r)
                    normal    = normal_distribution(r, 0, s)
                    return poisson * normal
                
                quadrature_σ = max(10 * gauss_r_σ, 1)
                tmp = integrate.fixed_quad(mixture, -quadrature_σ, quadrature_σ, args=(y_indiv, λ, gauss_r_σ), n=quadrature_dim)[0]
                return y_indiv, tmp

            y_test_mat = (lambda_vec - sample/2 * step).reshape([len(lambda_vec), 1])
            lambda_mat = (lambda_vec                  ).reshape([len(lambda_vec), 1])
            for ite in np.arange(-sample / 2 * step + step, sample / 2 * step, step):
                y_test_mat = np.hstack([y_test_mat, (lambda_vec + ite).reshape([len(lambda_vec), 1])])
                lambda_mat = np.hstack([lambda_mat, (lambda_vec      ).reshape([len(lambda_vec), 1])])

            pred_mat  = np.frompyfunc(vectorlize_quad, 4, 2)(y_test_mat, lambda_mat, gauss_r_σ, self.quadrature_dim)
            pred_y    = pred_mat[0].astype(float)
            pred_prob = pred_mat[1].astype(float)
            
            return pred_y, pred_prob
        else:
            return lambda_vec