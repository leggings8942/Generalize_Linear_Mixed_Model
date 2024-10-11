import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from scipy import integrate
from random import Random

def poisson_distribution(k, λ):    
    return stats.poisson.pmf(round(k), λ)

def binomial_distribution(k, n, p):
    return stats.binom.pmf(round(k), round(n), p)

def normal_distribution(x, loc=0, scale=1):
    return stats.norm.pdf(x, loc=loc, scale=scale)

def uniform_distribution(x, loc=0, scale=1):
    return stats.uniform.pdf(x, loc=loc, scale=scale)

class Update_Momentum:
    def __init__(self, alpha=0.1, momentum=0.9):
        self.alpha = alpha
        self.momentum = momentum
        self.correction = np.array([])
        self.isFirst = True

    def update(self, grads):
        if self.isFirst == True:
            self.correction = np.zeros(grads.shape)
            self.isFirst = False
        
        self.correction = (1 - self.momentum) * grads + self.momentum * self.correction

        return grads + self.alpha * self.correction

class Update_Adam:
    def __init__(self, alpha=0.001, beta1=0.999, beta2=0.9999):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1t = self.beta1
        self.beta2t = self.beta2
        self.m = np.array([])
        self.v = np.array([])
        self.isFirst = True

    def update(self, grads):
        if self.isFirst == True:
            self.m = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)
            self.isFirst = False

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        m_hat = self.m / (1 - self.beta1t)
        v_hat = self.v / (1 - self.beta2t)
        
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2

        return self.alpha * m_hat / np.sqrt(v_hat + 1e-8)

class Update_Rafael:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, rate=1e-3):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.rate  = rate
        self.beta1t = self.beta1
        self.beta2t = self.beta2
        self.beta3t = self.beta3
        self.m = np.array([])
        self.v = np.array([])
        self.w = np.array([])
        self.isFirst = True

    def update(self, grads):
        if self.isFirst == True:
            self.m = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)
            self.w = np.zeros(grads.shape)
            self.isFirst = False

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        m_hat = self.m / (1 - self.beta1t)

        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        v_hat = self.v / (1 - self.beta2t)

        self.w = self.beta3 * self.w + (1 - self.beta3) * ((grads - m_hat) ** 2)
        w_hat = self.w / (1 - self.beta3t)
        
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        self.beta3t *= self.beta3

        return self.alpha * np.sign(grads) * np.abs(m_hat) / np.sqrt(v_hat + 1e-8) / np.sqrt(w_hat + self.rate)

class PoissonRegression_Non_Bayes:
    def __init__(self, isGLMM=False, tol=1e-7, max_iterate=10000, gauss_quadrature_dim=3200, random_state=None):
        self.alpha   = np.array([], dtype=np.float64)
        self.alpha0  = np.float64(0.0)
        self.isGLMM  = isGLMM
        self.gauss_s = np.float64(-1.0)
        self.standardization = np.empty([2, 1])
        self.tol = tol
        self.max_iterate = max_iterate
        self.quadrature_dim = gauss_quadrature_dim
        #self.correct_alpha   = Update_Adam()
        #self.correct_alpha0  = Update_Adam()
        #self.correct_gauss_s = Update_Adam()
        self.correct_alpha   = Update_Rafael(alpha=0.001)
        self.correct_alpha0  = Update_Rafael(alpha=0.001)
        self.correct_gauss_s = Update_Rafael(alpha=0.01)

        self.random_state = random_state
        if self.random_state != None:
            self.random = Random(self.random_state)
        else:
            self.random = Random()

    def fit(self, x_train, y_train, visible_flg=False, lim_gauss_s=-0.6):
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
        x_train = (x_train - self.standardization[0]) / self.standardization[1]
        y_train = y_train.reshape([num, 1])

        if self.isGLMM == True:
            #正規方程式
            A = np.hstack([x_train, np.ones([num, 1])])
            b = np.log(y_train + 1e-16).reshape([num])

            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha, self.alpha0 = x[0:s].reshape([1, s]), x[s]
            self.gauss_s            = np.float64(-1.0)
            
            update = 99
            now_ite = 0
            while (update > self.tol) and (now_ite < self.max_iterate) and (self.gauss_s < lim_gauss_s):
                diff_alpha   = np.zeros(self.alpha.shape)
                diff_alpha0  = np.float64(0)
                diff_gauss_s = np.float64(0)

                lambda_vec = np.exp(np.sum(self.alpha * x_train, axis=1) + self.alpha0)
                lambda_vec = lambda_vec.reshape([num, 1])
                gauss_s    = np.exp(self.gauss_s)
                
                def vectorlize_quad(y_train_indi, λ, gauss_s, speci_dim):
                    def calc_likelihood(r, k, λ, s):
                        λ_r  = λ * np.exp(r)
                        poisson = poisson_distribution(k, λ_r)
                        normal  = normal_distribution(r, 0, s)
                        return poisson * normal

                    def calc_λ(r, k, λ, s):
                        λ_r  = λ * np.exp(r)
                        poisson = poisson_distribution(k, λ_r)
                        normal  = normal_distribution(r, 0, s)
                        return poisson * normal * λ

                    def calc_squared_r(r, k, λ, s):
                        λ_r  = λ * np.exp(r)
                        poisson = poisson_distribution(k, λ_r)
                        normal  = normal_distribution(r, 0, s)
                        return poisson * normal * (r ** 2)

                    range_σ    = max(10 * gauss_s, 1)
                    likelihood = integrate.fixed_quad(calc_likelihood, -range_σ, range_σ, args=(y_train_indi, λ, gauss_s), n=speci_dim)[0]
                    diff_alpha = integrate.fixed_quad(calc_λ,          -range_σ, range_σ, args=(y_train_indi, λ, gauss_s), n=speci_dim)[0]
                    diff_s     = integrate.fixed_quad(calc_squared_r,  -range_σ, range_σ, args=(y_train_indi, λ, gauss_s), n=speci_dim)[0]
                    
                    tmp = np.zeros(2)
                    tmp[0]   = diff_alpha / (likelihood + 1e-16)
                    tmp[1]   = diff_s     / (likelihood + 1e-16)
                    return tmp[0], tmp[1]

                diff_calc         = np.frompyfunc(vectorlize_quad, 4, 2)(y_train, lambda_vec, gauss_s, self.quadrature_dim)
                diff_alpha_calc   = diff_calc[0].astype(float).reshape([num, 1])
                diff_gauss_s_calc = diff_calc[1].astype(float).reshape([num, 1])

                diff_alpha_calc   = y_train - diff_alpha_calc
                diff_gauss_s_calc = diff_gauss_s_calc / (gauss_s ** 2) - 1
                
                diff_alpha   = np.sum(diff_alpha_calc * x_train, axis=0).reshape([1, s]) / num
                diff_alpha0  = np.sum(diff_alpha_calc)                                   / num
                diff_gauss_s = np.sum(diff_gauss_s_calc)                                 / num

                tmp_alpha    = self.correct_alpha.update(  diff_alpha)
                self.alpha   += tmp_alpha
                tmp_alpha0   = self.correct_alpha0.update( diff_alpha0)
                self.alpha0  += tmp_alpha0
                tmp_gauss_s  = self.correct_gauss_s.update(diff_gauss_s)
                self.gauss_s += tmp_gauss_s

                update_diff = np.sqrt(np.sum(diff_alpha ** 2) + diff_alpha0 ** 2 + diff_gauss_s ** 2)
                update  = np.sqrt(np.sum(tmp_alpha ** 2) + tmp_alpha0 ** 2 + tmp_gauss_s ** 2)
                now_ite = now_ite + 1

                if (now_ite % 10 == 0) and visible_flg:
                    lambda_ = np.exp(np.sum(self.alpha * x_train, axis=1) + self.alpha0)
                    lambda_ = lambda_.reshape([num, 1])
                    mse     = np.sum((y_train - lambda_) ** 2) / num

                    print(f"ite:{now_ite}  alpha0:{self.alpha0}  alpha:{self.alpha}  gauss_s:{self.gauss_s}  update_diff:{update_diff}  update:{update}  MSE:{mse}", flush=True)
                    #print(f"diff_alpha0:{diff_alpha0 / update}  diff_alpha:{diff_alpha / update}  diff_gauss_s:{diff_gauss_s / update}", flush=True)
                    #print(f"diff_alpha0:{tmp_alpha0}  diff_alpha:{tmp_alpha}  diff_gauss_s:{tmp_gauss_s}", flush=True)
                    #print(f"diff_alpha0:m:{self.correct_alpha0.m}  v:{self.correct_alpha0.v}  w:{self.correct_alpha0.w}", flush=True)
                    #print(f"diff_alpha:m:{self.correct_alpha.m}  v:{self.correct_alpha.v}  w:{self.correct_alpha.w}", flush=True)
            
        else:
            #正規方程式
            A = np.hstack([x_train, np.ones([num, 1])])
            b = np.log(y_train + 1e-16).reshape([num])

            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha, self.alpha0 = x[0:s], x[s]

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
        gauss_s    = np.exp(self.gauss_s)
        
        if self.isGLMM == True:
            def vectorlize_quad(y_indiv, λ, gauss_σ, quadrature_dim):
                def mixture(r, k, λ, s):
                    λ_r  = λ * np.exp(r)
                    poisson   = poisson_distribution(k, λ_r)
                    normal    = normal_distribution(r, 0, s)
                    return poisson * normal
                
                quadrature_σ = max(10 * gauss_σ, 1)
                tmp = integrate.fixed_quad(mixture, -quadrature_σ, quadrature_σ, args=(y_indiv, λ, gauss_σ), n=quadrature_dim)[0]
                return tmp
            
            prob           = np.frompyfunc(vectorlize_quad, 4, 1)(y_train, lambda_vec, gauss_s, self.quadrature_dim)
            prob           = prob.astype(float).reshape([num, 1])
            log_likelihood = np.sum(np.log(prob + 1e-16)) / num
            return log_likelihood

        else:
            prob           = np.frompyfunc(poisson_distribution, 2, 1)(y_train, lambda_vec)
            prob           = prob.astype(float).reshape([num, 1])
            log_likelihood = np.sum(np.log(prob + 1e-16)) / num
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
        
        x_test = (x_test - self.standardization[0]) / self.standardization[1]
        lambda_vec = np.exp(np.sum(self.alpha * x_test, axis=1) + self.alpha0)
        gauss_s    = np.exp(self.gauss_s)
        
        if self.isGLMM == True:
            def vectorlize_quad(y_indiv, λ, gauss_σ, quadrature_dim):
                def mixture(r, k, λ, s):
                    λ_r  = λ * np.exp(r)
                    poisson   = poisson_distribution(k, λ_r)
                    normal    = normal_distribution(r, 0, s)
                    return poisson * normal
                
                quadrature_σ = max(10 * gauss_σ, 1)
                tmp = integrate.fixed_quad(mixture, -quadrature_σ, quadrature_σ, args=(y_indiv, λ, gauss_σ), n=quadrature_dim)[0]
                return y_indiv, tmp

            y_test_mat = (lambda_vec - sample/2 * step).reshape([len(lambda_vec), 1])
            lambda_mat = (lambda_vec                  ).reshape([len(lambda_vec), 1])
            for ite in np.arange(-sample / 2 * step + step, sample / 2 * step, step):
                y_test_mat = np.hstack([y_test_mat, (lambda_vec + ite).reshape([len(lambda_vec), 1])])
                lambda_mat = np.hstack([lambda_mat, (lambda_vec      ).reshape([len(lambda_vec), 1])])

            pred_mat  = np.frompyfunc(vectorlize_quad, 4, 2)(y_test_mat, lambda_mat, gauss_s, self.quadrature_dim)
            pred_y    = pred_mat[0].astype(float)
            pred_prob = pred_mat[1].astype(float)
            
            return pred_y, pred_prob
        else:
            return lambda_vec