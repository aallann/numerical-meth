import params
import numpy as np 
import matplotlib.pyplot as plt
from math import pi

params.pretty()

class Arithmetic_Brownian_Motion:
    """ implementation of arithmetic brownian motion 
        ->  dSt = mu*dt + sigma*dWt; S_0 = s """
    def __init__(self, mu: float, sigma: float, T: float, s: float, npaths: int, nsteps: int):
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.s = s
        self.npaths = npaths
        self.nsteps = nsteps
        self.index()

    def sim(self):
        dW: np.array = np.sqrt(self.dt) * np.random.normal(size=(self.npaths, self.nsteps))
        dS: np.array = np.concatenate((np.zeros((self.npaths,1)), self.mu * self.dt + self.sigma * dW), axis=1) 
        S: np.array = np.cumsum(dS, axis=1)
        self.S = S
    
    def expectation_(self):
        self.EX = self.s + self.mu * self.t
    
    def sampled_mean_(self):
        self.mean = np.mean(self.S, axis=0)
    
    def variance_(self):
        self.varS = self.sigma**2 * self.t
    
    def sampled_variance_(self):
        self.svarS = np.mean(np.abs(self.S - self.EX))**2
    
    def mean_absolute_deviation_(self):
        self.mad = self.sigma * np.sqrt(2 * self.t / pi)
    
    def project_(self):
        self.sim()
        self.expectation_()
        self.sampled_mean_()
        self.variance_()
        self.sampled_variance_()
        self.mean_absolute_deviation_()

        plt.figure(1)
        plt.plot(self.t, self.S[:,:].T)
        plt.plot(self.t, self.EX, 'k--', alpha=0.7, label='Expected path $E[X(t)]$')
        plt.plot(self.t, np.mean(self.S, axis = 0), 'k', label= 'Mean path')
        plt.grid(True)
        plt.xlabel('$t$')
        plt.ylabel('$X(t)$', rotation=0, ha='right')
        plt.xlim(self.t.min(), self.t.max())
        plt.legend(fontsize=12)
        plt.title(f'Arithmetic Brownian Motion $dX(t) = \mu dt + \sigma dW(t)$')

        plt.figure(2)
        plt.plot(self.t, self.varS, label='Theory')
        plt.plot(self.t, np.var(self.S, axis=0), label='Sampled')
        plt.grid(True)
        plt.xlabel('$t$')
        plt.ylabel(r'$\mathrm{Var}(X_t) = E((X_t - E(X_t))^2)$')
        plt.xlim(self.t.min(), self.t.max())
        plt.legend(fontsize=12)
        plt.title(f'Arithmetic Brownian motion Mean Square Deviation')

        plt.figure(3)
        plt.plot(self.t, np.abs(self.mad), label='Theory')
        plt.plot(self.t, np.mean(np.abs(self.S - self.EX), axis=0), label='Sampled')
        plt.grid(True)
        plt.xlabel('$t$')
        plt.ylabel(r'$E|X_t - E[X_t]| = \sqrt{\frac{2\mathrm{Var}(X_t)}{\pi}}$')
        plt.xlim(self.t.min(), self.t.max())
        plt.legend(fontsize=12)
        plt.title(f'Arithmetic Brownian motion Mean Absolute Deviation')

        plt.figure(4)     
        bins: np.array = np.arange(-1, 1.02, 0.01)

        # Begin subplot for t = 0.1
        plt.subplot(3, 1, 1)
        plt.hist(self.S[:,20], bins=bins, density=True, alpha=0.7)
        plt.grid(True)
        plt.ylabel(r'$f_X(x, 0.1)$')
        plt.xlim(-1, 1)
        plt.ylim(0, 10)
        plt.legend(fontsize=12)
        plt.title(f'Arithmetic Brownian motion: Probability Density Function at diffrerent times')

        # Begin subplot for t = 0.4
        plt.subplot(3, 1, 2)
        plt.hist(self.S[:,80], bins=bins, density=True, alpha=0.7) 
        plt.grid(True)
        plt.ylabel(r'$f_X(x, 0.4)$')
        plt.xlim(-1, 1)
        plt.ylim(0, 10)
        plt.legend(fontsize=12)

        # Begin subplot for t = 1
        plt.subplot(3, 1, 3)
        plt.hist(self.S[:,-1], bins=bins, density=True, alpha=0.7) 
        plt.grid(True)
        plt.xlabel('$x$')
        plt.ylabel(r'$f_X(x, 1)$')
        plt.xlim(-1, 1)
        plt.ylim(0, 10)
        plt.legend(fontsize=12)

        plt.show()


    def index(self):
        self.t: np.array = np.linspace(0, self.T, self.nsteps + 1)
        self.dt: float = self.T / self.nsteps

def main(): 
    mu = 0.12
    sigma = -0.05
    T = 1
    s = 0
    npaths = 200
    nsteps = 200
    ABM = Arithmetic_Brownian_Motion(mu, sigma, T, s, npaths, nsteps)
    ABM.project_()
    pass

if __name__=='__main__':
    main()