from scipy import stats
import numpy as np

class RainbowOptionPrice():
    def __init__(self, S1, S2, k, r, s1, s2, T, p):
        self.S1 = S1
        self.S2 = S2
        self.k = k
        self.r = r
        self.s1 = s1
        self.s2 = s2
        self.T = T
        self.p = p
        
    def d1(self):
        a = np.log(self.S1 / self.k) + (self.r - 0.5 * self.s1**2) * self.T
        b = self.s1 * np.sqrt(self.T)
        return a / b
        
    def d2(self):
        a = np.log(self.S2 / self.k) + (self.r - 0.5 * self.s2**2) * self.T
        b = self.s2 * np.sqrt(self.T)
        return a / b

    def d11(self):
        return self.d1() + self.s1 * np.sqrt(self.T)

    def d22(self):
        return self.d2() + self.s2*np.sqrt(self.T)

    def sA(self):
        return np.sqrt(self.s1**2 - 2 * self.p * self.s1 * self.s2 + self.s2**2)

    def d1221(self):
        sA = self.sA()
        d12 = (np.log(self.S2 / self.S1) - 0.5 * sA**2 * self.T) / (sA * np.sqrt(self.T))
        d21 = (np.log(self.S1 / self.S2) - 0.5 * sA**2 * self.T) / (sA * np.sqrt(self.T))
        return d12, d21

    def p12(self):
        sA = self.sA()
        p1 = (self.p * self.s2 - self.s1) / sA
        p2 = (self.p * self.s1 - self.s2) / sA
        return p1, p2

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        d11 = self.d11()
        d22 = self.d22()
        d12, d21 = self.d1221()
        p1, p2 = self.p12()
        
        call = (self.S1 * stats.multivariate_normal.cdf([d11, d12], mean=[0, 0], cov=[[1, p1], [p1, 1]]) +
                self.S2 * stats.multivariate_normal.cdf([d22, d21], mean=[0, 0], cov=[[1, p2], [p2, 1]]) -
                self.k * np.exp(-self.r * self.T) * stats.multivariate_normal.cdf([d1, d2], mean=[0, 0], cov=[[1, self.p], [self.p, 1]]))
        
        return call
