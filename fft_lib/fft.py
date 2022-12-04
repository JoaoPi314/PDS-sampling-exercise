import math
import numpy as np

class FFT():
    
    def __init__(self, N):
        self.__set_N(N)
        self.__set_Wn()

    def get_N(self):
        return self.N

    def get_Wn(self):
        return self.Wn

    def __set_N(self, N):
        self.N = 2**math.ceil(math.log2(N))

    def __set_Wn(self):
        W = np.exp(-1j*2*np.pi/self.N)
        self.Wn = [W**i for i in range(0, self.N//2)]