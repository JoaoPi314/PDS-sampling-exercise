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
        '''
        This method will compute the size of the filter based on user input at 
        instantiaton moment. If the size passed is a power of two, the value will be
        assigned to N, else, the N will be assigned with the next power of two number
        '''
        self.N = 2**math.ceil(math.log2(N))

    def __set_Wn(self):
        '''
        This method will compute the Wn values. Considering the simmetry property, only
        half of the values must be calculated, the other half is obtained by just inverting
        the signal of the first half.
        '''
        W = np.exp(-1j*2*np.pi/self.N)
        self.Wn = [W**i for i in range(0, self.N//2)]