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
        self.Wn = np.array([W**i for i in range(0, self.N//2)])
    
    def fft_calc(self, data):
        '''
        This method will compute the FFT of a given data array. If the data hasn't a power of
        two size. The fft_calc will pad with zeros
        '''
        
        # Stop point
        if len(data) == 1:
            return data
        
        # Division in even and odd terms

        even_terms = np.array(data[::2])
        odd_terms = np.array(data[1::2])
        # print(f'data: {data}')
        # print(f'Even terms: {even_terms}')
        # print(f'Odd terms: {odd_terms}')
        
        # Computing FFT of the even and odd terms
        fft_even = self.fft_calc(even_terms)
        fft_odd = self.fft_calc(odd_terms)

        
        # Defining the Wn terms to the current level
        step_size = 2*len(self.Wn) // len(data)
        wn_terms = self.Wn[::step_size]        

        # Computing the butterfly operations

        fft_result = np.zeros(len(data), dtype='complex')

        for i, wn in enumerate(wn_terms):
            # print(i)
            fft_result[i] = fft_even[i] + wn*fft_odd[i]
            # Using the symmetry property
            fft_result[(i + len(data)//2)] = fft_even[i] - wn*fft_odd[i]
        

        return fft_result.tolist()