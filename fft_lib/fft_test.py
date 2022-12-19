import unittest
import pytest

from fft import *

class Test_fft_implementation(unittest.TestCase):
    
    def test_initiation_with_power_of_two_filter_size(self):
        
        i_data = 1024
        
        fft = FFT(i_data)
        o_data = fft.get_N()

        expected_data = 1024
        
        self.assertEqual(o_data, expected_data)

    def test_initiation_with_non_power_of_two_filter_size(self):
        i_data = [120, 300, 679]

        fft = [FFT(i) for i in i_data]
        o_data = [i.get_N() for i in fft]

        expected_data = [128, 512, 1024]

        for dt, expected_dt in zip(o_data, expected_data):
            self.assertEqual(dt, expected_dt)
    
    def test_filter_size_is_integer(self):
        
        i_data = 1024
        fft = FFT(i_data)
        o_data = fft.get_N()

        self.assertEqual(type(o_data), int)
    
    def test_fft_computes_half_of_Wn_to_power_of_two_N(self):
        N = 16
        
        fft = FFT(N)
        o_data = fft.get_Wn()
        
        expected_data = [1+0j, 
                          0.9238795325112867-0.3826834323650898j, 
                          0.7071067811865475-0.7071067811865476j, 
                          0.38268343236508967-0.9238795325112867j, 
                          -2.220446049250313e-16-1j, 
                          -0.38268343236509-0.9238795325112866j, 
                          -0.7071067811865477-0.7071067811865474j, 
                          -0.9238795325112868-0.38268343236508945j]
        
        self.assertEqual(o_data, expected_data)       

    def test_fft_computes_half_of_Wn_to_non_power_of_two_N(self):
        N = 25
        fft = FFT(N)
        o_data = fft.get_Wn()

        expected_data = [1+0j,
                         0.9807852804032304-0.19509032201612825j, 
                         0.9238795325112867-0.3826834323650897j,
                         0.8314696123025452-0.5555702330196022j,
                         0.7071067811865475-0.7071067811865475j,
                         0.5555702330196022-0.8314696123025451j, 
                         0.3826834323650897-0.9238795325112866j, 
                         0.1950903220161283-0.9807852804032303j, 
                         -0.9999999999999998j, 
                         -0.1950903220161282-0.9807852804032302j, 
                         -0.3826834323650896-0.9238795325112865j, 
                         -0.5555702330196021-0.831469612302545j, 
                         -0.7071067811865474-0.7071067811865474j, 
                         -0.8314696123025449-0.5555702330196021j, 
                         -0.9238795325112864-0.3826834323650896j, 
                         -0.9807852804032301-0.19509032201612825j]
        
        self.assertEqual(o_data, expected_data)

    def test_size_one_fft_returns_number_in_argument(self):
        i_data = [4]
        
        fft = FFT(1)

        o_data = fft.fft_calc(i_data)

        expected_data = [4]

        self.assertEqual(o_data, expected_data)

    def test_fft_of_size_two(self):

        i_data = [3, 2]

        fft = FFT(2)
        o_data = fft.fft_calc(i_data)

        expected_data = [5, 1]

        self.assertEqual(o_data, expected_data)

    def test_fft_of_size_four(self):

        i_data = [1, -1, 1, -1]

        fft = FFT(4)
        o_data = fft.fft_calc(i_data)

        expected_data = [0, 0, 4, 0]

        self.assertEqual(o_data, expected_data)

    def test_fft_of_size_eight(self):

        i_data = [1, -1, 1, -1, 5, 6, 7, 8]

        fft = FFT(8)
        o_data = fft.fft_calc(i_data)

        expected_data = [26 + 0j, -2.58578644 + 17.3137085j, -2. + 2j, 
                         -5.41421356 + 5.3137085j, 2. + 0j,
                        -5.41421356 - 5.3137085j, -2.-2j, 
                        -2.58578644-17.3137085j]

        print(o_data)
        print(expected_data)

        self.assertEqual(o_data, pytest.approx(expected_data))