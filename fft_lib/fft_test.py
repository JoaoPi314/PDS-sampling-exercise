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

    