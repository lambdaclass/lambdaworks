import unittest

from lambdaworks_py import *

class TestU256(unittest.TestCase):
    def test_constructor(self):
        """
        Test that it can be constructed
        """
        self.assertTrue(U256("1"))
        
if __name__ == '__main__':
    unittest.main()