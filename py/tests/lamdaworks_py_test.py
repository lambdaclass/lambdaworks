import unittest

from lambdaworks_py import *

class TestU256(unittest.TestCase):
    def test_constructor(self):
        """
        Test that it can be constructed
        """
        self.assertTrue(U256("1"))
        
class TestFieldElement(unittest.TestCase):
    def test_constructor(self):
        self.assertTrue(FieldElement(U256("1")))
        
class TestFibonacci(unittest.TestCase):
    @classmethod
    def _fibonacci_trace(cls, v0, v1):
        N = 32
        l = [v0, v1]
        for i in range(2, N):
            l.append(l[i - 1] + l[i - 2])
        return l

    def test_prove(self):
        trace = self.__class__._fibonacci_trace(FieldElement(U256("1")), FieldElement(U256("1")))

        proof_config = ProofConfig(count_queries=30, blowup_factor=4)
        result = prove(trace, proof_config)

        self.assertTrue(verify(result))
    

if __name__ == '__main__':
    unittest.main()