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

    def test_eq1(self):
        one = FieldElement(U256("1"))
        self.assertTrue(one == one)
    
    def test_eq2(self):
        one = FieldElement(U256("1"))
        another_one = FieldElement(U256("1"))
        self.assertTrue(one == another_one)
    
    def test_add(self):
        one = FieldElement(U256("1"))
        two = FieldElement(U256("2"))
        self.assertTrue(one + one == two)

    def test_sub(self):
        two = FieldElement(U256("2"))
        one = FieldElement(U256("1"))
        self.assertTrue(two - one == one)

    def test_mul(self):
        one = FieldElement(U256("1"))
        two = FieldElement(U256("2"))
        self.assertTrue(one * two == two)

    def test_div(self):
        one = FieldElement(U256("1"))
        two = FieldElement(U256("2"))
        self.assertTrue(two / one == two)

    def test_neg(self):
        one = FieldElement(U256("1"))
        two = FieldElement(U256("2"))
        self.assertTrue(two + (-one) == one)

    def test_pow(self):
        one = FieldElement.one()
        self.assertTrue(one.pow(2) == one)

    def test_pow(self):
        one = FieldElement.one()
        self.assertTrue(one ** 2 == one)

    def test_inv(self):
        one = FieldElement.one()
        two = FieldElement(U256("2"))
        self.assertTrue(two * (two.inv()) == one)
    

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

        proof_config = StarkProofConfig(count_queries=30, blowup_factor=4)
        # result = prove(trace, proof_config)

        #self.assertTrue(verify(result))
    

if __name__ == '__main__':
    unittest.main()
