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

    @unittest.expectedFailure
    def test_div_zero(self):
        one = FieldElement(U256("1"))
        zero = FieldElement(U256("0"))
        one / zero

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

if __name__ == '__main__':
    unittest.main()
