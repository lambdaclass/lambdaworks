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

        proof_config = StarkProofConfig(count_queries=30, blowup_factor=4)
        result = prove(trace, proof_config)

        self.assertTrue(verify(result))

class TestMerkleTree(unittest.TestCase):
    def test_constructor(self):
        self.assertTrue(U64MerkleTree([U64FE(1)]))

class TestU64Proof(unittest.TestCase):
    def test_constructor_raises_type_error_when_values_passed_are_not_a_list(self):
        with self.assertRaises(TypeError):
            U64Proof("bad data")
    
    def test_constructor_raises_type_error_when_an_element_is_not_a_field_element(self):
        with self.assertRaises(TypeError):
            U64Proof(["bad data"])
            
    def test_constructor(self):
        U64Proof([U64FE(1)])
        
    def test_serialize_and_deserialize_using_big_endian(self):
        proof = U64Proof([U64FE(2), U64FE(1), U64FE(1)])
        proof_reconstructed = U64Proof.from_bytes_be(proof.to_bytes_be())
        self.assertEqual(proof, proof_reconstructed)

    def test_serialize_and_deserialize_using_little_endian(self):
        proof = U64Proof([U64FE(2), U64FE(1), U64FE(1)])
        proof_reconstructed = U64Proof.from_bytes_le(proof.to_bytes_le())
        self.assertEqual(proof, proof_reconstructed)


if __name__ == '__main__':
    unittest.main()
