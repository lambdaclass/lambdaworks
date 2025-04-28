# lambdaworks MultiScalar Multiplication (MSM)

This contains implementations for the MultiScalar Multiplication (MSM):
- Naïve
- Pippenger

[Multiscalar multiplication](https://blog.lambdaclass.com/multiscalar-multiplication-strategies-and-challenges/) is an important primitive that appears in some polynomial commitment schemes and proof systems. It is also at the core of [EIP-4844](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-4844.md). Given a set of scalars in a [finite field](../field/README.md) $a_0, a_1, ..., a_n$ and [elliptic curve points](../elliptic_curve/README.md) $P_0, P_1, ... , P_n$, the MSM computes
$$P = \sum_k a_k P_k$$
where $a_k P_k$ is understood as applying the group operation with $P_k$ a number of $a_k$ times. For its application in a protocol, see [KZG](../../../crypto/src/commitments/README.md) 