## PLONK

PLONK is a popular cryptographic proving system within the Zero Knowledge (ZK) community due to its efficiency and flexibility. It enables the verification of complex computations executed by untrusted parties through the transformation of programs into circuit representations. The system relies on a process called arithmetization, which converts logical circuits into polynomial representations. The main idea behind arithmetization is to express the computation as a set of polynomial equations. The solutions to these equations correspond to the outputs of the circuit. In this section, we will delve into the mechanics of how arithmetization works in PLONK, as well as the protocol used to generate and verify proofs.


## Programs. Our toy example
For better clarity, we'll be using the following toy program throughout this recap.
```
INPUT:
  x

PRIVATE INPUT:
  e

OUTPUT:
  e * x - 1
```
The idea here is that the verifier holds some value $x$, say $x=3$. He gives them to the prover. She executes the program using her own chosen value $e$, and sends the output value, say $5$, along with a proof $\pi$ demonstrating correct execution of the program and obtaining the correct output.

In the context of PLONK, both the inputs and outputs of the program are considered *public inputs*. This may sound odd, but it is because these are the inputs to the verification algorithm. This is the algorithm that takes, in this case, the tuple $(3, 5, \pi)$ and outputs *Accept* if the toy program was executed with input $x=3$, some private value $e$ not revealed to the verifier, and out came $5$. Otherwise it outputs *Reject*.

PLONK can be used to delegate program executions to untrusted parties, but it can also be used as a proof of knowledge. Our program could be used by a prover to demostrate that she knows the multiplicative inverse of some value $x$ in the finite field without revealing it. She would do it by sending the verifier the tuple $(x, 0, \pi)$, where $\pi$ is the proof of the execution of our toy program.

In our toy example this is pointless because inverting field elements is easily performed by any verifier. But change our program to the following and you get proofs of knowledge of the preimage of SHA256 digests.
```
PRIVATE INPUT:
  e

OUTPUT:
  SHA256(e)
```
Here there's no input aside from the prover's private input. As we mentioned, the output $h$ of the program is then part of the inputs to the verification algorithm. Which in this case just takes $(h, \pi)$.

## PLONK Arithmetization
See the program as a sequence of gates that have left operand, a right operand and an output. The two most basic gates are multiplication and addition gates. One way of seeing our toy program is as a composition of two gates.
Gate 1: left: e, right: x, output: y = e * x
Gate 2: left: y, right: 1, output: z = y - 1

For $x=3$ and $e=2$ we get $y=6$ and $z=5$. We put this information in a table format like so, with three columns.
a, b, c
2, 3, 6
6, 1, 5

The fact that this table was constructed from the intermediate values of the gates evaluations implies that they satisfy algebraic equations. For example $a0 * b0 = c0$ and $a1 + b1 = c1$. Also $b1 = 1$.

PLONK has the flexibility to construct more sophisticated gates as combinations of those two. We'll see those in just a moment. For now let's start with addition and multiplication.


## Polynomial Commitment scheme

### Structured Reference String
A SRS is essentially a set of precomputed values that are agreed upon by all parties involved in a PLONK proof. These values serve as a kind of baseline or starting point for verifying the correctness of the proof.

The reason why a SRS is so crucial to PLONK is that it allows for efficient and scalable verification. Without a SRS, verifying a PLONK proof would require a lot of computational power and time, since it would involve complex calculations that would have to be performed from scratch each time a proof needed to be verified. With a SRS, however, the verification process becomes much simpler and faster, since the precomputed values can be reused across multiple proofs of different programs.

### Kate Zaverucha Goldberg

### Batched Kate Zaverucha Goldberg 

## Setup

## Proving algorithm

### Round 1
### Round 2
### Round 3
### Round 4
### Round 5


## Verification algorithm


