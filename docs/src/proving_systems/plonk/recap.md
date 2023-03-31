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
  e * x + x - 1
```
The observer would have noticed that this program could also be written as $(e + 1) * x - 1$, which is more sensible. But the way it is written now serves us to better explain the arithmetization of PLONK. So we'll stick to it.

The idea here is that the verifier holds some value $x$, say $x=3$. He gives them to the prover. She executes the program using her own chosen value $e$, and sends the output value, say $8$, along with a proof $\pi$ demonstrating correct execution of the program and obtaining the correct output.

In the context of PLONK, both the inputs and outputs of the program are considered *public inputs*. This may sound odd, but it is because these are the inputs to the verification algorithm. This is the algorithm that takes, in this case, the tuple $(3, 8, \pi)$ and outputs *Accept* if the toy program was executed with input $x=3$, some private value $e$ not revealed to the verifier, and out came $8$. Otherwise it outputs *Reject*.

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
See the program as a sequence of gates that have left operand, a right operand and an output. The two most basic gates are multiplication and addition gates. One way of seeing our toy program is as a composition of three gates.

Gate 1: left: e, right: x, output: u = e * x
Gate 2: left: u, right: x, output: v = e + x
Gate 3: left: v, right: 1, output: w = v - 1

From this we are going to build two matrices. The first matrix only depends on the program itself and not on any particular evaluation of it. It has one row for each gate and its columns are called $Q_L, Q_R, Q_O, Q_M, Q_C$. They encode the type of gate of the row. In our example it's the following. Don't worry if you don't get where it came from nor what it means. We'll get there. 

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     0 |     0 |     1 |    -1 |     0 |
|     1 |     1 |     0 |    -1 |     0 |
|     1 |     0 |     0 |    -1 |    -1 |

One thing we can start noting here is that this matrix is a sort of *selector*. For example the 1 in the $Q_M$ column of the first row indicates that it is a multiplication gate. The 1 in the $Q_L$ column of the second row indicates that it is an addition gate. There's a $0$ in the $Q_R$ column because the right operand is constant $1$ in this program. That's why there's a 1 in the $Q_C$ column. The $Q_O$ column is there to handle outputs. We are about to see why there are negative ones there.

The second matrix also has one row for each gate. It will be a matrix with all left, right and output values of all the gates. We call the columns of this matrix $L, R, O$. Let's build them for $x=3$ and $e=2$. In each gates we get $y=6$ and $z=5$. So the first matrix is:

|   L |   R |   O |
| --- | --- | --- |
|   2 |   3 |   6 |
|   6 |   3 |   9 |
|   9 |   0 |   8 |

The last gate subtracts a constant value that is part of the program and is not a variable. That's handled a bit different from the second gate and that's why there's a $0$ in the $R$ column. Actually any value could sit there. It won't change anything. Let's see why.

These matrices are designed to satisfy the following.

**Claim:** columns $L, R, O$ correspond to a valid evaluation of the circuit if and only if for all $i$ the following equality holds $$L_i (Q_L)_i + R_i *(Q_R)_i + L_i * R_i * Q_M + c_i * (Q_O)_i + (Q_C)_i = 0$$

In our example these are three equations:
$$ 2 * 0 + 3 * 0 +  2 * 3 * 1 + 6 * (-1) +  0 $$
$$ 6 * 1 + 3 * 1 +  6 * 3 * 0 + 9 * (-1) +  0 $$
$$ 9 * 1 + 0 * 0 +  9 * 0 * 0 + 8 * (-1) + (-1) $$

and indeed all three give $0$ as a result.

So generally an addition gate is represented by the row:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     1 |     1 |     0 |    -1 |     0 |

A multiplication gate is represented by the row:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     0 |     0 |     1 |    -1 |     0 |

Addition by constant C can be represented by the row:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     1 |     0 |     0 |    -1 |     C |

Multiplication by constant C can be represented by:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     C |     0 |     0 |    -1 |     0 |

And so on. As you might have already noticed, there are several ways of representing the same gate in some cases. Additionally, PLONK has the flexibility to construct more sophisticated gates as combinations of the five columns. And therefore the same program can be expressed in multiple ways. In our case all three gates can actually be merged into a single custom gate. The first matrix is ends up being a single row.

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     1 |     1 |     1 |    -1 |     1 |

and the trace matrix for this representation is

|   L |   R |   O |
| --- | --- | --- |
|   2 |   3 |   8 |

And we check that it satisfies the equation

$$ 2 * 1 + 3 * 1 + 2 * 3 * 1 + 8 * (-1) + (-1) = 0$$

Of course, we can't always squash an entire program into a single gate.

In general 

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


