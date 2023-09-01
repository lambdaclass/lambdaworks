# PLONK

PLONK is a popular cryptographic proving system within the Zero Knowledge (ZK) community due to its efficiency and flexibility. It enables the verification of complex computations executed by untrusted parties through the transformation of programs into circuit representations. The system relies on a process called arithmetization, which converts logical circuits into polynomial representations. The main idea behind arithmetization is to express the computation as a set of polynomial equations. The solutions to these equations correspond to the outputs of the circuit. In this section, we will delve into the mechanics of how arithmetization works in PLONK, as well as the protocol used to generate and verify proofs.

The paper can be found [here](https://eprint.iacr.org/2019/953.pdf)

## Notation
We use the following notation.

The symbol $\mathbb{F}$ denotes a finite field. It is fixed all along. The symbol $\omega$ denotes a primitive root of unity in $\mathbb{F}$.

All polynomials have coefficients in $\mathbb{F}$ and the variable is usually denoted by $X$. We denote polynomials by single letters like $p, a, b, z$. We only denote them as $z(X)$ when we want to emphasize the fact that it is a polynomial in $X$, or we need that to explicitly define a polynomial from another one. For example when composing a polynomial $z$ with the polynomial $\omega X$, the result being denoted by $z' := z(\omega X)$. The symbol $'$ is **not** used to denote derivatives.

When interpolating at a domain $H=\{h_0, \dots, h_n\} \subset \mathbb{F}$, the symbols $L_i$ denote the Lagrange basis. That is $L_i$ is the polynomial such that $L_i(h_j) = 0$ for all $j\neq i$, and that $L_i(h_i) = 1$.

If $M$ is a matrix, then $M_{i,j}$ denotes the value at the row $i$ and column $j$.

# The ideas and components

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

The idea here is that the verifier holds some value $x$, say $x=3$. He gives it to the prover. She executes the program using her own chosen value $e$, and sends the output value, say $8$, along with a proof $\pi$ demonstrating correct execution of the program and obtaining the correct output.

In the context of PLONK, both the inputs and outputs of the program are considered _public inputs_. This may sound odd, but it is because these are the inputs to the verification algorithm. This is the algorithm that takes, in this case, the tuple $(3, 8, \pi)$ and outputs _Accept_ if the toy program was executed with input $x=3$, some private value $e$ not revealed to the verifier, and out came $8$. Otherwise it outputs _Reject_.

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

This is the process that takes the circuit of a particular program and produces a set of mathematical tools that can be used to generate and verify proofs of execution. The end result will be a set of eight polynomials. To compute them we need first to define two matrices. We call them the $Q$ matrix and the $V$ matrix. The polynomials and the matrices depend only on the program and not on any particular execution of it. So they can be computed once and used for every execution instance. To understand what they are useful for, we need to start from _execution traces_.

### Circuits and execution traces

See the program as a sequence of gates that have left operand, a right operand and an output. The two most basic gates are multiplication and addition gates. In our example, one way of seeing our toy program is as a composition of three gates.

Gate 1: left: e, right: x, output: u = e \* x
Gate 2: left: u, right: x, output: v = e + x
Gate 3: left: v, right: 1, output: w = v - 1

On executing the circuit, all these variables will take a concrete value. All that information can be put in table form. It will be a matrix with all left, right and output values of all the gates. One row per gate. We call the columns of this matrix $L, R, O$. Let's build them for $x=3$ and $e=2$. We get $u=6$, $v=9$ and $w=8$. So the first matrix is:

| A   | B   | C   |
| --- | --- | --- |
| 2   | 3   | 6   |
| 6   | 3   | 9   |
| 9   | -   | 8   |

The last gate subtracts a constant value that is part of the program and is not a variable. So it actually has only one input instead of two. And the output is the result of subtracting $1$ to it. That's why it is handled a bit different from the second gate. The symbol "-" in the $R$ column is a consequence of that. With that we mean "any value" because it won't change the result. In the next section we'll see how we implement that. Here we'll use this notation when any value can be put there. In case we have to choose some, we'll default to $0$.

What we got is a valid execution trace. Not all matrices of that shape will be the trace of an execution. The matrices $Q$ and $V$ will be the tool we need to distinguish between valid and invalid execution traces.

### The $Q$ matrix

As we said, it only depends on the program itself and not on any particular evaluation of it. It has one row for each gate and its columns are called $Q_L, Q_R, Q_O, Q_M, Q_C$. They encode the type of gate of the rows and are designed to satisfy the following.

**Claim:** if columns $L, R, O$ correspond to a valid evaluation of the circuit then for all $i$ the following equality holds $$A_i (Q_L)_i + B_i (Q_R)_i + A_i  B_i  Q_M + C_i  (Q_O)_i + (Q_C)_i = 0$$

This is better seen with examples. A multiplication gate is represented by the row:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| 0     | 0     | 1     | -1    | 0     |

And the row in the trace matrix that corresponds to the execution of that gate is

| A   | B   | C   |
| --- | --- | --- |
| 2   | 3   | 6   |

The equation in the claim for that row is that $2 \times 0 + 3 \times 0 + 2 \times 3 \times 1 + 6 \times (-1) + 0$, which equals $0$. The next is an addition gate. This is represented by the row

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| 1     | 1     | 0     | -1    | 0     |

The corresponding row in the trace matrix its

| A   | B   | C   |
| --- | --- | --- |
| 6   | 3   | 9   |

And the equation of the claim is $6 \times 1 + 3 \times 1 + 2 \times 3 \times 0 + 9 \times (-1) + 0$, which adds up to $0$. Our last row is the gate that adds a constant. Addition by constant C can be represented by the row

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| 1     | 0     | 0     | -1    | C     |

In our case $C=-1$. The corresponding row in the execution trace is

| A   | B   | C   |
| --- | --- | --- |
| 9   | -   | 8   |

And the equation of the claim is $9 \times 1 + 0 \times 0 + 9 \times 0 \times 0 + 8 \times (-1) + C$. This is also zero.

Putting it altogether, the full $Q$ matrix is

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| 0     | 0     | 1     | -1    | 0     |
| 1     | 1     | 0     | -1    | 0     |
| 1     | 0     | 0     | -1    | -1    |

And we saw that the claim is true for our particular execution:
$$ 2 \times 0 + 3 \times 0 + 2 \times 3 \times 1 + 6 \times (-1) + 0 = 0 $$
$$ 6 \times 1 + 3 \times 1 + 6 \times 3 \times 0 + 9 \times (-1) + 0 = 0 $$
$$ 9 \times 1 + 0 \times 0 + 9 \times 0 \times 0 + 8 \times (-1) + (-1) = 0 $$

Not important to our example, but multiplication by constant C can be represented by:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| C     | 0     | 0     | -1    | 0     |

As you might have already noticed, there are several ways of representing the same gate in some cases. We'll exploit this in a moment.

### The $V$ matrix

The claim in the previous section is clearly not an "if and only if" statement because the following trace columns do satisfy the equations but do not correspond to a valid execution:

| A   | B   | C   |
| --- | --- | --- |
| 2   | 3   | 6   |
| 0   | 0   | 0   |
| 20  | -   | 19  |

The $V$ matrix encodes the carry of the results from one gate to the right or left operand of a subsequent one. These are called _wirings_. Like the $Q$ matrix, it's independent of the particular evaluation. It consists of indices for all input and intermediate variables. In this case that matrix is:

| L   | R   | O   |
| --- | --- | --- |
| 0   | 1   | 2   |
| 2   | 1   | 3   |
| 3   | -   | 4   |

Here $0$ is the index of $e$, $1$ is the index of $x$, $2$ is the index of $u$, $3$ is the index of $v$ and $4$ is the index of the output $w$. Now we can update the claim to have an "if and only if" statement.

**Claim:** Let $T$ be a matrix with columns $A, B, C$. It correspond to a valid evaluation of the circuit if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i (Q_R)_i + A_i  B_i  Q_M + C_i  (Q_O)_i + (Q_C)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k, l}$ we have $T_{i,j} = T_{k, l}$.

So now our malformed example does not pass the second check.

### Custom gates

Our matrices are fine now. But they can be optimized. Let's do that to showcase this flexibility of PLONK and also reduce the size of our example.

PLONK has the flexibility to construct more sophisticated gates as combinations of the five columns. And therefore the same program can be expressed in multiple ways. In our case all three gates can actually be merged into a single custom gate. The $Q$ matrix ends up being a single row.

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| 1     | 1     | 1     | -1    | 1     |

and also the $V$ matrix

| L   | R   | O   |
| --- | --- | --- |
| 0   | 1   | 2   |

The trace matrix for this representation is just

| A   | B   | C   |
| --- | --- | --- |
| 2   | 3   | 8   |

And we check that it satisfies the equation

$$ 2 \times 1 + 3 \times 1 + 2 \times 3 \times 1 + 8 \times (-1) + (-1) = 0$$

Of course, we can't always squash an entire program into a single gate.

### Public inputs

Aside from the gates that execute the program operations, additional rows must be incorporated into these matrices. This is due to the fact that the prover must demonstrate not only that she executed the program, but also that she used the appropriate inputs. Furthermore, the proof must include an assertion of the output value. As a result, a few extra rows are necessary. In our case these are the first two and the last one. The original one sits now in the third row.

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| -1    | 0     | 0     | 0     | 3     |
| -1    | 0     | 0     | 0     | 8     |
| 1     | 1     | 1     | -1    | 1     |
| 1     | -1    | 0     | 0     | 0     |

And this is the updated $V$ matrix

| L   | R   | O   |
| --- | --- | --- |
| 0   | -   | -   |
| 1   | -   | -   |
| 2   | 0   | 3   |
| 1   | 3   | -   |

The first row is there to force the variable with index $0$ to take the value $3$. Similarly the second row forces variable with index $1$ to take the value $8$. These two will be the public inputs of the verifier. The last row checks that the output of the program is the claimed one.

And the trace matrix is now

| A   | B   | C   |
| --- | --- | --- |
| 3   | -   | -   |
| 8   | -   | -   |
| 2   | 3   | 8   |
| 8   | 8   | -   |

With these extra rows, equations add up to zero only for valid executions of the program with input $3$ and output $8$.

An astute observer would notice that by incorporating these new rows, the matrix $Q$ is no longer independent of the specific evaluation. This is because the first two rows of the $Q_C$ column contain concrete values that are specific to a particular execution instance. To maintain independence, we can remove these values and consider them as part of an extra one-column matrix called $PI$ (stands for Public Input). This column has zeros in all rows not related to public inputs. We put zeros in the $Q_C$ columns. The responsibility of filling in the $PI$ matrix is of the prover and verifier. In our example it is

| $PI$ |
| ---- |
| 3    |
| 8    |
| 0    |
| 0    |

And the final $Q$ matrix is

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
| -1    | 0     | 0     | 0     | 0     |
| -1    | 0     | 0     | 0     | 0     |
| 1     | 1     | 1     | -1    | 1     |
| 1     | -1    | 0     | 0     | 0     |

We ended up with two matrices that depend only on the program, $Q$ and $V$. And two matrices that depend on a particular evaluation, namely the $ABC$ and $PI$ matrices. The updated version of the claim is the following:

**Claim:** Let $T$ be a matrix with columns $A, B, C$. It corresponds to a evaluation of the circuit if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i (Q_R)_i + A_i B_i Q_M + C_i (Q_O)_i + (Q_C)_i + (PI)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k,l}$ we have $T_{i,j} = T_{k,l}$.

### From matrices to polynomials

In the previous section we showed how the arithmetization process works in PLONK. For a program with $n$ public inputs and $m$ gates, we constructed two matrices $Q$ and $V$, of sizes $(n + m + 1) \times 5$ and $(n + m + 1) \times 3$ that satisfy the following. Let $N = n + m + 1.$

**Claim:** Let $T$ be a $N \times 3$ matrix with columns $A, B, C$ and $PI$ a $N \times 1$ matrix. They correspond to a valid execution instance with public input given by $PI$ if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i (Q_R)_i + A_i B_i Q_M + C_i (Q_O)_i + (Q_C)_i + (PI)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k,l}$ we have $T_{i,j} = T_{k,l}$, c) $(PI)_i = 0$ for all $i>n$.

Polynomials enter now to squash most of these equations. We will traduce the set of all equations in conditions (a) and (b) to just a few equations on polynomials.

Let $\omega$ be a primitive $N$-th root of unity and let $H = {\omega^i: 0\leq i < N}$. Let $a, b, c, q_L, q_R, q_M, q_O, q_C, pi$ be the polynomials of degree at most $N$ that interpolate the columns $A, B, C, Q_L, Q_R, Q_M, Q_O, Q_C, PI$ at the domain $H$. This means for example that $a(\omega^i) = A_i$ for all $i$. And similarly for all the other columns.

With this, condition (a) of the claim is equivalent to $$a(x) q_L(x) + b(x) q_R(x) + a(x) b(x) q_M(x) + c(x) q_O(x) + q_c(x) + pi(x) = 0$$ for all $x$ in $H$.This is just by definition of the polynomials. But in polynomials land this is also equivalent to (a) there exists a polynomial $t$ such that $$a q_L + b q_R + a b q_M + c q_O + q_c + pi = z_H t$$, where $z_H$ is the polynomial $X^N -1$.

To reduce condition (b) to polynomial equations we need to introduce the concept of permutation. A permutation is a rearrangement of a set. Usually denoted $\sigma$. For finite sets it is a map from a set to itself that takes all values. In our case the set will be the set of all pairs
$$I=\{(i,j): \text{ such that }0\leq i < N, \text{ and } 0\leq j < 3\}$$
The matrix $V$ induces a permutation of this set where $\sigma((i,j))$ is equal to the indices of the _next_ occurrence of the value at position $(i,j)$. If already at the last occurrence, go to the first one. By _next_ we mean the following occurrence as if the columns were stacked on each other. Let's see how this works in the example circuit. Recall $V$ is

| L   | R   | O   |
| --- | --- | --- |
| 0   | -   | -   |
| 1   | -   | -   |
| 2   | 0   | 3   |
| 1   | 3   | -   |

The permutation in this case is the map $\sigma((0,0)) = (2,1)$, $\sigma((0,1)) = (0, 3)$, $\sigma((0,2)) = (0,2)$, $\sigma((0,3)) = (0,1)$, $\sigma((2,1)) = (0,0)$, $\sigma((3,1)) = (2,2)$, $\sigma((2,2)) = (3,1)$. For the positions with `-` values doesn't really matter right now.

It's not hard to see that condition (b) is equivalent to: for all $(i,j)\in I$, $T_{i,j} = T_{\sigma((i,j))}$.

A little less obvious is that this condition is in turn equivalent to checking whether the following sets $A$ and $B$ are equal
$$A = \{((i,j), T_{i,j}): (i,j) \in I\}$$
$$B = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}.$$
The proof this equivalence is straightforward. Give it a try!

In our example the sets in question are respectively
$$\{((0,0), T_{0,0}), ((0,1), T_{0,1}), ((0,2), T_{0,2}), ((0,3), T_{0,3}), ((2,1), T_{2,1}), ((3,1), T_{3,1}), ((2,2), T_{2,2})\},$$
and
$$\{((2,1), T_{0,0}), ((0,3), T_{0,1}), ((0,2), T_{0,2}), ((0,1), T_{0,3}), ((0,0), T_{2,1}), ((2,2), T_{3,1}), ((3,1), T_{2,2})\},$$

You can check these sets coincide by inspection. Recall our trace matrix $T$ is

| A   | B   | C   |
| --- | --- | --- |
| 3   | -   | -   |
| 8   | -   | -   |
| 2   | 3   | 8   |
| 8   | 8   | -   |

Checking equality of these sets is something that can be reduced to polynomial equations. It is a very nice method that PLONK uses. To understand it better let's start with a simpler case.

#### Equality of sets

Suppose we have two sets $A=\{a_0, a_1\}$ $B=\{b_0, b_1\}$ of two field elements in $\mathbb{F}$. And we are interested in checking whether they are equal.

One thing we could do is compute $a_0a_1$ and $b_0b_1$ and compare them. If the sets are equal, then those elements are necessarily equal.

But the converse is not true. For example the sets $A=\{4, 15\}$ and $B=\{6, 10\}$ both have $60$ as the result of the product of their elements. But they are not equal. So this is not good to check equality.

Polynomials come to rescue here. What we can do instead is consider the following sets _of polynomials_ $A'=\{a_0 + X, a_1 + X\}$, $B'=\{b_0 + X, b_1 + X\}$. Sets $A$ and $B$ are equal if and only if sets $A'$ and $B'$ are equal. This is because equality of polynomials boils down to equality of their coefficients. But the difference with $A'$ and $B'$ is that now the approach of multiplying the elements works. That is, $A'$ and $B'$ are equal if and only if $(a_0 + X)(a_1 + X) = (b_0 + X)(b_1 + X)$. This is not entirely evident but follows from a property that polynomials have, called _unique factorization_. Here the important fact is that linear polynomials act as sort of prime factors. Anyway, you can take that for granted. The last part of this trick is to use the Schwartz-Zippel lemma and go back to the land of field elements. That means, if for some random element $\gamma$ we have $(a_0 + \gamma)(a_1 + \gamma) = (b_0 + \gamma)(b_1 + \gamma)$, then with overwhelming probability the equality $(a_0 + X)(a_1 + X) = (b_0 + X)(b_1 + X)$ holds.

Putting this altogether, if for some random element $\gamma$ we have $(a_0 + \gamma)(a_1 + \gamma) = (b_0 + \gamma)(b_1 + \gamma)$, then the sets $A$ and $B$ are equal. Of course this also holds for sets with more than two elements. Let's write that down.

_Fact:_ Let $A=\{a_0, \dots, a_{k-1}\}$ and $B=\{b_0, \dots, b_{k-1}\}$ be sets of field elements. If for some random $\gamma$ the following equality holds
$$\prod_{i=0}^{k-1}(a_i + \gamma) = \prod_{i=0}^{k-1}(b_i + \gamma),$$
then with overwhelming probability $A$ is equal to $B$.

And here comes the trick that reduces this check to polynomial equations. Let
$H$ be a domain of the form $\{1, \omega, \dots, \omega^{k-1}\}$ for some primitive $k$-th root of unity $\omega$. Let $f$ and $g$ be respectively the polynomials that interpolate the following values at $H$.
$$(a_0 + \gamma, \dots, a_{k-1} + \gamma),$$
$$(b_0 + \gamma, \dots, b_{k-1} + \gamma),$$

Then $\prod_{i=0}^{k-1}(a_i + \gamma)$ equals $\prod_{i=0}^{k-1}(b_i + \gamma)$ if and only if there exists a polynomial $Z$ such that
$$Z(\omega^0) = 1$$
$$Z(h)f(h) = g(h)Z(\omega h)$$
for all $h\in H$.

Let's see why. Suppose that $\prod_{i=0}^{k-1}(a_i + \gamma)$ equals $\prod_{i=0}^{k-1}(b_i + \gamma)$. Construct $Z$ as the polynomial that interpolates the following values $$(1, \frac{a_0 + \gamma}{b_0 + \gamma}, \frac{(a_0 + \gamma)(a_1 + \gamma)}{(b_0 + \gamma)(b_1 + \gamma)}, \dots, \prod_{i=0}^{k-2} \frac{a_i + \gamma}{b_i + \gamma}),$$
in the same domain as $f$ and $g$. That works. Conversely, suppose such a polynomial $Z$ exists. By evaluating the equation $Z(X)f(X) = g(X)Z(\omega X)$ at $1, \omega, \dots, \omega^{k-2}$ and using recursion we get that $Z(\omega^{k-1}) = \prod_{i=0}^{k-2}(a_i + \gamma)/\prod_{i=0}^{k-2}(b_i + \gamma)$. Moreover, evaluating it at $\omega^{k-1}$ we obtain that $$Z(\omega^{k-1})\frac{f(\omega^{k-1})}{g(\omega^{k-1})} = Z(\omega^k) = Z(w^0) = 1.$$
The second equality holds because $\omega^k = \omega^0$ since it is a $k$-th root of unity. Expanding with the values of $f, g$ and $Z$ one obtains that $\prod_{i=0}^{k-1}(a_i + \gamma)/\prod_{i=0}^{k-1}(b_i + \gamma)$ equals $1$. Which is what we wanted.

In summary. We proved the following:

_Fact:_ Let $A=\{a_0, \dots, a_{k-1}\}$ and $B=\{b_0, \dots, b_{k-1}\}$ be sets of field elements. Let $\gamma$ be a random field element. Let $\omega$ be a primitive $k$-th root of unity and $H=\{1, \omega, \omega^2, \dots, \omega^{k-1}\}$. Let $f$ and $g$ be respectively the polynomials that interpolate the values $\{a_0 + \gamma, \dots, a_{k-1} + \gamma\}$ and $\{b_0 + \gamma, \dots, b_{k-1} + \gamma\}$ at $H$. If there exists a polynomial $Z$ such that
$$Z(\omega^0) = 1$$
$$Z(X)f(X) = g(X)Z(\omega X)$$
for all $h\in H$, then with overwhelming probability the sets $A$ and $B$ are equal.

#### Sets of tuples

In the previous section we saw how to check whether two sets of field elements are equal using polynomial equations. To be able to use it in our context we need to extend it to sets of tuples of field elements. This is pretty straightforward.

Let's start with the easy case. Let $A=\{(a_0, a_1), (a_2, a_3)\}$ and $B=\{(b_0, b_1), (b_2, b_3)\}$ be two sets of pairs of field elements. That is $a_i, b_i \in \mathbb{F}$ for all $i$. The trick is very similar to the previous section.
$$A'=\{a_0 + a_1 Y + X, a_2 + a_3 Y + X\}$$
$$B'=\{b_0 + b_1 Y + X, b_2 + b_3 Y + X\}$$

Just as before, by looking at coefficients we can see that the sets $A$ and $B$ are equal if and only if $A'$ and $B'$ are equal.
And notice that these are sets of polynomials, we got rid of the tuples! And now the situation is very similar to the previous section. We have that $A'$ and $B'$ are equal if and only if the product of their elements coincide. This is true also because polynomials in two variables are a unique factorization domain. So as before, we can use the Schwartz-Zippel lemma. Precisely, if for random $\beta, \gamma$, the elements
$$(a_0 + \beta a_1 + \gamma)(a_2 + \beta a_3 + \gamma),$$
and
$$(b_0 + \beta b_1 + \gamma)(b_2 + \beta b_3 + \gamma)$$
coincide, then $A$ and $B$ are equal with overwhelming probability.

Here is the statement for sets of more than two pairs of field elements.

_Fact:_ Let $A=\{\bar a_0, \dots, \bar a_{k-1}\}$ and $B=\{\bar b_0, \dots, \bar b_{k-1}\}$ be sets of pairs of field elements. So that $\bar a_i = (a_{i,0}, a_{i,1})$ and the same for $\bar b_i$. Let $\beta, \gamma$ be a random field elements. Let $\omega$ be a $k$-th root of unity and $H=\{1, \omega, \omega^2, \dots, \omega^{k-1}\}$. Let $f$ and $g$ be respectively the polynomials that interpolate the values
$$\{a_{i,0} + a_{i,1}\beta + \gamma, \dots, a_{k-1,0} + a_{k-1,1}\beta + \gamma\},$$
and
$$\{b_{i,0} + b_{i,1}\beta + \gamma, \dots, b_{k-1,0} + b_{k-1,1}\beta + \gamma\},$$
at $H$. If there exists a polynomial $Z$ such that
$$Z(\omega^0) = 1$$
$$Z(X)f(X) = g(X)Z(\omega X)$$
for all $h\in H$, then with overwhelming probability the sets $A$ and $B$ are equal.

#### Going back to our case

Recall we want to rephrase condition (b) in terms of polynomials. We have already seen that condition (b) is equivalent to $A$ and $B$ being equal, where
$$A = \{((i,j), T_{i,j}): (i,j) \in I\}$$
and
$$B = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}.$$

We cannot directly use the facts of the previous sections because our sets are not sets of field elements. Nor are they sets of pairs of field elements. They are sets of pairs with some indexes $(i,j)$ in the first coordinate and a field element $v$ in the second one. So the solution is to convert them to sets of pairs of field elements and apply the result of the previous section. So how do we map an element of the form $((i,j), v)$ to something of the form $(a_0, a_1)$ with $a_0$ and $a_1$ field elements? The second coordinate is trivial, we can just leave $v$ as it is and take $a_1 = v$. For the indexes pair $(i,j)$ there are multiple ways. The important thing to achieve here is that different pairs get mapped to different field elements. Recall that $i$ ranges from $0$ to $N-1$ and $j$ ranges from $0$ to $2$. One way is to take a $3N$-th primitive root of unity $\eta$ and define $a_0 = \eta^{3i + j}$. Putting it altogether, we are mapping the pair $((i,j), v)$ to the pair $(\eta^{3i + j}, v)$, which is a pair of field elements. Now we can consider the sets
$$A = \{(\eta^{3i + j}, T_{i,j}): (i,j) \in I\}$$
and
$$B = \{(\eta^{3k + l}, T_{i,j}): (i,j) \in I, \sigma((i,j)) = (k, l)\}.$$
We have that condition (b) is equivalent to $A$ and $B$ being equal.

Applying the method of the previous section to these sets, we obtain the following.

_Fact:_ Let $\eta$ be a $3N$-th root of unity and $\beta$ and $\gamma$ random field elements. Let $D = \{1, \eta, \eta^2, \dots, \eta^{3N-1}\}$. Let $f$ and $g$ be the polynomials that interpolate, respectively, the following values at $D$:
$$\{T_{i,j} + \eta^{3i + j}\beta + \gamma: (i,j) \in I\},$$
and
$$\{T_{i,j} + \eta^{3k + l}\beta + \gamma: (i,j) \in I, \sigma((i,j)) = (k,l)\},$$
Suppose there exists a polynomial $Z$ such that
$$Z(\eta^0) = 1$$
$$Z(d)f(d) = g(d)Z(\eta d),$$
for all $h\in D$.
Then the sets $A = \{((i,j), T_{i,j}): (i,j) \in I\}$ and $B = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}$ are equal with overwhelming probability.

One last minute definitions. Notice that $\omega=\eta^3$ is a primitive $N$-th root of unity. Let $H = \{1, \omega, \omega^2, \dots, \omega^{N-1}\}$.

Define $S_{\sigma 1}$ to be the interpolation at $H$ of
$$\{\eta^{3k + l}: (i,0) \in I, \sigma((i,0)) = (k,l)\},$$
Similarly define $S_{\sigma 2}$ and $S_{\sigma 3}$ to be the interpolation at $H$ of the sets of values
$$\{\eta^{3k + l}: (i,1) \in I, \sigma((i,1)) = (k,l)\},$$
$$\{\eta^{3k + l}: (i,2) \in I, \sigma((i,2)) = (k,l)\},$$
These will be useful during the protocol to work with such polynomials $Z$ and the above equations.

#### A more compact form

The last fact is equivalent the following. There's no new idea here, just a more compact form of the same thing that allows the polynomial $Z$ to be of degree at most $N$.

_Fact:_ Let $\omega$ be a $N$-th root of unity. Let $H = \{1, \omega, \omega^2, \dots, \omega^{N-1}\}$. Let $k_1$ and $k_2$ be two field elements such that $\omega^i \neq \omega^jk_1 \neq \omega^lk_2$ for all $i,j,l$. Let $\beta$ and $\gamma$ be random field elements. Let $f$ and $g$ be the polynomials that interpolate, respectively, the following values at $H$:
$$\{(T_{0,j} + \omega^{i}\beta + \gamma)(T_{1,j} + \omega^{i}k_1\beta + \gamma)(T_{2,j} + \omega^{i}k_2\beta + \gamma): 0\leq i<N\},$$
and
$$\{(T_{0,j} + S_{\sigma1}(\omega^i)\beta + \gamma)(T_{0,j} + S_{\sigma2}(\omega^i)\beta + \gamma)(T_{0,j} + S_{\sigma3}(\omega^i)\beta + \gamma): 0\leq i<N\},$$
Suppose there exists a polynomial $Z$ such that
$$Z(\omega^0) = 1$$
$$Z(d)f(d) = g(d)Z(\omega d),$$
for all $h\in D$.
Then the sets $A = \{((i,j), T_{i,j}): (i,j) \in I\}$ and $B = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}$ are equal with overwhelming probability.

## Common preprocessed input

We have arrived at the eight polynomials we mentioned at the beginning:
$$q_L, q_R, q_M, q_O, q_C, S_{\sigma 1}, S_{\sigma 2}, S_{\sigma 3}.$$

These are what's called the _common preprocessed input_.

## Wrapping up the whole thing
Let's try to wrap up what we have so far. We started from a program. We saw that it can be seen as a sequence of gates with left, right and output values. That's called a circuit. From this two matrices $Q$ and $V$ can be computed that capture the gates logic.

Executing the circuit leaves us with matrices $T$ and $PI$, called the trace matrix and the public input matrix, respectively. Everything we want to prove boils down to check that such matrices are valid. And we have the following result.

**Fact:** Let $T$ be a $N \times 3$ matrix with columns $A, B, C$ and $PI$ a $N \times 1$ matrix. They correspond to a valid execution instance with public input given by $PI$ if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i (Q_R)_i + A_i B_i Q_M + C_i (Q_O)_i + (Q_C)_i + (PI)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k,l}$ we have $T_{i,j} = T_{k,l}$, c) $(PI)_i = 0$ for all $i>n$.

Then we constructed polynomials $q_L, q_R, q_M, q_O, q_C, S_{\sigma1},S_{\sigma2}, S_{\sigma3}$, $f$, $g$ off the matrices $Q$ and $V$. They are the result of interpolating at a domain $H = \{1, \omega, \omega^2, \dots, \omega^{N-1}\}$ for some $N$-th primitive root of unity and a few random values. And also constructed polynomials $a,b,c, pi$ off the matrices $T$ and $PI$. Loosely speaking, the above fact can be reformulated in terms of polynomial equations as follows.

**Fact:** Let $z_H = X^N - 1$. Let $T$ be a $N \times 3$ matrix with columns $A, B, C$ and $PI$ a $N \times 1$ matrix. They correspond to a valid execution instance with public input given by $PI$ if and only if

a) There is a polynomial $t_1$ such that the following equality holds $$a q_L + b q_R + a b q_M + c q_O + q_C + pi = z_H t_1,$$

b) There are polynomials $t_2, t_3$, $z$ such that $zf - gz' = z_H t_2$ and $(z-1)L_1 = z_H t_3$, where $z'(X) = z(X\omega)$

You might be wondering where the polynomials $t_i$ came from. Recall that for a polynomial $F$, we have $F(h) = 0$ for all $h \in H$ if and only if $F = z_H t$ for some polynomial $t$.

Finally both conditions (a) and (b) are equivalent to a single equation (c) if we let more randomness to come into play. This is:

(c) Let $\alpha$ be a random field element. There is a polynomial $t$ such that
$$
\begin{aligned}
z_H t = &a q_L + b q_R + a b q_M + c q_O + q_C + pi \\
        &+ \alpha(gz' - fz) \\
        &+ \alpha^2(z-1)L_1 \\
\end{aligned}
$$

This last step is not obvious. You can check the paper to see the proof. Anyway, this is the equation you'll recognize below in the description of the protocol.

Randomness is a delicate matter and an important part of the protocol is where it comes from, who chooses it and when they choose it. Check out the protocol to see how it works.

