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

The idea here is that the verifier holds some value $x$, say $x=3$. He gives it to the prover. She executes the program using her own chosen value $e$, and sends the output value, say $8$, along with a proof $\pi$ demonstrating correct execution of the program and obtaining the correct output.

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
This is the process that takes the circuit of a particular program and produces a set of mathematical tools that can be used to generate and verify proofs of execution. The end result will be a set of eight polynomials. To compute them we need first to define two matrices. We call them the $Q$ matrix and the $V$ matrix. The polynomials and the matrices depend only on the program and not on any particular execution of it. So they can be computed once and used for every execution instance. To understand what they are useful for, we need to start from *execution traces*.

### Circuits and execution traces
See the program as a sequence of gates that have left operand, a right operand and an output. The two most basic gates are multiplication and addition gates. In our example, one way of seeing our toy program is as a composition of three gates.

Gate 1: left: e, right: x, output: u = e * x
Gate 2: left: u, right: x, output: v = e + x
Gate 3: left: v, right: 1, output: w = v - 1

On executing the circuit, all these variables will take a concrete value. All that information can be put in table form. It will be a matrix with all left, right and output values of all the gates. One row per gate. We call the columns of this matrix $L, R, O$. Let's build them for $x=3$ and $e=2$. We get $u=6$, $v=9$ and $w=5$. So the first matrix is:

|   A |   B |   C |
| --- | --- | --- |
|   2 |   3 |   6 |
|   6 |   3 |   9 |
|   9 |   - |   8 |

The last gate subtracts a constant value that is part of the program and is not a variable. So it actually has only one input instead of two. And the output is the result of subtracting $1$ to it. That's why it is handled a bit different from the second gate. The symbol "-" in the $R$ column is a consequence of that. With that we mean "any value" because it won't change the result. In the next section we'll see how we implement that. Here we'll use this notation when any value can be put there. In case we have to choose some, we'll default to $0$.

What we got is a valid execution trace. Not all matrices of that shape will be the trace of an execution. The matrices $Q$ and $V$ will be the tool we need to distinguish between valid and invalid execution traces.

### The $Q$ matrix

As we said, it only depends on the program itself and not on any particular evaluation of it. It has one row for each gate and its columns are called $Q_L, Q_R, Q_O, Q_M, Q_C$. They encode the type of gate of the rows and are designed to satisfy the following.

**Claim:** if columns $L, R, O$ correspond to a valid evaluation of the circuit then for all $i$ the following equality holds $$A_i (Q_L)_i + B_i *(Q_R)_i + A_i * B_i * Q_M + C_i * (Q_O)_i + (Q_C)_i = 0$$

This is better seen with examples. A multiplication gate is represented by the row:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     0 |     0 |     1 |    -1 |     0 |

And the row in the trace matrix that corresponds to the execution of that gate is 

|   A |   B |   C |
| --- | --- | --- |
|   2 |   3 |   6 |

The equation in the claim for that row is that $2 * 0 + 3 * 0 + 2 * 3 * 1 + 6 * (-1) + 0$, which equals $0$. The next is an addition gate. This is represented by the row

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     1 |     1 |     0 |    -1 |     0 |

The corresponding row in the trace matrix its

|   A |   B |   C |
| --- | --- | --- |
|   6 |   3 |   9 |

And the equation of the claim is $6 * 1 + 3 * 1 + 2 * 3 * 0 + 9 * (-1) + 0$, which adds up to $0$. Our last row is the gate that adds a constant. Addition by constant C can be represented by the row

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     1 |     0 |     0 |    -1 |     C |

In our case $C=-1$. The corresponding row in the execution trace is 

|   A |   B |   C |
| --- | --- | --- |
|   9 |   - |   8 |

And the equation of the claim is $9 * 1 + 0 * 0 + 9 * 0 * 0 + 8 * (-1) + C$. This is also zero.

Putting it altogether, the full $Q$ matrix is

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     0 |     0 |     1 |    -1 |     0 |
|     1 |     1 |     0 |    -1 |     0 |
|     1 |     0 |     0 |    -1 |    -1 |

And we saw that the claim is true for our particular execution:
$$ 2 * 0 + 3 * 0 +  2 * 3 * 1 + 6 * (-1) +  0 = 0 $$
$$ 6 * 1 + 3 * 1 +  6 * 3 * 0 + 9 * (-1) +  0 = 0 $$
$$ 9 * 1 + 0 * 0 +  9 * 0 * 0 + 8 * (-1) + (-1) = 0 $$

Not important to our example, but multiplication by constant C can be represented by:

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     C |     0 |     0 |    -1 |     0 |

As you might have already noticed, there are several ways of representing the same gate in some cases. We'll exploit this in a moment.

### The $V$ matrix

The claim in the previous section is clearly not an "if and only if" statement because the following trace columns do satisfy the equations but do not correspond to a valid execution:

|   A |   B |   C |
| --- | --- | --- |
|   2 |   3 |   6 |
|   0 |   0 |   0 |
|  20 |   - |  19 |

The $V$ matrix encodes the carry of the results from one gate to the right or left operand of a subsequent one. These are called *wirings*. Like the $Q$ matrix, it's independent of the particular evaluation. It consists of indices for all input and intermediate variables. In this case that matrix is:

|   L |   R |   O |
| --- | --- | --- |
|   0 |   1 |   2 |
|   2 |   1 |   3 |
|   3 |   - |   4 |

Here $0$ is the index of $e$, $1$ is the index of $x$, $2$ is the index of $u$, $3$ is the index of $v$ and $4$ is the index of the output $w$. Now we can update the claim to have an "if and only if" statement.

**Claim:** Let $T$ be a matrix with columns $A, B, C$. It correspond to a valid evaluation of the circuit if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i *(Q_R)_i + A_i * B_i * Q_M + C_i * (Q_O)_i + (Q_C)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k, l}$ we have $T_{i,j} = T_{k, l}$.

So now our malformed example does not pass the second check.

### Custom gates

Our matrices are fine now. But they can be optimized. Let's do that to showcase this flexibility of PLONK and also reduce the size of our example.

PLONK has the flexibility to construct more sophisticated gates as combinations of the five columns. And therefore the same program can be expressed in multiple ways. In our case all three gates can actually be merged into a single custom gate. The $Q$ matrix ends up being a single row.

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|     1 |     1 |     1 |    -1 |     1 |

and also the $V$ matrix

|   L |   R |   O |
| --- | --- | --- |
|   0 |   1 |   2 |

The trace matrix for this representation is just

|   A |   B |   C |
| --- | --- | --- |
|   2 |   3 |   8 |

And we check that it satisfies the equation

$$ 2 * 1 + 3 * 1 + 2 * 3 * 1 + 8 * (-1) + (-1) = 0$$

Of course, we can't always squash an entire program into a single gate.

### Public inputs

Aside from the gates that execute the program operations, additional rows must be incorporated into these matrices. This is due to the fact that the prover must demonstrate not only that she executed the program, but also that she used the appropriate inputs. Furthermore, the proof must include an assertion of the output value. As a result, a few extra rows are necessary. In our case these are the first two and the last one. The original one sits now in the third row.

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|    -1 |     0 |     0 |     0 |     3 |
|    -1 |     0 |     0 |     0 |     8 |
|     1 |     1 |     1 |    -1 |     1 |
|     1 |    -1 |     0 |     0 |     0 |

And this is the updated $V$ matrix

|   L |   R |   O |
| --- | --- | --- |
|   0 |   - |   - |
|   1 |   - |   - |
|   2 |   0 |   3 |
|   1 |   3 |   - |

The first row is there to force the variable with index $0$ to take the value $3$. Similarly the second row forces variable with index $1$ to take the value $8$. These two will be the public inputs of the verifier. The last row checks that the output of the program is the claimed one.

And the trace matrix is now

|   A |   B |   C |
| --- | --- | --- |
|   3 |   - |   - |
|   8 |   - |   - |
|   2 |   3 |   8 |
|   8 |   8 |   - |

With these extra rows, equations add up to zero only for valid executions of the program with input $3$ and output $8$.

An astute observer would notice that by incorporating these new rows, the matrix $Q$ is no longer independent of the specific evaluation. This is because the first two rows of the $Q_C$ column contain concrete values that are specific to a particular execution instance. To maintain independence, we can remove these values and consider them as part of an extra one-column matrix called $PI$ (stands for Public Input). This column has zeros in all rows not related to public inputs. We put zeros in the $Q_C$ columns. The responsibility of filling in the $PI$ matrix is of the prover and verifier. In our example it is

| $PI$  |
| ----- |
|     3 |
|     8 |
|     0 |
|     0 |

And the final $Q$ matrix is

| $Q_L$ | $Q_R$ | $Q_M$ | $Q_O$ | $Q_C$ |
| ----- | ----- | ----- | ----- | ----- |
|    -1 |     0 |     0 |     0 |     0 |
|    -1 |     0 |     0 |     0 |     0 |
|     1 |     1 |     1 |    -1 |     1 |
|     1 |    -1 |     0 |     0 |     0 |

We ended up with two matrices that depend only on the program, $Q$ and $V$. And two matrices that depend on a particular evaluation, namely the $ABC$ and $PI$ matrices. The updated version of the claim is the following:

**Claim:** Let $T$ be a matrix with columns $A, B, C$. It corresponds to a evaluation of the circuit if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i * (Q_R)_i + A_i * B_i * Q_M + C_i * (Q_O)_i + (Q_C)_i + (PI)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k,l}$ we have $T_{i,j} = T_{k,l}$.

### From matrices to polynomials
In the previous section we showed how the arithmetization process works in PLONK. For a program with $n$ public inputs and $m$ gates, we constructed two matrices $Q$ and $V$, of sizes $(n + m + 1) \times 5$ and $(n + m + 1) \times 3$ that satisfy the following. Let $N = n + m + 1.

**Claim:** Let $T$ be a $N \times 3$ matrix with columns $A, B, C$ and $PI$ a $N \times 1$ matrix. They correspond to a valid execution instance with public input given by $PI$ if and only if a) for all $i$ the following equality holds $$A_i (Q_L)_i + B_i * (Q_R)_i + A_i * B_i * Q_M + C_i * (Q_O)_i + (Q_C)_i + (PI)_i = 0,$$ b) for all $i,j,k,l$ such that $V_{i,j} = V_{k,l}$ we have $T_{i,j} = T_{k,l}$, c) $(PI)_i = 0$ for all $i>n$.

Polynomials enter now to squash most of these equations. We will traduce the set of all equations in conditions (a) and (b) to just a few equations on polynomials.

Let $\omega$ be an N root of unity and let $H = {\omega^i: 0\leq i < N}$. Let $a, b, c, q_L, q_R, q_M, q_O, q_C, pi$ be the polynomials of degree at most $n + m$ that interpolate the columns $A, B, C, Q_L, Q_R, Q_M, Q_O, Q_C, PI$ at the domain $H$. This means for example that $a(\omega^i) = A_i$ for all $i$. And similarly for all the other columns.

With this, condition (a) of the claim is equivalent to $$a(x) * q_L(x) + b(x) * q_R(x) + a(x) * b(x) * q_M(x) + c(x) * q_O(x) + q_c(x) + pi(x) = 0$$ for all $x$ in $H$.This is just by definition of the polynomials. But in polynomials land this is also equivalent to (a) there exists a polynomial $t$ such that $$a * q_L + b * q_R + a * b * q_M + c * q_O + q_c + pi = z_H * t$$, where $z_H$ is the polynomial $X^N -1$.

To reduce condition (b) to polynomial equations we need to introduce the concept of permutation. A permutation is a rearrangement of a set. Usually denoted $\sigma$. For finite sets it is a map from a set to itself that takes all values. In our case the set will be the set of all pairs
$$I=\{(i,j): \text{ such that }0\leq i < N, \text{ and } 0\leq j < 3\}$$
The matrix $V$ induces a permutation of this set where $\sigma((i,j))$ is equal to the indices of the *next* occurrence of the value at position $(i,j)$. If already at the last occurrence, go to the first one. By *next* we mean the following occurrence as if the columns were stacked on each other. Let's see how this works in the example circuit. Recall $V$ is 

|   L |   R |   O |
| --- | --- | --- |
|   0 |   - |   - |
|   1 |   - |   - |
|   2 |   0 |   3 |
|   1 |   3 |   - |

The permutation in this case is the map $\sigma((0,0)) = (2,1)$, $\sigma((0,1)) = (0, 3)$, $\sigma((0,2)) = (0,2)$, $\sigma((0,3)) = (0,1)$, $\sigma((2,1)) = (0,0)$, $\sigma((3,1)) = (2,2)$, $\sigma((2,2)) = (3,1)$. For the positions with `-` values doesn't really matter right now.

It's not hard to see that condition (b) is equivalent to: for all $(i,j)\in I$, $T_{i,j} = T_{\sigma((i,j))}$.

This in turn is equivalent to checking whether the following sets are equal 
$$\{((i,j), T_{i,j}): (i,j) \in I\} = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}.$$

In our example the sets in question are respectively
$$\{((0,0), T_{0,0}), ((0,1), T_{0,1}), ((0,2), T_{0,2}), ((0,3), T_{0,3}), ((2,1), T_{2,1}), ((3,1), T_{3,1}), ((2,2), T_{2,2})\},$$
and
$$\{((2,1), T_{0,0}), ((0,3), T_{0,1}), ((0,2), T_{0,2}), ((0,1), T_{0,3}), ((0,0), T_{2,1}), ((2,2), T_{3,1}), ((3,1), T_{2,2})\},$$

You can check these sets coincide by inspection. Recall our trace matrix $T$ is

|   A |   B |   C |
| --- | --- | --- |
|   3 |   - |   - |
|   8 |   - |   - |
|   2 |   3 |   8 |
|   8 |   8 |   - |

Checking equality of these sets is something that can be reduced to polynomial equations. It is a very nice method that PLONK uses. To understand it better let's start with a simpler case.

#### Equality of sets
Suppose we have two sets $A=\{a_0, a_1\}$ $B=\{b_0, b_1\}$ of two field elements in $\mathbb{F}$. And we are interested in checking whether they are equal.

One thing we could do is compute $a_0a_1$ and $b_0b_1$ and compare them. If the sets are equal, then those elements are necessarily equal.

But the converse is not true. For example the sets $A=\{4, 15\}$ and $B=\{6, 10\}$ both have $60$ as the result of the product of their elements. But they are not equal. So this is not good to check equality.

Polynomials come to rescue here. What we can do instead is consider the following sets *of polynomials* $A'=\{a_0 + X, a_1 + X\}$, $B'=\{b_0 + X, b_1 + X\}$. Sets $A$ and $B$ are equal if and only if sets $A'$ and $B'$ are equal. This is because equality of polynomials boils down to equality of their coefficients. But the difference with $A'$ and $B'$ is that now the approach of multiplying the elements works. That is, $A'$ and $B'$ are equal if and only if $(a_0 + X)(a_1 + X) = (b_0 + X)(b_1 + X)$. This is not entirely evident but follows from a property that polynomials have, called *unique factorization domain*. Here the important fact is that linear polynomials act as sort of prime factors. Anyway, you can take that for granted. The last part of this trick is to use the Schwarz-Zippel lemma and go back to the land of field elements. That means, if for some random element $\gamma$ we have $(a_0 + \gamma)(a_1 + \gamma) = (b_0 + \gamma)(b_1 + \gamma)$, then with overwhelming probability the equality $(a_0 + X)(a_1 + X) = (b_0 + X)(b_1 + X)$ holds.

Puttings this altogether, if for some random element $\gamma$ we have $(a_0 + \gamma)(a_1 + \gamma) = (b_0 + \gamma)(b_1 + \gamma)$, then the sets $A$ and $B$ are equal. Of course this also holds for sets with more than two elements. Let's write that down.

*Fact:* Let $A=\{a_0, \dots, a_{k-1}\}$ and $B=\{b_0, \dots, b_{k-1}\}$ be sets of field elements. If for some random $\gamma$ the following equality holds
$$\prod_{i=0}^k(a_i + \gamma) = \prod_{i=0}^k(b_i + \gamma),$$
then with overwhelming probability $A$ is equal to $B$.

And here comes the trick that reduces this check to a polynomial equation. Let
$H$ be a domain of the form $\{1, \omega, \dots, \omega^{k-1}\}$ for some $k$-th root of unity $\omega$. Let $f, g, Z$ be respectively the polynomials that interpolate the following values in $H$.
$$(a_0 + \gamma, \dots, a_{k-1} + \gamma),$$
$$(b_0 + \gamma, \dots, b_{k-1} + \gamma),$$

Then $\prod_{i=0}^k(a_i + \gamma)$ equals $\prod_{i=0}^k(b_i + \gamma)$ if and only if there exists a polynomial $Z$ of degree at most $k$ such that
$$Z(\omega^0) = 1$$
$$Z(X)g(X) = f(X)Z(\omega X)$$

Let's see why. Suppose that $\prod_{i=0}^k(a_i + \gamma)$ equals $\prod_{i=0}^k(b_i + \gamma)$. Construct $Z$ as the polynomial that interpolates the following values $$(1, \frac{a_0 + \gamma}{b_0 + \gamma}, \frac{(a_0 + \gamma)(a_1 + \gamma)}{(b_0 + \gamma)(b_1 + \gamma)}, \dots, \prod_{i=0}^{k-1} \frac{a_i + \gamma}{b_i + \gamma}),$$
in the same domain as $f$ and $g$. That works. Conversely, suppose such a polynomial $Z$ exists. By evaluating the equation $Z(X)g(X) = f(X)Z(\omega X)$ in $1, \omega, \dots, \omega^{k-2}$ we get that $Z$ actually is the polynomial that interpolates those values. Moreover, evaluating it in $\omega^{k-1}$ we obtain that $$Z(\omega^{k-1})\frac{f(\omega^{k-1})}{g(\omega^{k-1})} = Z(\omega^k) = Z(w^0) = 1.$$
The second equality holds because $\omega^k = \omega^0$ since it is a $k$-th root of unity. Expanding with the values of $f, g$ and $Z$ one obtains that $\prod_{i=0}^k(a_i + \gamma)/\prod_{i=0}^k(b_i + \gamma)$ equals $1$. Which is what we wanted.

In summary. We proved the following:

*Fact:* Let $A=\{a_0, \dots, a_{k-1}\}$ and $B=\{b_0, \dots, b_{k-1}\}$ be sets of field elements. Let $\gamma$ be a random field element. Let $\omega$ be a $k$-th root of unity. Let $f$ and $g$ be respectively the polynomials that interpolate the values ${a_0 + \gamma, \dots, a_{k-1} + \gamma}$ and ${b_0 + \gamma, \dots, b_{k-1} + \gamma}$ in the powers of $\omega$. If there exists a polynomial $Z$ of degree at most $k$ such that 
$$Z(\omega^0) = 1$$
$$Z(X)g(X) = f(X)Z(\omega X)$$
then with overwhelming probability sets $A$ and $B$ are equal.

#### Going back to our case
Recall we want to rephrase condition (b) in terms of polynomials. And we are going to apply the previous fact to the sets we need.
We have: 
$$A = \{((i,j), T_{i,j}): (i,j) \in I\}$$
and 
$$B = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}$$
And we have already seen that condition (b) is equivalent to $A$ and $B$ being equal.

We cannot directly use the above fact because our sets are not sets of field elements. They are sets of pairs with some indexes $(i,j)$ in the first coordinate and a field element $v$ in the second one. To solve this, we are going to map each of these pairs to a single field element. 

For this purpose, we first need a few things. Recall that $N$ is the size of each of the columns of the trace matrix $T$ and our sets have now an element for each of the entries of the matrix. That's $3N$ elements. So we need $\eta$ a $3N$-th root of unity. Additionally, sample a random field element $\beta$. With all this we can do the following. For a pair $((i,j), v)$, define the field element: $$v + \eta^{j + 3i} \beta.$$
Using this we transform our set $A$ into
$$\hat A = \{T_{i,j} + \eta^{j + 3i}\beta: (i,j) \in I\}$$
and transform our set $B$ into

$$\hat B = \{T_{i,j} + \eta^{l + 3k}\beta: (i,j) \in I, (k,l) = \sigma((i,j))\}$$
A very similar argument using polynomials as before can be applied here to deduce that if these new sets $\hat A$, $\hat B$ are equal, then with overwhelming probability the original sets $A$ and $B$ are equal. The element $\beta$ comes from applying the Schwarz-Zippel lemma again. But we leave that to the reader to keep these notes short. 

The gain here is that $\hat A$ and $\hat B$ are sets of field elements and we can apply the fact from before!

Recall in our example the sets in question are respectively
$$A = \{((0,0), T_{0,0}), ((0,1), T_{0,1}), ((0,2), T_{0,2}), \dots\},$$
and
$$B = \{((2,1), T_{0,0}), ((0,3), T_{0,1}), ((0,2), T_{0,2}), \dots\},$$

And for these, the sets $\hat A$ and $\hat B$ are 
$$\hat A = \{T_{0,0} + \eta^{0}\beta, T_{0,1} + \eta^{1}\beta, T_{0,2} + \eta^{2}\beta, \dots\}$$
$$\hat B = \{T_{0,0} + \eta^{7}\beta, T_{0,1} + \eta^{3}\beta, T_{0,2} + \eta^{2}\beta, \dots\},$$

Again you can check that these are equal by replacing the values of $T_{i,j}$.

So finally we arrive at

*Fact:* The sets $A = \{((i,j), T_{i,j}): (i,j) \in I\}$ and $B = \{(\sigma((i,j)), T_{i,j}): (i,j) \in I\}$ are equal if and only if there exists a polynomial $Z$ of degree at most $3N$ such that 
$$Z(\eta^0) = 1$$
$$Z(X)g(X) = f(X)Z(\eta X),$$
where $f$ and $g$ interpolate the values following values, respectively
$$\{T_{i,j} + \eta^{j + 3i} + \gamma: (i,j) \in I\},$$
and
$$\{T_{i,j} + \eta^{l + 3k} + \gamma: (i,j) \in I, \sigma((i,j)) = (k,l)\},$$


There's one extra optimization to be done here, that can reduce the degree of the polynomial $Z$ to $N$ instead of $3N$. But for now let's leave it like that.


#### Recap



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



