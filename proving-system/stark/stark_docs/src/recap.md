# STARKs Recap

# Verifying Computation through Polynomials

In general, we express computation in our proving system by providing an *execution trace* satisfying certain *constraints*. The execution trace is a table containing the state of the system at every step of computation. This computation needs to follow certain rules to be valid; these rules are our *constraints*.

The constraints for our computation are expressed using an `Algebraic Intermediate Representation` or `AIR`. This representation uses polynomials to encode constraints, which is why sometimes they are called `polynomial constraints`.

To make all this less abstract, let's go through two examples.

## Fibonacci numbers

Throughout this section and the following we will use this example extensively to have a concrete example. Even though it's a bit contrived (no one cares about computing fibonacci numbers), it's simple enough to be useful. STARKs and proving systems in general are very abstract things; having an example in mind is essential to not get lost.

Let's say our computation consists of calculating the `k`-th number in the fibonacci sequence. This is just the sequence of numbers $a_n$ satisfying

$$
a_0 = 1 \\
a_1 = 1 \\
a_{n+2} = a_{n + 1} + a_n
$$

An execution trace for this just consists of a table with one column, where each row is the `i`-th number in the sequence:

| a_i    | 
| ------ |
| 1      |
| 1      |
| 2      |
| 3      |
| 5      |
| 8      |
| 13      |
| 21      |

A `valid` trace for this computation is a table satisfying two things:

- The first two rows are `1`.
- The value on any other row is the sum of the two preceding ones.

The first item is called a `boundary constraint`, it just enforces specific values on the trace at certain points. The second one is a `transition constraint`; it tells you how to go from one step of computation to the next.

## Cairo

The example above is extremely useful to have a mental model, but it's not really useful for anything else. The problem is it just works for the very narrow example of computing fibonacci numbers. If we wanted to prove execution of something else, we would have to write an `AIR` for it.

What we're actually aiming for is an `AIR` for an entire general purpose `Virtual Machine`. This way, we can provide proofs of execution for *any* computation using just one `AIR`. This is what [cairo](https://www.cairo-lang.org/docs/) as a programming language does. Cairo code compiles to the bytecode of a virtual machine with an already defined `AIR`. The general flow when using cairo is the following:

- User writes a cairo program.
- The program is compiled into Cairo's VM bytecode.
- The VM executes said code and provides an execution trace for it.
- The trace is passed on to a STARK prover, which creates a proof of correct execution according to Cairo's AIR.
- The proof is passed to a verifier, who checks that the proof is valid.

Ultimately, our goal is to give the tools to write a STARK prover for the cairo VM and do so. However, this is not a good example to start out as it's incredibly complex. The execution trace of a cairo program has around 30 columns, some for general purpose registers, some for other reasons. Cairo's AIR contains a lot of different transition constraints, encoding all the different possible instructions (arithmetic operations, jumps, etc).

Use the fibonacci example as your go-to for understanding all the moving parts; keep the Cairo example in mind as the thing we are actually building towards.

# Step by step walkthrough

Below we go through a step by step explanation of a STARK prover. We will assume the trace of the fibonacci sequence mentioned above; it consists of only one column of length $2^n$. In this case, we'll take `n=3`. The trace looks like this

| a_i    | 
| ------ |
| a_0      |
| a_1      |
| a_2      |
| a_3      |
| a_4      |
| a_5      |
| a_6      |
| a_7      |

## Trace polynomial

The first step is to interpolate these values to generate the `trace polynomial`. This will be a polynomial encoding all the information about the trace. The way we do it is the following: in the finite field we are working in, we take an `8`-th primitive root of unity, let's call it `g`. It being a primitive root means two things:

- `g` is an `8`-th root of unity, i.e., $g^8 = 1$.
- Every `8`-th root of unity is of the form $g^i$ for some $0 \leq i \leq 7$.

With `g` in hand, we take the trace polynomial `t` to be the one satisfying

$$
t(g^i) = a_i
$$

From here onwards, we will talk about the validity of the trace in terms of properties that this polynomial must satisfy. We will also implicitly identify a certain power of $g$ with its corresponding trace element, so for example we sometimes think of $g^5$ as $a_5$, the fifth row in the trace, even though technically it's $t$ *evaluated in* $g^5$ that equals $a_5$.

We talked about two different types of constraints the trace must satisfy to be valid. They were:

- The first two rows are `1`.
- The value on any other row is the sum of the two preceding ones.

In terms of `t`, this translates to

- $t(g^0) = 1$ and $t(g) = 1$.
- $t(x g^2) - t(xg) - t(x) = 0$ for all $x \in \{g^0, g^1, g^2, g^3, g^4, g^5\}$. This is because multiplying by `g` is the same as advancing a row in the trace.

## Composition Polynomial

To convince the verifier that the trace polynomial satisfies the relationships above, the prover will construct another polynomial that shows that both the boundary and transition constraints are satisfied and commit to it. We call this polynomial the `composition polynomial`, and usually denote it with $H$. Constructing it involves a lot of different things, so we'll go step by step introducing all the moving parts required.

### Boundary polynomial

To show that the boundary constraints are satisfied, we construct the `boundary polynomial`. Recall that our boundary constraints are $t(g^0) = t(g) = 1$. Let's call $P$ the polynomial that interpolates these constraints, that is, $P$ satisfies:

$$
P(1) = 1 \\
P(g) = 1
$$

The boundary polynomial $B$ is defined as follows:

$$
B(x) = \dfrac{t(x) - P(x)}{(x - 1) (x - g)}
$$

The denominator here is called the `boundary zerofier`, and it's the polynomial whose roots are the elements of the trace where the boundary constraints must hold.

How does $B$ encode the boundary constraints? The idea is that, if the trace satisfies said constraints, then

$$
t(1) - P(1) = 1 - 1 = 0 \\
t(g) - P(g) = 1 - 1 = 0
$$

so $t(x) - P(x)$ has $1$ and $g$ as roots. Showing these values are roots is the same as showing that $B(x)$ is a polynomial instead of a rational function, and that's why we construct $B$ this way.

### Transition constraint polynomial
To convince the verifier that the transition constraints are satisfied, we construct the `transition constraint polynomial` and call it $C(x)$. It's defined as follows:

$$
C(x) = \dfrac{t(xg^2) - t(xg) - t(x)}{\prod_{i = 0}^{5} (x - g^i)}
$$

How does $C$ encode the transition constraints? We mentioned above that these are satisfied if the polynomial in the numerator vanishes in the elements $\{g^0, g^1, g^2, g^3, g^4, g^5\}$. As with $B$, this is the same as showing that $C(x)$ is a polynomial instead of a rational function.

### Constructing $H$
With the boundary and transition constraint polynomials in hand, we build the `composition polynomial` $H$ as follows: The verifier will sample four numbers $\alpha_1, \alpha_2, \beta_1, \beta_2$ and $H$ will be

$$
H(x) = B(x) (\alpha_1 x^{D - deg(B)} + \beta_1) + C(x) (\alpha_2 x^{D - deg(C)} + \beta_2)
$$

where $D$ is the smallest power of two greater than the degrees of both $B$ and $C$, so for example if $deg(B) = 3$ and $deg(C) = 6$, then $D = 8$.

Why not just take $H(x) = B(x) + C(x)$? The reason for the alphas and betas is to make the resulting $H$ be always different and unpredictable for the prover, so they can't precompute stuff beforehand. The $x^{D - deg(...)}$ term is simply an optimization to make sure that $H$ always ends up having a power of two as its degree; it adds nothing to the security or soundness of the system, it just allows code implementations to run faster by using the Fast Fourier Transform.

With what we discussed above, showing that the constraints are satisfied is equivalent to saying that `H` is a polynomial and not a rational function (we are simplifying things a bit here, but it works for our purposes).

## Commiting to $H$

To show $H$ is a polynomial we are going to use the `FRI` protocol, which we treat as a black box. For all we care, a `FRI` proof will verify if what we committed to is indeed a polynomial. Thus, the prover will provide a `FRI` commitment to `H`, and if it passes, the verifier will be convinced that the constraints are satisfied.

There is one catch here though: how does the verifier know that `FRI` was applied to `H` and not any other polynomial? For this we need to add an additional step to the protocol. 


## Consistency check
After commiting to `H`, the prover needs to show that `H` was constructed correctly according to the formula above. To do this, it will ask the prover to provide an evaluation of `H` on some random point `z` and evaluations of the trace at the points $t(z), t(zg)$ and $t(zg^2)$.

Because the boundary and transition constraints are a public part of the protocol, the verifiers knows them, and thus the only thing it needs to compute the evaluation $H(z)$ by themselves are the three trace evaluations mentioned above. Because it asked the prover for them, it can check both sides of the equation:

$$
H(z) = B(z) (\alpha_1 z^{D - deg(B)} + \beta_1) + C(z) (\alpha_2 z^{D - deg(C)} + \beta_2)
$$

and be convinced that $H$ was constructed correctly.

We are still not done, however, as the prover could have now cheated on the values of the trace evaluations.

## Deep Composition Polynomial

TODO

## FRI and low degree extensions

TODO: Move this? it's a digression into LDEs, roots of unity, etc. It probably belongs in an explanation about the constraint evaluation blowup factor on the implementation side.

In our case, we will take a primitive $16$-th root of unity $\omega$. This means, as before:

- $\omega$ is an $16$-th root of unity, i.e., $\omega^{16} = 1$.
- Every $16$-th root of unity is of the form $\omega^i$ for some $0 \leq i \leq 15$.

Additionally, we also take it so that $\omega$ satisfies $\omega^2 = g$.

The evaluation of $t$ on the set $\{\omega^i : 0 \leq i \leq 15\}$ is called a *low degree extension* (`LDE`) of $t$. Notice this is not a new polynomial, they're evaluations of $t$ on some set of points. Also note that because $\omega^2 = g$, the `LDE` contains all the evaluations of $t$ on the set of powers of $g$. In fact,

$$
    \{t(\omega^{2i}) : 0 \leq i \leq 15\} = \{t(g^i) : 0 \leq i \leq 7\}
$$

This will be important later on.

For our `LDE`, we chose $16$-th roots of unity, but we could have chosen any other power of two greater than $8$. In general, this choice is called the `blowup factor`, so that if the trace has $2^n$ elements, a blowup factor of $b$ means our LDE evaluates over the $2^{n} * b$ roots of unity ($b$ needs to be a power of two). The blowup factor is a parameter of the protocol related to its security.
