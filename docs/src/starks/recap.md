# STARKs Recap

# Verifying Computation through Polynomials

In general, we express computation in our proving system by providing an *execution trace* satisfying certain *constraints*. The execution trace is a table containing the state of the system at every step of computation. This computation needs to follow certain rules to be valid; these rules are our *constraints*.

The constraints for our computation are expressed using an `Algebraic Intermediate Representation` or `AIR`. This representation uses polynomials to encode constraints, which is why sometimes they are called `polynomial constraints`.

To make all this less abstract, let's go through two examples.

## Fibonacci numbers

Throughout this section and the following we will use this example extensively to have a concrete example. Even though it's a bit contrived (no one cares about computing fibonacci numbers), it's simple enough to be useful. STARKs and proving systems in general are very abstract things; having an example in mind is essential to not get lost.

Let's say our computation consists of calculating the `k`-th number in the fibonacci sequence. This is just the sequence of numbers \\(a_n\\) satisfying

\\[
    a_0 = 1 
\\]
\\[
    a_1 = 1
\\]
\\[
    a_{n+2} = a_{n + 1} + a_n
\\]

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

# Fibonacci step by step walkthrough

Below we go through a step by step explanation of a STARK prover. We will assume the trace of the fibonacci sequence mentioned above; it consists of only one column of length \\(2^n\\). In this case, we'll take `n=3`. The trace looks like this

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

- `g` is an `8`-th root of unity, i.e., \\(g^8 = 1\\).
- Every `8`-th root of unity is of the form \\(g^i\\) for some \\(0 \leq i \leq 7\\).

With `g` in hand, we take the trace polynomial `t` to be the one satisfying

$$
t(g^i) = a_i
$$

From here onwards, we will talk about the validity of the trace in terms of properties that this polynomial must satisfy. We will also implicitly identify a certain power of \\(g\\) with its corresponding trace element, so for example we sometimes think of \\(g^5\\) as \\(a_5\\), the fifth row in the trace, even though technically it's \\(t\\) *evaluated in* \\(g^5\\) that equals \\(a_5\\).

We talked about two different types of constraints the trace must satisfy to be valid. They were:

- The first two rows are `1`.
- The value on any other row is the sum of the two preceding ones.

In terms of `t`, this translates to

- \\(t(g^0) = 1\\) and \\(t(g) = 1\\).
- \\(t(x g^2) - t(xg) - t(x) = 0\\) for all \\(x \in \{g^0, g^1, g^2, g^3, g^4, g^5\}\\). This is because multiplying by `g` is the same as advancing a row in the trace.

## Composition Polynomial

To convince the verifier that the trace polynomial satisfies the relationships above, the prover will construct another polynomial that shows that both the boundary and transition constraints are satisfied and commit to it. We call this polynomial the `composition polynomial`, and usually denote it with \\(H\\). Constructing it involves a lot of different things, so we'll go step by step introducing all the moving parts required.

### Boundary polynomial

To show that the boundary constraints are satisfied, we construct the `boundary polynomial`. Recall that our boundary constraints are \\(t(g^0) = t(g) = 1\\). Let's call \\(P\\) the polynomial that interpolates these constraints, that is, \\(P\\) satisfies:

$$
P(1) = 1 \\
P(g) = 1
$$

The boundary polynomial \\(B\\) is defined as follows:

$$
B(x) = \dfrac{t(x) - P(x)}{(x - 1) (x - g)}
$$

The denominator here is called the `boundary zerofier`, and it's the polynomial whose roots are the elements of the trace where the boundary constraints must hold.

How does \\(B\\) encode the boundary constraints? The idea is that, if the trace satisfies said constraints, then

$$
t(1) - P(1) = 1 - 1 = 0 \\
$$
$$
t(g) - P(g) = 1 - 1 = 0
$$

so \\(t(x) - P(x)\\) has \\(1\\) and \\(g\\) as roots. Showing these values are roots is the same as showing that \\(B(x)\\) is a polynomial instead of a rational function, and that's why we construct \\(B\\) this way.

### Transition constraint polynomial
To convince the verifier that the transition constraints are satisfied, we construct the `transition constraint polynomial` and call it \\(C(x)\\). It's defined as follows:

$$
C(x) = \dfrac{t(xg^2) - t(xg) - t(x)}{\prod_{i = 0}^{5} (x - g^i)}
$$

How does \\(C\\) encode the transition constraints? We mentioned above that these are satisfied if the polynomial in the numerator vanishes in the elements \\(\{g^0, g^1, g^2, g^3, g^4, g^5\}\\). As with \\(B\\), this is the same as showing that \\(C(x)\\) is a polynomial instead of a rational function.

### Constructing \\(H\\)
With the boundary and transition constraint polynomials in hand, we build the `composition polynomial` \\(H\\) as follows: The verifier will sample four numbers \\(\beta_1, \beta_2\\) and \\(H\\) will be

$$
H(x) = \beta_1 B(x) + \beta_2 C(x)
$$


Why not just take \\(H(x) = B(x) + C(x)\\)? The reason for the betas is to make the resulting \\(H\\) be always different and unpredictable for the prover, so they can't precompute stuff beforehand.

With what we discussed above, showing that the constraints are satisfied is equivalent to saying that `H` is a polynomial and not a rational function (we are simplifying things a bit here, but it works for our purposes).

## Commiting to \\(H\\)

To show \\(H\\) is a polynomial we are going to use the `FRI` protocol, which we treat as a black box. For all we care, a `FRI` proof will verify if what we committed to is indeed a polynomial. Thus, the prover will provide a `FRI` commitment to `H`, and if it passes, the verifier will be convinced that the constraints are satisfied.

There is one catch here though: how does the verifier know that `FRI` was applied to `H` and not any other polynomial? For this we need to add an additional step to the protocol. 


## Consistency check
After commiting to `H`, the prover needs to show that `H` was constructed correctly according to the formula above. To do this, it will ask the prover to provide an evaluation of `H` on some random point `z` and evaluations of the trace at the points \\(t(z), t(zg)\\) and \\(t(zg^2)\\).

Because the boundary and transition constraints are a public part of the protocol, the verifier knows them, and thus the only thing it needs to compute the evaluation \\((z)\\) by itself are the three trace evaluations mentioned above. Because it asked the prover for them, it can check both sides of the equation:

$$
H(z) = \beta_1 B(z) + \beta_2 C(z)
$$

and be convinced that \\(H\\) was constructed correctly.

We are still not done, however, as the prover could have now cheated on the values of the trace or composition polynomial evaluations.

## Deep Composition Polynomial

There are two things left the prover needs to show to complete the proof:

- That \\(H\\) effectively is a polynomial, i.e., that the constraints are satisfied.
- That the evaluations the prover provided on the consistency check were indeed evaluations of the trace polynomial and composition polynomial on the out of domain point `z`.

Earlier we said we would use the `FRI` protocol to commit to `H` and show the first item in the list. However, we can slightly modify the polynomial we do `FRI` on to show both the first and second items at the same time. This new modified polynomial is called the `DEEP` composition polynomial. We define it as follows:

$$
Deep(x) = \gamma_1 \dfrac{H(x) - H(z)}{x - z} + \gamma_2 \dfrac{t(x) - t(z)}{x - z} + \gamma_3 \dfrac{t(x) - t(zg)}{x - zg} + \gamma_4 \dfrac{t(x) - t(zg^2)}{x - zg^2}
$$

where the numbers \\(\gamma_i\\) are randomly sampled by the verifier.

The high level idea is the following: If we apply `FRI` to this polynomial and it verifies, we are simultaneously showing that

- \\(H\\) is a polynomial and the prover indeed provided `H(z)` as one of the out of domain evaluations. This is the first summand in `Deep(x)`.
- The trace evaluations provided by the prover were the correct ones, i.e., they were \\(t(z)\\), \\(t(zg)\\), and \\(t(zg^2)\\). These are the remaining summands of the `Deep(x)`.

### Consistency check
The prover needs to show that `Deep` was constructed correctly according to the formula above. To do this, the verifier will ask the prover to provide:

- An evaluation of `H` on `z` and `x_0`
- Evaluations of the trace at the points \\(t(z)\\), \\(t(zg)\\), \\(t(zg^2)\\) and \\(t(x_0)\\)

Where `z` is the same random, out of domain point used in the consistency check of the composition polynomial, and `x_0` is a random point that belongs to the trace domain.

With the values provided by the prover, the verifier can check both sides of the equation:

$$
Deep(x_0) = \gamma_1 \dfrac{H(x_0) - H(z)}{x_0 - z} + \gamma_2 \dfrac{t(x_0) - t(z)}{x_0 - z} + \gamma_3 \dfrac{t(x_0) - t(zg)}{x_0 - zg} + \gamma_4 \dfrac{t(x_0) - t(zg^2)}{x_0 - zg^2}
$$

The prover also needs to show that the trace evaluation \\(t(x_0)\\) belongs to the trace. To achieve this, it needs to commit the merkle roots of `t` and the merkle proof of \\(t(x_0)\\).

## Summary

We summarize below the steps required in a STARK proof for both prover and verifier.

### Prover side

- Compute the trace polynomial `t` by interpolating the trace column over a set of \\(2^n\\)-th roots of unity \\(\{g^i : 0 \leq i < 2^n\}\\).
- Compute the boundary polynomial `B`.
- Compute the transition constraint polynomial `C`.
- Construct the composition polynomial `H` from `B` and `C`.
- Sample an out of domain point `z` and provide the evaluations \\(H(z)\\), \\(t(z)\\), \\(t(zg)\\), and \\(t(zg^2)\\) to the verifier.
- Sample a domain point `x_0` and provide the evaluations \\(H(x_0)\\) and \\(t(x_0)\\) to the verifier.
- Construct the deep composition polynomial `Deep(x)` from `H`, `t`, and the evaluations from the item above.
- Do `FRI` on `Deep(x)` and provide the resulting FRI commitment to the verifier.
- Provide the merkle root of `t` and the merkle proof of \\(t(x_0)\\).

### Verifier side

- Take the evaluations \\(H(z)\\), \\(H(x_0)\\), \\(t(z)\\), \\(t(zg)\\), \\(t(zg^2)\\) and \\(t(x_0)\\) the prover provided.
- Reconstruct the evaluations \\(B(z)\\) and \\(C(z)\\) from the trace evaluations we were given. Check that the claimed evaluation \\(H(z)\\) the prover gave us actually satisfies
    $$
    H(z) =  \beta_1 B(z) + \beta_2 C(z)
    $$
- Check that the claimed evaluation \\(Deep(x_0)\\) the prover gave us actually satisfies
    $$
    Deep(x_0) = \gamma_1 \dfrac{H(x_0) - H(z)}{x_0 - z} + \gamma_2 \dfrac{t(x_0) - t(z)}{x_0 - z} + \gamma_3 \dfrac{t(x_0) - t(zg)}{x_0 - zg} + \gamma_4 \dfrac{t(x_0) - t(zg^2)}{x_0 - zg^2}
    $$
- Using the merkle root and the merkle proof the prover provided, check that \\(t(x_0)\\) belongs to the trace.
- Take the provided `FRI` commitment and check that it verifies.

# Simplifications and Omissions

The walkthrough above was for the fibonacci example which, because of its simplicity, allowed us to sweep under the rug a few more complexities that we'll have to tackle on the implementation side. They are:

### Multiple trace columns

Our trace contained only one column, but in the general setting there can be multiple (the Cairo `AIR` has around 30). This means there isn't just *one* trace polynomial, but several; one for each column. This also means there are multiple boundary constraint polynomials.

The general idea, however, remains the same. The deep composition polynomial `H` is now the sum of several terms containing the boundary constraint polynomials \\(B_1(x), \dots, B_k(x)\\) (one per column), and each \\(B_i\\) is in turn constructed from the \\(i\\)-th trace polynomial \\(t_i(x)\\).

### Multiple transition constraints

Much in the same way, our fibonacci `AIR` had only one transition constraint, but there could be several. We will therefore have multiple transition constraint polynomials \\(C_1(x), \dots, C_n(x)\\), each of which encodes a different relationship between rows that must be satisfied. Also, because there are multiple trace columns, a transition constraint can mix different trace polynomials. One such constraint could be

$$
C_1(x) = t_1(gx) - t_2(x)
$$

which means "The first column on the next row has to be equal to the second column in the current row".

Again, even though this seems way more complex, the ideas remain the same. The composition polynomial `H` will now include a term for every \\(C_i(x)\\), and for each one the prover will have to provide out of domain evaluations of the trace polynomials at the appropriate values. In our example above, to perform the consistency check on \\(C_1(x)\\) the prover will have to provide the evaluations \\(t_1(zg)\\) and \\(t_2(z)\\).

### Composition polynomial decomposition

In the actual implementation, we won't commit to \\(H\\), but rather to a decomposition of \\(H\\) into an even term \\(H_1(x)\\) and an odd term \\(H_2(x)\\), which satisfy

$$
H(x) = H_1(x^2) + x H_2(x^2)
$$

This way, we don't commit to \\(H\\) but to \\(H_1\\) and \\(H_2\\). This is just an optimization at the code level; once again, the ideas remain exactly the same.

### FRI, low degree extensions and roots of unity

We treated `FRI` as a black box entirely. However, there is one thing we do need to understand about it: low degree extensions.

When applying `FRI` to a polynomial of degree \\(n\\), we need to provide evaluations of it over a domain with *more* than \\(n\\) points. In our case, the `DEEP` composition polynomial's degree is around the same as the trace's, which is, at most, \\(2^n - 1\\) (because it interpolates the trace containing \\(2^n\\) points).

The domain we are going to choose to evaluate our `DEEP` polynomial on will be a set of *higher* roots of unity. In our fibonacci example, we will take a primitive \\(16\\)-th root of unity \\(\omega\\). As a reminder, this means:

- \\(\omega\\) is an \\(16\\)-th root of unity, i.e., \\(\omega^{16} = 1\\).
- Every \\(16\\)-th root of unity is of the form \\(\omega^i\\) for some \\(0 \leq i \leq 15\\).

Additionally, we also take it so that \\(\omega\\) satisfies \\(\omega^2 = g\\) (\\(g\\) being the \\(8\\)-th primitive root of unity we used to construct `t`).

The evaluation of \\(t\\) on the set \\(\{\omega^i : 0 \leq i \leq 15\}\\) is called a *low degree extension* (`LDE`) of \\(t\\). Notice this is not a new polynomial, they're evaluations of \\(t\\) on some set of points. Also note that, because \\(\omega^2 = g\\), the `LDE` contains all the evaluations of \\(t\\) on the set of powers of \\(g\\). In fact,

$$
    \{t(\omega^{2i}) : 0 \leq i \leq 15\} = \{t(g^i) : 0 \leq i \leq 7\}
$$

This will be extremely important when we get to implementation.

For our `LDE`, we chose \\(16\\)-th roots of unity, but we could have chosen any other power of two greater than \\(8\\). In general, this choice is called the `blowup factor`, so that if the trace has \\(2^n\\) elements, a blowup factor of \\(b\\) means our LDE evaluates over the \\(2^{n} * b\\) roots of unity (\\(b\\) needs to be a power of two). The blowup factor is a parameter of the protocol related to its security.
