# STARKs Recap

## Verifying Computation through Polynomials

In general, we express computation in our proving system by providing an *execution trace* satisfying certain *constraints*. The execution trace is a table containing the state of the system at every step of computation. This computation needs to follow certain rules to be valid; these rules are our *constraints*.

The constraints for our computation are expressed using an `Algebraic Intermediate Representation` or `AIR`. This representation uses polynomials to encode constraints, which is why sometimes they are called `polynomial constraints`.

To make all this less abstract, let's go through two examples.

### Fibonacci numbers

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

### Cairo

The example above is extremely useful to have a mental model, but it's not really useful for anything. The problem is it just works for the very narrow example of computing fibonacci numbers. If we wanted to prove execution of something else, we would have to write an `AIR` for it.

What we're actually aiming for is an `AIR` for an entire general purpose `Virtual Machine`. This way, we can provide proofs of execution for *any* computation using just one `AIR`. This is what [cairo](https://www.cairo-lang.org/docs/) as a programming language does. Cairo code compiles to the bytecode of a virtual machine with an already defined `AIR`. The general flow when using cairo is the following:

- User writes a cairo program.
- The program is compiled into Cairo's VM bytecode.
- The VM executes said code and provides an execution trace for it.
- The trace is passed on to a STARK prover, which creates a proof of correct execution according to Cairo's AIR.
- The proof is passed to a verifier, who checks that the proof is valid.

Ultimately, our goal is to give the tools to write a STARK prover for the cairo VM and do so. However, this is not a good example to start out as it's incredibly complex. The execution trace of a cairo program has around 30 columns, some for general purpose registers, some for other reasons. Cairo's AIR contains a lot of different transition constraints, encoding all the different possible instructions (arithmetic operations, jumps, etc).

Use the fibonacci example as your go-to for understanding all the moving parts; keep the Cairo example in mind as the thing we are actually building towards.

## Step by step walkthrough

Below we go through a step by step explanation of a STARK prover. We will assume the trace of the fibonacci sequence mentioned above; it consists of of only one column of length $2^n$. In this case, we'll take `n=3`. The trace looks like this

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

### Trace polynomial

The first step is to interpolate these values to generate the `trace polynomial`. This will be a polynomial encoding all the information about the trace. The way we do it is the following: in the finite field we are working in, we take an `8`-th primitive root of unity, let's call it `g`. It being a primitive root means two things:

- $g^8 = 1$ (`g` is an `8`-th root of unity).
- Every `8`-th root of unity is of the form $g^i$ for some $0 \leq i \leq 7$.

With `g` in hand, we take the trace polynomial `t` to be the one satisfying

$$
t(g^i) = a_i
$$

From here onwards, we will talk about the validity of the trace in terms of properties that this polynomial must satisfy.

We talked about two different types of constraints the trace must satisfy to be valid. They were:

- The first two rows are `1`.
- The value on any other row is the sum of the two preceding ones.

In terms of `t`, this translates to

- $t(g^0) = 1$ and $t(g) = 1$.
- $t(x g^2) - t(xg) - t(x) = 0$ for all $x \in \{g^0, g^1, g^2, g^3, g^4, g^5\}$. This is because multiplying by `g` is the same as advancing a row in the trace.

### 
