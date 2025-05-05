# lambdaworks Fast Fourier Transform

This folder contains the [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT) over finite fields (also known as number theoretic transform, NTT). If you are unfamiliar with how lambdaworks handles fields, see [examples](../../../../examples/README.md). Currently, the following algorithms are supported:
- Cooley-Tukey Radix-2
- Cooley-Tukey Radix-4

We are also planning on adding a mixed radix algorithm. To use the FFT, the length of the vector, $n$, should be a power of $2$ (or $4$), that is, $2^m = n$. The FFT should be used with fields implementing the `IsFFTFriendly` trait. The FFT works by recursively breaking a length $n$ FFT into $2$ $n/2$ FFTs, until we reach a sufficiently small size that can be solved. The core operation of the FFT is the butterfly. To combine the elements, we need to sample the twiddle factors, which we obtain from the roots of unity. The FFT can accept the input in natural order and returns the output in reverse order (nr) or the input is in reverse order and the output is in natural order (rn).

Since the main applications of the FFT are related to polynomial evaluation and interpolation, we provide functions describing these operations, which call the FFT under the hood:
- `evaluate_fft`
- `evaluate_offset_fft`
- `interpolate_fft`
- `interpolate_offset_fft`

These functions can be used with [univariate polynomials](./README.md). To use the functions,
```rust
let p_1 = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5) FE::new(6)]);
let evaluations = Polynomial::evaluate_offset_fft(p_1, 4, 4, FE::new(3))?;
```
Interpolate takes a vector of length $2^m$, which we take them to be the evaluations of a polynomial $p$ over values of $x$ of the form $\{ offset.g^0, offset.g, offset.g^2 \dots offset.g^{n - 1} \}$, where $g$ is a generator of the $n$-th roots of unity. For example,
```rust
let evaluations = [FE::new(1), FE::new(2), FE::new(3) FE::new(4)];
let poly = Polynomial::interpolate_fft(&evaluations).unwrap();
```

These building blocks are used, for example, in the computation of the trace polynomials in the STARK protocol. The following function computes the polynomials whose evaluations coincide with the trace columns:
```rust
pub fn compute_trace_polys<S>(&self) -> Vec<Polynomial<FieldElement<F>>>
    where
        S: IsFFTField + IsSubFieldOf<F>,
        FieldElement<F>: Send + Sync,
    {
        let columns = self.columns();
        #[cfg(feature = "parallel")]
        let iter = columns.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = columns.iter();

        iter.map(|col| Polynomial::interpolate_fft::<S>(col))
            .collect::<Result<Vec<Polynomial<FieldElement<F>>>, FFTError>>()
            .unwrap()
    }
```

## Background on the Fast-Fourier transform for finite fields

Given a finite field $\mathbb{F}_p$ we say that $\omega$ is an r-th primitive root of unity if $\omega^r \equiv 1 \pmod{p}$ and $\omega^k \not \equiv 1 \pmod{p}$ for $k \not \equiv 0 \pmod{r}$ (this means that $\omega$'s powers are different from 1 for all $k = 1,2,... r - 1$ and $\omega$ spans a group with $r$ elements). If $n$ divides $r$, $\omega^{r/n}$ is an n-th primitive root.

Throughout, we will work with $2^m$-th roots of unity, which have a nice structure. If $\omega$ is such a root, we see that the set formed by its powers is given by $1, \omega, \omega^2, ... , - 1, - \omega , - \omega^2 , ...$, since $\omega^{2^{m - 1}} \equiv 1 \pmod{p}$. This way, we see that we always have $b$ and $-b$ in the group, which can save a lot of time if we use these points to evaluate a polynomial.

Take a polynomial with coefficients in $\mathbb{F}_p$, $p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n$. We can write the polynomial in the following form:

$$p(x) = (a_0 + a_2 x^2 + a_4 x^4 + ... + a_{n - 1} x^{n - 1}) + x  (a_1 + a_3 x^2 + a_5 x^4 + ... + a_{n} x^{n - 1})$$

More compactly,

$$p(x) = p_e (x^2 ) + x p_o (x^2 )$$

The original polynomial has $n + 1$ coefficients, while each of the $p_i (x^2 )$ has $(n + 1)/2$ (we will suppose that the polynomial has $n + 1 = 2^m$, if not we can pad the polynomial with zero coefficients in the highest powers). Take $b$ and $-b$: both have the same square, $b^2 = ( - b )^2 = c$, so

$$p(b) = p_e (c) + b p_o (c)$$

$$p(b) = p_e (c) - b p_o (c)$$

This means that, for the price of evaluating $p_e (c)$ and $p_o (c)$ (each needing at most $\mathcal{O}((n + 1)/2)$ operations) we get two evaluations of $p(x)$ for the cost of one additional multiplication and addition/subtraction. Because $b$ is a $2^m$-th root of unity, $b^2$ is an $2^{m - 1}$-th root of unity. 

Since $p_e (x^2)$ and $p_o(x^2)$ are also polynomials, we can use the same trick once again. Moreover, the nice group structure we have says we need to only evaluate at half the $2^{m - 2}$ roots of unity. This provides a powerful recursive structure, where we reduce the problem to computing the evaluations of two polynomials over a smaller subset. We can do this until we get to the square roots of unity (1 and -1), where it is easy to compute and then we just recombine all the evaluations! 

We will provide an example for $n = 7$, which will be sufficient to grasp the general idea. We will denote $\omega$ a primitive 8-th root of unity and $i = \omega^2$ a primitive 4-th root of unity. We have the polynomial:

$$p(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4 + a_5 x^5 + a^6 x^6 + a_7 x^7$$

The 8th-roots of unity are $1, \omega, i, \omega^3, -1, -\omega, -i, -\omega^3$. The evaluations would be:

$$p(1) = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a^6 + a_7$$

$$p(\omega) = a_0 + a_1 \omega + a_2 i + a_3 \omega^3 + - a_4 - a_5 \omega - a^6 i - a_7 \omega^3$$

$$p(i) = a_0 + a_1 i - a_2 - a_3 i + a_4 + a_5 i - a^6 - a_7 i$$

$$p(\omega^3) = a_0 + a_1 \omega^3 - a_2 i + a_3 \omega  - a_4 - a_5 \omega^3 + a^6 i - a_7 \omega$$

$$p(- 1) = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a^6 - a_7$$

$$p(- \omega) = a_0 - a_1 \omega + a_2 i - a_3 \omega^3 + - a_4 + a_5 \omega - a^6 i + a_7 \omega^3$$

$$p(-i) = a_0 - a_1 i - a_2 + a_3 i + a_4 - a_5 i - a^6 + a_7 i$$

$$p(\omega^3) = a_0 - a_1 \omega^3 - a_2 i - a_3 \omega + - a_4 + a_5 \omega^3 + a^6 i + a_7 \omega$$

Let's work with the idea behind the FFT and see that we arrive at the same results (when doing the computations, we used $\omega^4 \equiv - 1 \pmod{p}$) and $i^2 \equiv -1 \pmod{p}$. The first step is breaking $p(x)$ into its odd and even parts:

$$p_e (x^2) = a_0 + a_2 x^2 + a_4 x^4 + a_6 x^6$$

$$p_o (x^2) = a_1 + a_3 x^2 + a_5 x^4 + a_7 x^6$$

We can break them again,

$$p_{ee} (x^4) = a_0 + a_4 x^4$$

$$p_{eo} (x^4) = a_2 + a_6 x^4$$

$$p_{oe} (x^4) = a_1 + a_5 x^4$$

$$p_{oo} (x^4) = a_3 + a_7 x^4$$

Since ${\omega^r}^4 \equiv {\omega^4}^r \equiv (-1)^r \pmod{p}$, so $x^4$ can only take two values, -1 and 1. Each of the polynomials can be easily evaluated:

$$p_{ee} (1) = a_0 + a_4$$

$$p_{ee} (- 1) = a_0 - a_4$$

$$p_{eo} (1) = a_2 + a_6$$

$$p_{eo} (- 1) = a_2 - a_6$$

$$p_{oe} (1) = a_1 + a_5$$

$$p_{oe} (- 1) = a_1 - a_5$$

$$p_{oo} (1) = a_3 + a_7$$

$$p_{oo} (- 1) = a_3 - a_7$$

If you take a look at each pair of evaluations, they have the form:

$$p(w) = a + w b$$

$$p(-w) = a - wb$$

These are the butterflies, where $w = 1$ in this case, which is one of the elements in the square roots of unity. We now have to go one level above, using $p_{ee} (1)$ and $p_{eo} (1)$ to obtain the evaluations of $p_e (x)$ at $x = \pm \sqrt{1}$. Remember that in the change of variables $y = x^2$ we sent both $x_0$ and $- x_0$ to the same point. Similarly, we can use $p_{ee} (- 1)$ and $p_{eo} (-1)$ to reconstruct the evaluations of $p_e (i)$ and $p_e (-i)$ since $i^2 = ( - i)^2 = - 1$. Below we are giving those 8 evaluations, four for $p_e (x)$ and four for $p_o (x)$:

$$p_{e} (1) = (a_0 + a_4) + (a_2 + a_6)$$

$$p_{e} (i) = (a_0 - a_4) + i (a_2 - a_6)$$

$$p_{e} (- 1) = (a_0 + a_4) - (a_2 + a_6)$$

$$p_{e} (- i) = (a_0 - a_4) - i (a_2 - a_6)$$

$$p_{o} (1) = (a_1 + a_5) + (a_3 + a_7)$$

$$p_{o} (i ) = (a_1 - a_5) + i (a_3 - a_7)$$

$$p_{o} (- 1) = (a_1 + a_5) - (a_3 + a_7)$$

$$p_{o} (- i ) = (a_1 - a_5) - i (a_3 - a_7)$$

We have written things in this way so that you can clearly see the butterflies on this level. Each term from the level below is in brackets. If you look at the first and third elements, they are of the form:

$$p_{e} (1) = a_e + b_e$$

$$p_{e} (- 1) = a_e - b_e$$

Similarly, for the second and fourth,

$$p_{e} (i) = a_o + i b_o$$

$$p_{e} (- i) = a_o - i b_o$$

The same structure is followed by $p_o (x)$. Here, there are two different elements when combining the elements: 1 and $i$, which correspond to the $4$-th roots of unity.

Finally, we can go one step above and combine $p_e (t^2)$ and $p_o (t^2)$ to obtain $p (t)$. We take the first and fifth elements, $p_e (1)$ and $p_o (1)$ and generate $p (1)$ and $p(- 1)$:

$$p (1) = ((a_0 + a_4) + (a_2 + a_6)) + ((a_1 + a_5) + (a_3 + a_7))$$

$$p (\omega ) = ((a_0 - a_4) + i (a_2 - a_6)) + \omega ((a_1 - a_5) + i (a_3 - a_7))$$

$$p (i) = ((a_0 + a_4) - (a_2 + a_6)) + i ((a_1 + a_5) - (a_3 + a_7))$$

$$p (\omega^3) = ((a_0 - a_4) - i (a_2 - a_6)) + \omega^3 ((a_1 - a_5) - i (a_3 - a_7))$$

$$p (- 1) = ((a_0 + a_4) + (a_2 + a_6)) - ((a_1 + a_5) + (a_3 + a_7))$$

$$p (- \omega ) = ((a_0 - a_4) + i (a_2 - a_6)) - \omega ((a_1 - a_5) + i (a_3 - a_7))$$

$$p (- i) = ((a_0 + a_4) - (a_2 + a_6)) - i ((a_1 + a_5) - (a_3 + a_7))$$

$$p (- \omega^3) = ((a_0 - a_4) - i (a_2 - a_6)) - \omega^3 ((a_1 - a_5) - i (a_3 - a_7))$$

The terms are organized differently from the direct evaluation, but this is just to show clearly how all the butterflies have been combined at each stage. We can organize all the $a_k$ according to their index:

$$p(1) = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7$$

$$p(\omega) = a_0 + a_1 \omega + a_2 i + a_3 \omega^3 - a_4 - a_5 \omega - a_6 i  - a_7 \omega^3$$

$$p(i) = a_0 + a_1 i - a_2 - a_3 i + a_4 + a_5 i - a_6 - a_7 i$$

$$p (\omega^3) = a_0 + a_1 \omega^3 - a_2 i + a_3 \omega - a_4 - a_5 \omega^3 + a_6 i - a_7 \omega$$

$$p(- 1) = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7$$

$$p(- \omega) = a_0 - a_1 \omega + a_2 i - a_3 \omega^3 - a_4 + a_5 \omega - a_6 i  + a_7 \omega^3$$

$$p(- i) = a_0 - a_1 i - a_2 + a_3 i + a_4 - a_5 i - a_6 + a_7 i$$

$$p (- \omega^3) = a_0 - a_1 \omega^3 - a_2 i - a_3 \omega - a_4 + a_5 \omega^3 + a_6 i + a_7 \omega$$

where we have used that $\omega i = \omega^3$ and $\omega^3 i = -\omega$. For larger FFTs, we just have to continue breaking down and combining the elements using the butterflies with their corresponding twiddle factors.

