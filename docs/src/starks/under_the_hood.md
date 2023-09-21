# How this works under the hood

In this section we go over how a few things in the `prove` and `verify` functions are implemented. If you just need to *use* the prover, then you probably don't need to read this. If you're going through the code to try to understand it, read on.

We will once again use the fibonacci example as an ilustration. Recall from the `recap` that the main steps for the prover and verifier are the following:

### Prover side

- Compute the trace polynomial `t` by interpolating the trace column over a set of $2^n$-th roots of unity $\{g^i : 0 \leq i < 2^n\}$.
- Compute the boundary polynomial `B`.
- Compute the transition constraint polynomial `C`.
- Construct the composition polynomial `H` from `B` and `C`.
- Sample an out of domain point `z` and provide the evaluation $H(z)$ and all the necessary trace evaluations to reconstruct it. In the fibonacci case, these are $t(z)$, $t(zg)$, and $t(zg^2)$.
- Sample a domain point `x_0` and provide the evaluation $H(x_0)$ and $t(x_0)$.
- Construct the deep composition polynomial `Deep(x)` from `H`, `t`, and the evaluations from the item above.
- Do `FRI` on `Deep(x)` and provide the resulting FRI commitment to the verifier.
- Provide the merkle root of `t` and the merkle proof of $t(x_0)$.

### Verifier side

- Take the evaluation $H(z)$ along with the trace evaluations the prover provided.
- Reconstruct the evaluations $B(z)$ and $C(z)$ from the trace evaluations. Check that the claimed evaluation $H(z)$ the prover gave us actually satisfies
    $$
    H(z) = \beta_1 B(z) + \beta_2 C(z)
    $$
- Take the evaluations $H(x_0)$ and $t(x_0)$.
- Check that the claimed evaluation $Deep(x_0)$ the prover gave us actually satisfies
    $$
    Deep(x_0) = \gamma_1 \dfrac{H(x_0) - H(z)}{x_0 - z} + \gamma_2 \dfrac{t(x_0) - t(z)}{x_0 - z} + \gamma_3 \dfrac{t(x_0) - t(zg)}{x_0 - zg} + \gamma_4 \dfrac{t(x_0) - t(zg^2)}{x_0 - zg^2}
    $$
- Take the provided `FRI` commitment and check that it verifies.
- Using the merkle root and the merkle proof the prover provided, check that $t(x_0)$ belongs to the trace.

Following along the code in the `prove` and `verify` functions, most of it maps pretty well to the steps above. The main things that are not immediately clear are:

- How we take the constraints defined in the `AIR` through the `compute_transition` method and map them to transition constraint polynomials.
- How we then construct `H` from them and the boundary constraint polynomials.
- What the composition polynomial even/odd decomposition is.
- What an `ood` frame is.
- What the transcript is.

# Reconstructing the transition constraint polynomials

This is possibly the most complex part of the code, so what follows is a long explanation for it.

In our fibonacci example, after obtaining the trace polynomial `t` by interpolating, the transition constraint polynomial is

$$
C(x) = \dfrac{t(xg^2) - t(xg) - t(x)}{\prod_{i = 0}^{5} (x - g^i)}
$$

On our `prove` code, if someone passes us a fibonacci `AIR` like the one we showed above used in one of our tests, we somehow need to construct $C(x)$. However, what we are given is not a polynomial, but rather this method

```rust
fn compute_transition(
        &self,
        frame: &air::frame::Frame<Self::Field>,
    ) -> Vec<FieldElement<Self::Field>> {
    let first_row = frame.get_row(0);
    let second_row = frame.get_row(1);
    let third_row = frame.get_row(2);

    vec![third_row[0] - second_row[0] - first_row[0]]
}
```

So how do we get to $C(x)$ from this? The answer is interpolation. What the method above is doing is the following: if you pass it a frame that looks like this

$$
\begin{bmatrix} t(x_0) \\ t(x_0g) \\ t(x_0g^2) \end{bmatrix}
$$

for any given point $x_0$, it will return the value

$$
t(x_0g^2) - t(x_0g) - t(x_0)
$$

which is the numerator in $C(x_0)$. Using the `transition_exemptions` field we defined in our `AIR`, we can also compute evaluations in the denominator, i.e. the zerofier evaluations. This is done under the hood by the `transition_divisors()` method.

The above means that even though we don't explicitly have the polynomial $C(x)$, we can evaluate it on points given an appropriate frame. If we can evaluate it on enough points, we can then interpolate them to recover $C(x)$. This is exactly how we construct both transition constraint polynomials and subsequently the composition polynomial `H`.

The job of evaluating `H` on enough points so we can then interpolate it is done by the `ConstraintEvaluator` struct. You'll notice `prove` does the following

```rust
let constraint_evaluations = evaluator.evaluate(
    &lde_trace,
    &lde_roots_of_unity_coset,
    &alpha_and_beta_transition_coefficients,
    &alpha_and_beta_boundary_coefficients,
);
```

This function call will return the evaluations of the boundary terms 

$$
\beta_i^B B_i(x)
$$

and constraint terms

$$
\beta_i^T C_i(x)
$$

for every $i$. The `constraint_evaluations` value returned is a `ConstraintEvaluationTable` struct, which is nothing more than a big list of evaluations of each polynomial required to construct `H`.

With this in hand, we just call

```rust
let composition_poly =  
    constraint_evaluations.compute_composition_poly(&   lde_roots_of_unity_coset);
```

which simply interpolates the sum of all evaluations to obtain `H`.

Let's go into more detail on how the `evaluate` method reconstructs $C(x)$ in our fibonacci example. It receives the `lde_trace` as an argument, which is this:

$$
\begin{bmatrix} t(\omega^0) \\ t(\omega^1) \\ \dots \\ t(\omega^{15}) \end{bmatrix}
$$

where $\omega$ is the primitive root of unity used for the `LDE`, that is, $\omega$ satisfies $\omega^2 = g$. We need to recover $C(x)$, a polynomial whose degree can't be more than $t(x)$'s. Because $t$ was built by interpolating `8` points (the trace), we know we can recover $C(x)$ by interpolating it on 16 points. We choose these points to be the `LDE` roots of unity 

$$
\{\omega^0, \omega, \omega^2, \dots, \omega^{15}\}
$$

Remember that to evaluate $C(x)$ on these points, all we need are the evaluations of the polynomial

$$
t(xg^2) - t(xg) - t(x)
$$

as the zerofier ones we can compute easily. These become:

$$
t(\omega^0 g^2) - t(\omega^0 g) - t(\omega^0) \\
t(\omega g^2) - t(\omega g) - t(\omega) \\
t(\omega^2 g^2) - t(\omega^2 g) - t(\omega^2) \\
\vdots \\
t(\omega^{15} g^2) - t(\omega^{15} g) - t(\omega^{15}) \\
$$

If we remember that $\omega^2 = g$, this is

$$
t(\omega^4) - t(\omega^2) - t(\omega^0) \\
t(\omega^5) - t(\omega^3) - t(\omega) \\
t(\omega^6) - t(\omega^4) - t(\omega^2) \\
\vdots \\
t(\omega^{3}) - t(\omega) - t(\omega^{15}) \\
$$

and we can compute each evaluation here by calling `compute_transition` on the appropriate frame built from the `lde_trace`. Specifically, for the first evaluation we can build the frame:

$$
\begin{bmatrix} t(\omega^0) \\ t(\omega^2) \\ t(\omega^{4}) \end{bmatrix}
$$

Calling `compute_transition` on this frame gives us the first evaluation. We can get the rest in a similar fashion, which is what this piece of code in the `evaluate` method does:

```rust
for (i, d) in lde_domain.iter().enumerate() {
    let frame = Frame::read_from_trace(
        lde_trace,
        i,
        blowup_factor,
        &self.air.context().transition_offsets,
    )

    let mut evaluations = self.air.compute_transition(&frame);

    ...
}
```

Each iteration builds a frame as above and computes one of the evaluations needed. The rest of the code just adds the zerofier evaluations, along with the alphas and betas. It then also computes boundary polynomial evaluations by explicitly constructing them.

### Verifier

The verifier employs the same trick to reconstruct the evaluations on the out of domain point $C_i(z)$ for the consistency check.

# Even/odd decomposition for `H`

At the end of the recap we talked about how in our code we don't actually commit to `H`, but rather an even/odd decomposition for it. These are two polynomials `H_1` and `H_2` that satisfy

$$
H(x) = H_1(x^2) + x H_2(x^2)
$$

This all happens on this piece of code

```rust
let composition_poly =
    constraint_evaluations.compute_composition_poly(&lde_roots_of_unity_coset);

let (composition_poly_even, composition_poly_odd) = composition_poly.even_odd_decomposition();

// Evaluate H_1 and H_2 in z^2.
let composition_poly_evaluations = vec![
    composition_poly_even.evaluate(&z_squared),
    composition_poly_odd.evaluate(&z_squared),
];
```

After this, we don't really use `H` anymore, but rather `H_1` and `H_2`. There's not that much to say other than that.

# Out of Domain Frame

As part of the consistency check, the prover needs to provide evaluations of the trace polynomials in all the points needed by the verifier to check that `H` was constructed correctly. In the fibonacci example, these are $t(z)$, $t(zg)$, and $t(zg^2)$. In code, the prover passes these evaluations as a `Frame`, which we call the out of domain (`ood`) frame. 

The reason we do this is simple: with the frame in hand, the verifier can reconstruct the evaluations of the constraint polynomials $C_i(z)$ by calling the `compute_transition` method on the ood frame and then adding the alphas, betas, and so on, just like we explained in the section above.

# Transcript

Throughout the protocol, there are a number of times where the verifier randomly samples some values that the prover needs to use (think of the alphas and betas used when constructing `H`). Because we don't actually have an interaction between prover and verifier, we emulate it by using a hash function, which we assume is a source of randomness the prover can't control.

The job of providing these samples for both prover and verifier is done by the `Transcript` struct, which you can think of as a stateful `rng`; whenever you call `challenge()` on a transcript you get a random value and the internal state gets mutated, so the next time you call `challenge()` you get a different one. You can also call `append` on it to mutate its internal state yourself. This is done a number of times throughout the protocol to keep the prover honest so it can't predict or manipulate the outcome of `challenge()`.

Notice that to sample the same values, both prover and verifier need to call `challenge` and `append` in the same order (and with the same values in the case of `append`) and the same number of times.

The idea explained above is called the Fiat-Shamir heuristic or just `Fiat-Shamir`, and is more generally used throughout proving systems to remove interaction between prover and verifier. Though the concept is very simple, getting it right so the prover can't cheat is not, but we won't go into that here.

# Proof

The generated proof has got all the information needed for the verifier to verify it:
- Trace length: The number of rows of the trace table, needed to know the max degree of the polynomials that appear in the system.
- LDE trace commitments.
- DEEP composition polynomial out of domain even and odd evaluations.
- DEEP composition polynomial root.
- FRI layers merkle roots.
- FRI last layer value.
- Query list.
- DEEP composition poly openings.
- Nonce: Proof of work setting used to generate the proof.

# Special considerations

## FFT evaluation and interpolation
When evaluating or interpolating a polynomial, if the input (be it coefficients or evaluations) size isn't a power of two then the FFT API will extend it with zero padding until this requirement is met. This is because the library currently only uses a radix-2 FFT algorithm.

Also, right now FFT only supports inputs with a size up to $2^{2^64}$ elements.

# Other

## Why use roots of unity?

Whenever we interpolate or evaluate trace, boundary and constraint polynomials, we use some $2^n$-th roots of unity. There are a few reasons for this:

- Using roots of unity means we can use the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) and its inverse to evaluate and interpolate polynomials. This method is much faster than the naive Lagrange interpolation one. Since a huge part of the STARK protocol involves both evaluating and interpolating, this is a huge performance improvement.
- When computing boundary and constraint polynomials, we divide them by their `zerofiers`, polynomials that vanish on a few points (the trace elements where the constraints do not hold). These polynomials take the form

    $$
    Z(X) = \prod (X - x_i)
    $$

    where the $x_i$ are the points where we want it to vanish.

    When implementing this, evaluating this polynomial can be very expensive as it involves a huge product. However, if we are using roots of unity, we can use the following trick. The vanishing polynomial for all the $2^n$ roots of unity is

    $$
    X^{2^n} - 1
    $$

    Instead of expressing the zerofier as a product of the places where it should vanish, we express it as the vanishing polynomial above divided by the `exemptions` polynomial; the polynomial whose roots are the places where constraints don't need to hold. 

    $$
    Z(X) = \dfrac{X^{2^n} - 1}{\prod{(X - e_i)}}
    $$

    where the $e_i$ are now the points where we don't want it to vanish. This `exemptions` polynomial in the denominator is usually much smaller, and because the vanishing polynomial in the numerator is only two terms, evaluating it is really fast.

## What is a primitive root of unity?

The $n$-th roots of unity are the numbers $x$ that satisfy

$$
x^n = 1
$$

There are $n$ such numbers, because they are the roots of the polynomial $X^n - 1$. The set of $n$-th roots of unity always has a `generator`, a root $g$ that can be used to obtain every other root of unity by exponentiating. What this means is that the set of $n$-th roots of unity is

$$
\{g^i : 0 \leq i < n\}
$$

Any such generator `g` is called a *primitive root of unity*. It's called primitive because it allows us to recover any other root.

Here are a few important things to keep in mind, some of which we use throughout our implementation:

- There are always several primitive roots. If $g$ is primitive, then any power $g^k$ with $k$ coprime with $n$ is also primitive. As an example, if $g$ is a primitive $8$-th root of unity, then $g^3$ is also primitive.
- We generally will not care about which primitive root we choose; what we do care about is being *consistent*. We should always choose the same one throughout our code, otherwise computations will go wrong.
- Because $g^n = 1$, the powers of $g$ wrap around. This means

    $$
    g^{n + 1} = g \\
    g^{n + 2} = g^2
    $$

  and so on.
- If $w$ is a primitive $2^{n + 1}$-th root of unity, then $w^2$ is a primitive $2^n$-th root of unity. In general, if $w$ is a primitive $2^{n + k}$-th primitive root of unity, then $w^{2^k}$ is a primitive $2^n$-th root of unity.

## Why use Cosets?

When we perform `FRI` on the `DEEP` composition polynomial, the low degree extension we use is not actually over a set of higher roots of unity than the ones used for the trace, but rather a *coset* of it. A coset is simply a set of numbers all multiplied by the same element. We call said element the `offset`. In our case, a coset of the $2^n$-th roots of unity with primitive root $\omega$ and offset `h` is the set

$$
\{h \omega^i : 0 \leq i < 2^n\}
$$

So why not just do the LDE without the offset? The problem is in how we construct and evaluate the composition polynomial `H`. Let's say our trace polynomial was interpolated over the $2^n$-th roots of unity with primitive root $g$, and we are doing the LDE over the $2^{n + 1}$-th roots of unity with primitive root $\omega$, so $\omega^2 = g$ (i.e. the blowup factor is `2`).

Recall that `H` is a sum of terms that include boundary and transition constraint polynomials, and each one of them includes a division by a `zerofier`; a polynomial that vanishes on some roots of unity $g^i$. This is because the zerofier is what tells us which rows of the trace our constraint should apply on.

When doing `FRI`, we have to provide evaluations over the LDE domain we are using. If we don't include the offset, our domain is

$$
\{\omega^i : 0 \leq i < 2^{n + 1}\}
$$

Note that, because $w^2 = g$, some of the elements on this set (actually, half of them) are powers of $g$. If while doing `FRI` we evaluate `H` on them, the zerofier could vanish and we'd be dividing by zero. We introduce the offset to make sure this can't happen.

NOTE: a careful reader might note that we can actually evaluate `H` on the elements $g^i$, since on a valid trace the zerofiers will actually divide the polynomials on their numerator. The problem still remains, however, because of performance. We don't want to do polynomial division if we don't need to, it's much cheaper to just evaluate numerator and denominator and then divide. Of course, this only works if the denominator doesn't vanish; hence, cosets.

