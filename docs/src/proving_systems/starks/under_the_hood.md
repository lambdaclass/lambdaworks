# How this works under the hood

In this section we go over how a few things in the `prove` and `verify` functions are implemented. If you just need to *use* the prover, then you probably don't need to read this. If you're going through the code to try to understand it, read on.

We will once again use the fibonacci example as an ilustration. Recall from the `recap` that the main steps for the prover and verifier are the following:

### Prover side

- Compute the trace polynomial `t` by interpolating the trace column over a set of $2^n$-th roots of unity $\{g^i : 0 \leq i < 2^n\}$.
- Compute the boundary polynomial `B`.
- Compute the transition constraint polynomial `C`.
- Construct the composition polynomial `H` from `B` and `C`.
- Sample an out of domain point `z` and provide the evaluation $H(z)$ and all the necessary trace evaluations to reconstruct it. In the fibonacci case, these are $t(z)$, $t(zg)$, and $t(zg^2)$.
- Construct the deep composition polynomial `Deep(x)` from `H`, `t`, and the evaluations from the item above.
- Do `FRI` on `Deep(x)` and provide the resulting FRI commitment to the verifier.

### Verifier side

- Take the evaluation $H(z)$ along with the trace evaluations the prover provided.
- Reconstruct the evaluations $B(z)$ and $C(z)$ from the trace evaluations. Check that the claimed evaluation $H(z)$ the prover gave us actually satisfies
    $$
    H(z) = B(z) (\alpha_1 z^{D - deg(B)} + \beta_1) + C(z) (\alpha_2 z^{D - deg(C)} + \beta_2)
    $$
- Take the provided `FRI` commitment and check that it verifies.

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
B_i(x) (\alpha_i x^{D - deg(B)} + \beta_i)
$$

and constraint terms

$$
C_i(x) (\alpha_i x^{D - deg(C)} + \beta_i)
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
