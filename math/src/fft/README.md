# lambdaworks Fast Fourier Transform

This folder contains the [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT) over finite fields (also known as number theoretic transform, NTT). If you are unfamiliar with how lambdaworks handles fields, see [examples](https://github.com/lambdaclass/lambdaworks/blob/main/examples/README.md). Currently, the following algorithms are supported:
- Cooley-Tukey Radix-2
- Cooley-Tukey Radix-4

We are also planning on adding a mixed radix algorithm. To use the FFT, the length of the vector, $n$, should be a power of $2$ (or $4$), that is, $2^m = n$. The FFT should be used with fields implementing the `IsFFTFriendly` trait. The FFT works by recursively breaking a length $n$ FFT into $2$ $n/2$ FFTs, until we reach a sufficiently small size that can be solved. The core operation of the FFT is the butterfly. To combine the elements, we need to sample the twiddle factors, which we obtain from the roots of unity. The FFT can accept the input in natural order and returns the output in reverse order (nr) or the input is in reverse order and the output is in natural order (rn).

Since the main applications of the FFT are related to polynomial evaluation and interpolation, we provide functions describing these operations, which call the FFT under the hood:
- `evaluate_fft`
- `evaluate_offset_fft`
- `interpolate_fft`
- `interpolate_offset_fft`

These functions can be used with [univariate polynomials](https://github.com/lambdaclass/lambdaworks/tree/main/math/src/polynomial). To use the functions,
```rust
let p_1 = Polynomial::new(&[FE::new(3), FE::new(4), FE::new(5) FE::new(6)]);
let evaluations = Polynomial::evaluate_offset_fft(p_1, 4, 4, FE::new(3))?;
```
Interpolate takes a vector of length $2^m$, which we take them to be the evaluations of a polynomial $p$ over values of $x$ of the form $\{ offset.g^0, offset.g, offset.g^2 \dots offset.g^{n - 1} \}$, where $g$ is a generator of the $n$-th roots of unity. For example,
```rust
let evaluations = [FE::new(1), FE::new(2), FE::new(3) FE::new(4)]
let poly = Polynomial::interpolate_fft(&evaluations).unwrap()
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
