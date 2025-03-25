# Circle Fast-Fourier Transform (CircleFFT)

This folder contains all the necessary tools to work with the [circle FFT](https://eprint.iacr.org/2024/278), which is a suitable way of performing an analogue of the [radix-2 FFT algorithm](../fft/README.md) over fields which are not smooth. We say a finite field is smooth if the size of multiplicative group of the field is divisible by a sufficiently high power of 2. In the case of $\mathbb{Z}_p$, the previous sentence indicates that $p - 1 = 2^m c$, where $m$ is sufficiently large (for example, $2^{25}$), ensuring we can use the radix-2 Cooley-Tuckey algorithm for the FFT with vectors of size up to $2^{25}$.

For an introduction to circle STARKs, we recommend [our blog](https://blog.lambdaclass.com/an-introduction-to-circle-starks/) or [Vitalik's explanation](https://vitalik.eth.limo/general/2024/07/23/circlestarks.html)

## What is the Circle Group?

The Circle group consists of all points (x, y) such that x² + y² = 1, where the group operation is:

(a, b) + (c, d) = (a * c - b * d, a * d + b * c)

This mathematical structure provides several advantages for computational operations:

1. **Computational efficiency**: Allows faster operations compared to other structures.
2. **Fast transform**: Enables a version of FFT (Fast Fourier Transform) adapted to the Circle group (CFFT).
3. **Algebraic properties**: Provides special properties that can be leveraged for various cryptographic applications.

## Implementation

This module includes the following components:

- **CirclePoint**: Represents a point in the Circle group.
- **CFFT (Circle Fast Fourier Transform)**: Algorithm for evaluating polynomials at multiple points in the Circle group.
- **Cosets**: Implementation of cosets of the Circle group.
- **Polynomials**: Operations with polynomials adapted to the Circle group.
- **Twiddles**: Rotation factors used in the CFFT algorithm.

These components provide the foundation for building Circle-based cryptographic applications, including Circle STARKs.

## Implementation Details for CFFT

### Overview
The Circle Fast Fourier Transform (CFFT) is used to evaluate polynomials efficiently over points in a standard coset of the Circle group. The implementation uses an in-place FFT algorithm operating on slices whose length is a power of two.

### In-Place FFT Computation
The `cfft` function processes the input through `log₂(n)` layers (where *n* is the input length). In each layer:
- The input is divided into chunks of size `2^(i+1)` (for layer *i*).
- Butterfly operations are applied to pairs of elements within each chunk using precomputed twiddle factors.

### Twiddle Factors
- **Purpose:** Twiddle factors are rotation factors used in the butterfly operations.
- **Computation:** They are computed by the `get_twiddles` function based on coset points in the Circle group.
- **Configuration:** The factors are configured differently depending on whether the CFFT is used for evaluation or interpolation (e.g., order reversal or inversion).

### Bit-Reversal Permutation
Before applying the FFT:
- The input coefficients must be reordered into bit-reversed order.
- This reordering, performed by a helper function (e.g., `in_place_bit_reverse_permute`), is essential for the correctness of the in-place FFT computation.

### Result Ordering
After the FFT:
- The computed evaluations are not in natural order.
- The helper function `order_cfft_result_naive` rearranges these evaluations into the natural order, aligning them with the original order of the coset points.

### Inverse FFT (ICFFT)
- **Process:** The inverse FFT (`icfft`) mirrors the forward FFT's layered approach, but with operations that invert the transformation.
- **Scaling:** After processing, the result is scaled by the inverse of the input length to recover the original polynomial coefficients.

### Design Considerations and Optimizations
- **Modularity:** Separating FFT computation from data reordering enhances the maintainability and clarity of the implementation.
- **Optimization Potential:** Although the current ordering functions use a straightforward (naive) approach, there is room for further optimization (e.g., in-place value swapping).
- **Balance:** The design strikes a balance between code clarity and performance, making it easier to understand and improve in the future.

## API Usage

### Working with Circle group points

```rust
use lambdaworks_math::field::{
    element::FieldElement,
    fields::mersenne31::field::Mersenne31Field,
};
use lambdaworks_math::circle::point::CirclePoint;

// Create a point in the Circle group
let x = FieldElement::<Mersenne31Field>::from(2);
let y = FieldElement::<Mersenne31Field>::from(3);
let normalized_point = CirclePoint::new(x, y);

// Get the neutral element (identity) of the group
let zero = CirclePoint::<Mersenne31Field>::zero();

// Group operations
let point1 = CirclePoint::<Mersenne31Field>::GENERATOR;
let point2 = point1.double(); // Double a point
let point3 = point1 + point2; // Add two points
let point4 = point1 * 8; // Scalar multiplication
```

### Polynomial evaluation using CFFT

```rust
use lambdaworks_math::field::{
    element::FieldElement,
    fields::mersenne31::field::Mersenne31Field,
};
use lambdaworks_math::circle::polynomial::{evaluate_cfft, interpolate_cfft};

// Define polynomial coefficients
let coefficients: Vec<FieldElement<Mersenne31Field>> = (0..8)
    .map(|i| FieldElement::<Mersenne31Field>::from(i as u64))
    .collect();

// Evaluate the polynomial at Circle group points
let evaluations = evaluate_cfft(coefficients.clone());

// Interpolate to recover the coefficients
let recovered_coefficients = interpolate_cfft(evaluations);
```

### Using CFFT directly

```rust
use lambdaworks_math::field::{
    element::FieldElement,
    fields::mersenne31::field::Mersenne31Field,
};
use lambdaworks_math::circle::{
    cfft::{cfft, icfft},
    cosets::Coset,
    twiddles::{get_twiddles, TwiddlesConfig},
};
use lambdaworks_math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;

// Prepare data (must be a power of 2 in length)
let mut data: Vec<FieldElement<Mersenne31Field>> = (0..8)
    .map(|i| FieldElement::<Mersenne31Field>::from(i as u64))
    .collect();

// Prepare for CFFT
let domain_log_2_size = data.len().trailing_zeros();
let coset = Coset::new_standard(domain_log_2_size);
let config = TwiddlesConfig::Evaluation;
let twiddles = get_twiddles(coset.clone(), config);

// Bit-reverse permutation (required before CFFT)
in_place_bit_reverse_permute(&mut data);

// Perform CFFT in-place
cfft(&mut data, twiddles);

// For inverse CFFT
let inverse_config = TwiddlesConfig::Interpolation;
let inverse_twiddles = get_twiddles(coset, inverse_config);

// Perform inverse CFFT
icfft(&mut data, inverse_twiddles);
```

### Working with Cosets

```rust
use lambdaworks_math::circle::cosets::Coset;
use lambdaworks_math::circle::point::CirclePoint;
use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;

// Create a standard coset of size 2^n
let log_2_size = 3; // For a coset of size 8
let coset = Coset::new_standard(log_2_size);

// Get the coset generator
let generator = coset.generator;

// Get all elements of the coset
let elements = coset.elements();
```

## Applications

The Circle group operations and CFFT implementation in this module serve as fundamental building blocks for various cryptographic applications, including:

1. **Circle STARKs**: A variant of STARKs that can be built using these Circle operations.
2. **Efficient polynomial operations**: For fields that don't have the smoothness property required by traditional FFT.
3. **Zero-knowledge proof systems**: Components for building efficient ZK systems.

## References

- [An Introduction to Circle STARKs](https://blog.lambdaclass.com/an-introduction-to-circle-starks/) - LambdaClass Blog
- [Vitalik's Explanation of Circle STARKs](https://vitalik.eth.limo/general/2024/07/23/circlestarks.html)
- [Circle FFT Paper](https://eprint.iacr.org/2024/278)
- [Anatomy of a STARK](https://aszepieniec.github.io/stark-anatomy/) - Detailed explanation of STARKs
- [STARKs, Part I: Proofs with Polynomials](https://vitalik.ca/general/2017/11/09/starks_part_1.html) - Vitalik Buterin's series on STARKs
