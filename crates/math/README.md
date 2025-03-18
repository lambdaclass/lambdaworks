# lambdaworks-math [![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/lambdaworks-math.svg
[crates.io]: https://crates.io/crates/lambdaworks-math


## Usage
Add this to your `Cargo.toml`
```toml
[dependencies]
lambdaworks-math = "0.11.0"
```

## Structure
This crate contains all the relevant mathematical building blocks needed for proof systems and cryptography. The main parts are:
- [Finite Fields](./src/field/README.md)
- [Elliptic curves](./src/elliptic_curve/README.md)
- [Polynomials - univariate and multivariate](./src/polynomial/README.md)
- [Large unsigned integers](./src/unsigned_integer/)
- [Fast Fourier Transform](./src/fft/README.md)
- [Optimized Multiscalar Multiplication](./src/msm/)
