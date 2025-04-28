# FFT Benchmarks

## Polynomial interpolation methods comparison

Three methods of polynomial interpolation were benchmarked, with different input sizes each time:

- **CPU Lagrange:** Finding the Lagrange polynomial of a set of random points via a naive algorithm (see `math/src/polynomial.rs:interpolate()`)
- **CPU FFT:** Finding the lowest degree polynomial that interpolates pairs of twiddle factors and Fourier coefficients (the results of applying the Fourier transform to the coefficients of a polynomial) (see `math/src/polynomial.rs:interpolate_fft()`).


All values of time are in milliseconds. Those cases which were greater than 30 seconds were marked respectively as they're too slow and weren't worth to be benchmarked. The input size refers to *d + 1* where *d* is the polynomial's degree (so size is amount of coefficients).

| Input size | CPU Lagrange | CPU FFT   |
|------------|--------------|-----------|
| 2^4        | 2.2 ms       | 0.2 ms    |
| 2^5        | 9.6 ms       | 0.4 ms    |
| 2^6        | 42.6 ms      | 0.8 ms    |
| 2^7        | 200.8 ms     | 1.7 ms    |
| ...        | ...          | ..        |
| 2^21       | >30000 ms    | 28745  ms |
| 2^22       | >30000 ms    | >30000 ms |
| 2^23       | >30000 ms    | >30000 ms |
| 2^24       | >30000 ms    | >30000 ms |
