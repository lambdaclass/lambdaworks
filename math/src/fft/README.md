# lambdaworks Fast Fourier Transform

This folder contains the fast Fourier transform (FFT) over finite fields (also known as number theoretic transform, NTT). Currently, the following algorithms are supported:
- Cooley-Tukey Radix-2
- Cooley-Tukey Radix-4

We are also planning on adding a mixed radix algorithm. To use the FFT, the length of the vector, $n$, should be a power of $2$ (or $4$), that is, $2^m = n$.

The core operation of the FFT is the butterfly. To combine the elements, we need to sample the twiddle factors, which we obtain from the roots of unity.
