# lambdaworks Fast Fourier Transform

This folder contains the [fast Fourier transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT) over finite fields (also known as number theoretic transform, NTT). If you are unfamiliar with how lambdaworks handles fields, see [examples](https://github.com/lambdaclass/lambdaworks/blob/main/examples/README.md). Currently, the following algorithms are supported:
- Cooley-Tukey Radix-2
- Cooley-Tukey Radix-4

We are also planning on adding a mixed radix algorithm. To use the FFT, the length of the vector, $n$, should be a power of $2$ (or $4$), that is, $2^m = n$. The FFT should be used with fields implementing the ´IsFFTFriendly´ trait.

The core operation of the FFT is the butterfly. To combine the elements, we need to sample the twiddle factors, which we obtain from the roots of unity.

Since the main applications of the FFT are related to polynomial evaluation and interpolation, we provide functions describing these operations, which call the FFT under the hood:
- ´evaluate_fft´
- ´evaluate_offset_fft´
- ´interpolate_fft´
- ´interpolate_offset_fft´
