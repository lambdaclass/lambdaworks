# Brief introduction to polynomial evaluation and the Fourier transform

## What is the Fourier transform?

The Fourier transform is a mathematical operation whose development was motivated after Joseph Fourier claimed that a there's a set of functions, which includes discontinuous and non-periodic ones, which can be represented by a series of trigonometric functions weighted by scalars called the *Fourier coefficients* obtained via the FT. It transforms the *way of representing a function*, from one into another.

The FT has enormous use cases in many different areas, mainly in the study of signals and systems. In particular the *Discrete* Fourier Transform (DFT) can be used for studying a discrete-time signal, which is defined as a set of points $a[n]$.

The DFT can be thought as a matrix multiplication in the form of:

$$
Y = WX
$$

where $W$ is a square matrix, called the *DFT matrix*, $X$ a column vector with the elements of $a[n]$ (the input) and $Y$ the output of the DFT.

## Coefficient and point-value representation of a polynomial
A polynomial of degree $d$ is typically represented as a collection of $d + 1$ elements, called the polynomial's coefficients:

$$
P(x) = a_0 + a_1x + a_2x^2 + a_3x^3 + \ldots + a_dx^d
$$

there's an alternative way of representing a poly apart from its coefficients, called the point-value representation. There's a theorem which states that there exists a **unique** polynomial of degree $d$ that *interpolates* $d + 1$ data points $(x_0, y_0), (x_1, y_1), \ldots$, and also a polynomial $P(x)$ is said to *interpolate* a point set if every one is contained in the image of $P(x)$ (simply meaning that there's only one polynomial whose curve passes through all the points).

So, we can add names to the act of transforming a representation into another:
  - a polynomial is **evaluated** when some values (called evaluations) are obtained and paired with their pre-images to form a point-value set which represents the polynomial. This is calculating $(x_i, P(x_i))$ for $i = 0, 1, 2, \ldots, d$, where $d$ is the polynomial's degree.
  - a polynomial is **interpolated** when its coefficient representation is found from the points it interpolates. One can think of interpolation as the
  inverse of evaluation (because composing these two yield an identity).

If $a[n]$ is a succession that holds every coefficient of the coefficient representation of a polynomial, then a way of evaluating it is to multiply a column vector of these with a square matrix which holds the $x_0, x_1, x_2, \ldots$ pre-image elements used for obtaining evaluations. These need to be elevated to some power that depends on the value's position on the matrix. The DFT matrix fulfills these requirements, meaning that the **Discrete Fourer Transform is a way of evaluating a polynomial**. The elements of the DFT matrix are called the *twiddle factors*.

Because of this matrix structure, the DFT matrix is easily invertible, so executing an inverse operation of the DFT, called the *Inverse Discrete Fourier Transform* is trivial:

$$
X = W^{-1}Y
$$

meaning that one could also interpolate a polynomial via the execution of a Fourier transform whose twiddle factors (elements of $W$) were modified in a specific manner.

Executing a DFT for polynomial evaluation has an equivalent time complexity ($O(n^2)$) than evaluating it in a naive way (by calculating all $P(x_i)$), but there exists a faster algorithm for a DFT, with a complexity of $O(n\log n)$, called the *Fast Fourier Transform*.

## What does this all have to do with Lambdaworks?
As it was said on our [FAQ](../proving_systems/starks/faq.md), a huge part of the STARK protocol involves evaluating and interpolating polynomials as the point-value representation is more efficient and straightforward to use for operating with polynomials.
