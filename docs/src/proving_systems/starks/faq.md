# Frequently Asked Questions

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

Note that, because $w^2 = g$, some of the elements on this set (actually, half of them) are powers of $g$. If while doing `FRI` we evalaute `H` on them, the zerofier will vanish and we'll be dividing by zero. We introduce the offset to make sure this can't happen.

NOTE: a careful reader might note that we can actually evaluate `H` on the elements $g^i$, since on a valid trace the zerofiers will actually divide the polynomials on their numerator. The problem still remains, however, because of performance. We don't want to do polynomial division if we don't need to, it's much cheaper to just evaluate numerator and denominator and then divide. Of course, this only works if the denominator doesn't vanish; hence, cosets.

----------

TODO:
- What's the ce blowup factor?
- What's the out of domain frame?
