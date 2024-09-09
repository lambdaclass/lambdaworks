# lambdaworks Polynomial Commitment Schemes

This folder contains lambdaworks polynomial commitment schemes (PCS). The following commitment schemes are supported:
- [KZG10](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf)

## Introduction to KZG commitment scheme

The Kate, Zaverucha, Goldberg (KZG) commitment is a polynomial commitment scheme that works over pairing-friendly elliptic curves, such as BN-254 and BLS12-381. It is important to have the following notation in mind:
- $\mathbb{F_p }$ is the base field of the curve, defined by the prime $p$.
- $\mathbb{F_r }$ is the scalar field associated with the curve, defined by the prime $r$.
- $G_1$ is the largest subgroup/group of prime order of the elliptic curve (the number of elements in the subgroup/group is $r$).
- $G_2$ is the subgroup/group of prime order (equal to $r$) of the twist curve.
- $G_t$ is the multiplicative subgroup of the $r$-th roots of unity of an extension field. For BN-254 and BLS12-381, the extension field is $\mathbb{F_{p^{12} }}$ (a degree twelve extension) and each element of $x \in G_t$ satisfies that $x^r = 1$.

Throughout, we will use the additive notation for the groups $G_1$ and $G_2$ and multiplicative notation for $G_t$. So, given elements $x , y \in G_1$, we will write $x + y = z$, but if $x , y \in G_t$, we have $x \times y = z$.

An elliptic curve is given by the pairs of points $(x , y)$ in $\mathbb{F_p } \times \mathbb{F_p }$, satisfying the equation $y^2 = x^3 + a x + b$. We can find a cyclic group/subgroup of order $r$ that satisfy the equation. This is the group $G_1$.

In a similar way, the group $G_2$ is given by a cyclic group of prime order $r$ that satisfied the twisted curve's equation $y^2 = x^3 + a^\prime x + b^\prime$, where $x, y$ live in an extension field of $\mathbb{F_p }$, typically $\mathbb{F_{p^2 } }$.

Given that both $G_1$, $G_2$ and $G_t$ are cyclic groups, we have elements $g_1$, $g_2$ and $g_t$, called generators, such that when we apply the group operation repeatedly, we span all the elements in the group. For notation purposes, we will denote $[a]_1 = a g_1 = g_1 + g_1 + g_1 + g_1 + ... + g_1$, where we add $a$ copies of $g_1$. Similarly, $[a]_2 = a g_2$ and $[a]_t = g_t^{a}$. More concretely, $\{ g_1 , 2g_1 , 3g_1 , 4g_1 , \dots (r - 1)g_1 \} = G_1$. Note that if we do $m g_1$, and $m \geq r$, then this will yield the same as $s g_1$ where $s \equiv m \pmod{r}$ and $0 \leq s \leq r - 1$.

The whole scheme depends on a pairing function (also known as bilinear map) $e: G_1 \times G_2 \rightarrow G_t$ which satisfies the following properties:
