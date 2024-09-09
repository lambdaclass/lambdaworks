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
- $e(x , y) \neq 1$ if $x \neq \mathcal{O}$ and $y \neq \mathcal{O}$ (non-degeneracy).
- $e([a]_1 , [b]_2 ) = \left( e(g_1 , g_2)\right)^{a b} = g_t^{ab} (bilinearity).

There are two parties, prover and verifier. They share a public Structured Reference String (SRS) or trusted setup, given by:
- $\{ g_1 , [\tau]_1 , [\tau^2 ]_1 , \dots , [\tau^{n - 1} ]_1 \}$ consisting of the powers of a random, yet secret number $\tau$, with degree bounded by $n - 1$, hidden inside $G_1$
- $\{ g_2 , [\tau]_2  \}$ = $\{ g_2 , \tau g_2  \}$ contains the generator and $\tau$ hidden inside $G_2$ (for practical purposes, we don't need additional powers).

Knowing these sets of points does not allow recovering the secret $\tau$ or any of its powers (security under the algebraic model).

A polynomial of degree bound by $n - 1$ is an expression $p(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_{n - 1} x^{n - 1}$. The coefficient $a_k \neq 0$ accompanying the largest power is called the degree of the polynomial $\mathrm{deg} (p)$. The coefficients of the polynomial belong to the field $\mathbb{F_r }$. We can commit to a polynomial of degree at most $n - 1$ by performing the following multiscalar multiplication (MSM):

$P = \mathrm{cm} (p) = a_0 g_1 + a_1 [\tau ]_1 + a_2 [\tau^2 ]_1 + \dots + a_{n - 1} [\tau^{n - 1} ]_1$ 

This operation works, since we have points over an elliptic curve, and multiplication by a scalar and addition are defined properly. The commitment to $P$ is a point on the elliptic curve, which is equal to $p(\tau ) g_1 = P$. This commitment achieves the two properties we need:
- Hiding
- Binding

Given a commitment to $p$, we can prove evaluations of $p$ at points $z$. We will focus on the simplest case, where we want to show that $p(z) = y$, where $z$ and $y$ live in $\mathbb{F_r}$. The following fact will help us prove the evaluation (it is called the [polynomial remainder theorem](https://en.wikipedia.org/wiki/Polynomial_remainder_theorem)):

If $p(z) = y$ then $p^\prime = p(x) - y$ is divisible by $x - z$. Another way to state this is that there exists a polynomial $q(x)$ of degree $\mathrm{deg} (p) - 1$ such that $p(x) - y = p^\prime (x) = (x - z) q(x)$. 

Providing the quotient would allow the verifier to check the evaluation, but the problem is that the verifier does not know $p(x)$ in full. Given that $\tau$ is a secret point at random, we could check the previous equality in just one point, $\tau$, that is:

$p(\tau ) - y = (\tau - z) q(\tau )$

Due to the [Schwartz-Zippel lemma](https://en.wikipedia.org/wiki/Schwartz%E2%80%93Zippel_lemma), if the equality above holds, then, with high probability $p(x) - y = p^\prime (x) = (x - z) q(x)$. We could send, therefore $Q = \mathrm{cm} (q) = q(\tau ) g_1$ using the MSM. If $q(x) = b_0 + b_1 x + b_2 x^2 + \dots + b_{n - 1} x^{n - 2}$,

$Q = \mathrm{cm} (q) = b_0 g_1 + b_1 [\tau ]_1 + \dots b_{n - 2} [\tau^{n - 2} ]_1$

In the context of EIP-4844, $P$ is called the commitment and $Q$ is the evaluation proof. We can use the pairing to check the equality at $\tau$. We compute the following:
- $e( P - y g_1 , g_2 ) = \left( e(g_1 , g_2 ) \right)^{ p(\tau ) - y }$
- $e( Q , [\tau]_2 - z g_2 ) = \left( e(g_1 , g_2 ) \right)^{ q(\tau ) (\tau - z)}$

If the two pairings are equal, this means that $p(\tau ) - y \equiv q(\tau ) (\tau - z) \pmod{r}$. In practice, we use an alternative formulation,

$e( P - y g_1 , - g_2 ) \times e( Q , [\tau]_2 - z g_2 ) \equiv 1 \pmod{r}$

This is more efficient, since we can compute the pairing using two Miller loops but just one final exponentiation.

## References

- [Constantine](https://github.com/mratsim/constantine/blob/master/constantine/commitments/kzg.nim)
- [EIP-4844](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-4844.md)
- [KZG proof verification](https://github.com/ethereum/consensus-specs/blob/v1.4.0-beta.1/specs/deneb/polynomial-commitments.md#verify_kzg_proof_batch)
- [Multiproofs KZG](https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html)
- [Fast amortized KZG proofs](https://eprint.iacr.org/2023/033)