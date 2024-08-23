# Protocol

## Details and tricks

### Polynomial commitment scheme

A polynomial commitment scheme (PCS) is a cryptographic tool that allows one party to commit to a polynomial, and later prove properties of that polynomial.
This commitment polynomial hides the original polynomial's coefficients and can be publicly shared without revealing any information about the original polynomial.
Later, the party can use the commitment to prove certain properties of the polynomial, such as that it satisfies certain constraints or that it evaluates to a certain value at a specific point.

In the implementation section we'll explain the inner workings of the Kate-Zaverucha-Goldberg scheme, a popular PCS chosen in Lambdaworks for PLONK.

For the moment we only need the following about it:

It consists of a finite group $\mathbb{G}$ and the following algorithms:
- **Commit($f$)**: This algorithm takes a polynomial $f$ and produces an element of the group $\mathbb{G}$. It is called the commitment of $f$ and is denoted by $\left[f\right]_1$. It is homomorphic in the sense that $\left[f + g\right]_1 = \left[f\right]_1 + \left[g\right]_1$. The former sum being addition of polynomials. The latter is addition in the group $\mathbb{G}$.
- **Open($f$, $\zeta$ )**: It takes a polynomial $f$ and a field element $\zeta$ and produces an element $\pi$ of the group $\mathbb{G}$. This element is called an opening proof for $f(\zeta)$. It is the proof that $f$ evaluated at $\zeta$ gives $f(\zeta)$.
- **Verify($\left[f\right]_1$, $\pi$, $\zeta$, $y$)**: It takes group elements $\left[f\right]_1$ and $\pi$, and also field elements $\zeta$ and $y$. With overwhelming probability it outputs _Accept_ if $f(z)=y$ and _Reject_ otherwise.


### Blindings

As you will see in the protocol, the prover reveals the value taken by a bunch of the polynomials at a random $\zeta$. In order for the protocol to be _Honest Verifier Zero Knowledge_, these polynomials need to be _blinded_. This is a process that makes the values of these polynomials at $\zeta$ seemingly random by forcing them to be of certain degree. Here's how it works.

Let's take for example the polynomial $a$ the prover constructs. This is the interpolation of the first column of the trace matrix $T$ at the domain $H$.
This matrix has all of the left operands of all the gates. The prover wishes to keep them secret.
Say the trace matrix $T$ has $N$ rows. $H$ is $\{1, \omega,\omega^2, \dots, \omega^{N - 1}\}$. The invariant that the prover cannot violate is that $a_{\text{blinded}}(\omega^i)$ must take the value $T_{0, i}$, for all $i$. This is what the interpolation polynomial $a$ satisfies. And is the unique such polynomial of degree at most $N - 1$ with such property. But for higher degrees, there are many such polynomials.

The _blinding_ process takes $a$ and a desired degree $M\geq N$, and produces a new polynomial $a_{\text{blinded}}$ of degree exactly $M$. This new polynomial satisfies that $a_{\text{blinded}}(\omega^i) = a(\omega^i)$ for all $i$. But outside $H$ differs from $a$.

This may seem hard but it's actually very simple. Let $z_H$ be the polynomial $z_H = X^N - 1$. If $M = N + k$, with $k \geq 0$, then sample random values $b_0, \dots, b_k$ and define
$$a_{\text{blinded}} := (b_0 + b_1 X + \cdots + b_k X^k )z_H + a$$

The reason why this does the job is that $z_H(\omega^i) = 0$ for all $i$. Therefore the added term vanishes at $H$ and leaves the values of $a$ at $H$ unchanged.

### Linearization trick

This is an optimization in PLONK to reduce the number of checks of the verifier.

One of the main checks in PLONK boils down to check that $p(\zeta) = z_H(\zeta) t(\zeta)$, with $p$ some polynomial that looks like $p = a q_L + b q_R + ab q_M + \cdots$, and so on. In particular the verifier needs to get the value $p(\zeta)$ from somewhere.

For the sake of simplicity, in this section assume $p$ is exactly $a q_L + bq_R$. Secret to the prover here are only $a, b$. The polynomials $q_L$ and $q_R$ are known also to the verifier. The verifier will already have the commitments $\left[a\right]_1, \left[b\right]_1, \left[q_L\right]_1$ and $\left[q_R\right]_1$. So the prover could send just $a( \zeta )$, $b( \zeta )$ along with their opening proofs and let the verifier compute by himself $q_L(\zeta)$ and $q_R(\zeta)$. Then with all these values the verifier could compute $p(\zeta) = a(\zeta) q_L (\zeta) + b(\zeta) q_R (\zeta)$. And also use his commitments to validate the opening proofs of $a(\zeta)$ and $b(\zeta)$.

This has the problem that computing $q_L (\zeta)$ and $q_R (\zeta)$ is expensive. The prover can instead save the verifier this by sending also $q_L (\zeta), q_R (\zeta)$ along with opening proofs. Since the verifier will have the commitments $\left[q_L\right]_1$ and $\left[q_R\right]_1$ beforehand, he can check that the prover is not cheating and cheaply be convinced that the claimed values are actually $q_L(\zeta)$ and $q_R(\zeta)$. This is much better. It involves the check of four opening proofs and the computation of $p(\zeta)$ off the values received from the prover. But it can be further improved as follows.

As before, the prover sends $a(\zeta), b(\zeta)$ along with their opening proofs. She constructs the polynomial $f = a(\zeta)q_L + b(\zeta)q_R$. She sends the value $f(\zeta)$ along with an opening proof of it. Notice that the value of $f(\zeta)$ is exactly $p(\zeta)$. The verifier can compute by himself $\left[f\right]_1$ as $a(\zeta)\left[q_L\right]_1 + b(\zeta)\left[q_R\right]_1$. The verifier has everything to check all three openings and get convinced that the claimed value $f(\zeta)$ is true. And this value is actually $p(\zeta)$. So this means no more work for the verifier. And the whole thing got reduced to three openings.

This is called the linearization trick. The polynomial $f$ is called the _linearization_ of $p$.


## Setup

There's a one time setup phase to compute some values common to any execution and proof of the particular circuit. Precisely, the following commitments are computed and published.
$$\left[q_L\right]_1 , \left[q_R\right]_1 , \left[q_M\right]_1 , \left[q_O\right]_1 , \left[q_C\right]_1 , \left[S_{ \sigma 1 }\right]_1 , \left[S_{ \sigma 2 }\right]_1 , \left[S_{ \sigma 3 }\right]_1$$

## Proving algorithm

Next we describe the proving algorithm for a program of size $N$. That includes public inputs. Let $\omega$ be a primitive $N$-th root of unity. Let $H=\{1, \omega, \omega^2, \dots, \omega^{N - 1}\}$. Define $Z_H := X^N - 1$.

Assume the eight polynomials of common preprocessed input are already given.

The prover computes the trace matrix $T$ as described in the first sections. That means, with the first rows corresponding to the public inputs. It should be a $N \times 3$ matrix.

### Round 1

Add to the transcript the following:
$$\left[ S_{\sigma 1 } \right]_1, \left[ S_ { \sigma 2 } \right]_1, \left[ S { \sigma 3 } \right]_1, \left[ q_L \right]_1, \left[ q_R \right]_1, \left[ q_M \right]_1, \left[ q_O \right]_1, \left[ q_C \right]_1$$

Compute polynomials $a',b',c'$ as the interpolation polynomials of the columns of $T$ at the domain $H$.
Sample random $b_1, b_2, b_3, b_4, b_5, b_6$
Let

$a := (b_1 X + b_2 )Z_H + a'$

$b := (b_3 X + b_4 )Z_H + b'$

$c := (b_5 X + b_6 )Z_H + c'$

Compute $\left[a\right]_1, \left[b\right]_1, \left[c\right]_1$ and add them to the transcript.

### Round 2

Sample $\beta, \gamma$ from the transcript.

Let $z_0 = 1$ and define recursively for $0\leq k < N$.

$$
z_{k + 1} = z_k \frac{(a_k + \beta \omega^k + \gamma) (b_k + \beta \omega^k k_1 + \gamma)(c_k + \beta \omega^k k_2 + \gamma)}{(a_k + \beta S_{ \sigma 1} (\omega^k ) + \gamma) (b_k + \beta S_{ \sigma 2} (\omega^k) + \gamma)(c_k + \beta S_{ \sigma 3} (\omega^k) + \gamma)}
$$

Compute the polynomial $z'$ as the interpolation polynomial at the domain $H$ of the values $(z_0, \dots, z_{ N - 1 })$.

Sample random values $b_7, b_8, b_9$ and let $z = (b_7 X^2 + b_8 X + b_9 )Z_H + z'$.

Compute $\left[z\right]_1$ and add it to the transcript.

### Round 3

Sample $\alpha$ from the transcript.

Let $pi$ be the interpolation of the public input matrix $PI$ at the domain $H$.

Let

$$
\begin{aligned}
p_1 &= aq_L + bq_R + abq_M + cq_O + q_C + pi \\
p_2 &= (a + \beta X + \gamma)(b + \beta k_1 X + \gamma)(c + \beta k_2 X + \gamma)z -  (a + \beta S_{ \sigma 1} + \gamma)(b + \beta S_{ \sigma 2} + \gamma)(c + \beta S_{ \sigma 3} + \gamma)z(\omega X)\\
p_3 &= (z - 1)L_1
\end{aligned}
$$

and define $p = p_1 + \alpha p_2 + \alpha^2 p_3$. Compute $t$ such that $p = t Z_H$. Write $t = t_{lo}' + X^{N + 2} t_{mid}' + X^{ 2 ( N + 2 )} t_{hi}'$ with $t_{lo}', t_{mid}'$ and $t_{hi}'$ polynomials of degree at most $N + 1$.

Sample random $b_{10}, b_{11}$ and define

$$
\begin{aligned}
t_{lo} &= t_{lo}' + b_{10} X^{ N + 2} \\
t_{mid} &= t_{mid}' - b_{10} + b_{11} X^{ N + 2} \\
t_{hi} &= t_{hi}' - b_{11}
\end{aligned}
$$

Compute $\left[t_{lo} \right]_1, \left[t_{mid} \right]_1,\left[t_{hi} \right]_1$ and add them to the transcript.

### Round 4

Sample $\zeta$ from the transcript.

Compute $\bar a = a(\zeta), \bar b = b(\zeta), \bar c = c(\zeta), \bar s_{\sigma 1} =  S_{\sigma 1}(\zeta), \bar s_{\sigma 2} = S_{\sigma 2}(\zeta), \bar z_\omega = z(\zeta\omega)$ and add them to the transcript.

### Round 5

Sample $\upsilon$ from the transcript.

Let

$$
\begin{aligned}
\hat p_{nc1} &= \bar aq_L + \bar bq_R + \bar a\bar bq_M + \bar cq_O + q_C \\
\hat p_{nc2} &=(\bar a + \beta\zeta + \gamma)(\bar b + \beta k_1\zeta + \gamma)(\bar c + \beta k_2 \zeta + \gamma)z - (\bar a + \beta \bar s_{ \sigma 1 } + \gamma)(\bar b + \beta \bar s_{ \sigma 2 } + \gamma)\beta \bar z_\omega S_{ \sigma 3 } \\
\hat p_{nc3} &= L_1(\zeta) z
\end{aligned}
$$

Define

$$
\begin{aligned}
p_{nc} &= p_{nc1} + \alpha p_{nc2} + \alpha^2 p_{nc3} \\
t_{\text{partial}} &= t_{lo} + \zeta^{ N + 2}t_{mid} + \zeta^{ 2 ( N + 2 )}t_{hi}
\end{aligned}
$$

The subscript $nc$ stands for "non-constant", as is the part of the linearization of $p$ that has non-constant factors. The subscript "partial" indicates that it is a partial evaluation of $t$ at $\zeta$. Partial meaning that only some power of $X$ ar replaced by the powers of $\zeta$. So in particular $t_{\text{partial}}(\zeta) = t(\zeta)$.

Let $\pi_{\text{batch}}$ be the opening proof at $\zeta$ of the polynomial $f_{\text{batch}}$ defined as
$$t_{\text{partial}} +\upsilon p_{nc} + \upsilon^2 a + \upsilon^3 b + \upsilon^4 c + \upsilon^5 S_{\sigma 1} + \upsilon^6 S_{\sigma 2}$$

Let $\pi_{\text{single}}$ be the opening proof at $\zeta\omega$ of the polynomial $z$.

Compute $\bar p_{nc} := p_{nc}(\zeta)$ and $\bar t = t(\zeta)$.

### Proof

The proof is:
$$\left[a\right]_1, \left[b\right]_1 , \left[c\right]_1, \left[z\right]_1, \left[t_{lo} \right]_1, \left[t_{mid} \right]_1, \left[t_{hi} \right]_1, \bar{a}, \bar{b}, \bar{c}, \bar{s_{ \sigma 1 }}, \bar{s_{ \sigma 2 }}, \bar{z_\omega}, \pi_{\mathrm{batch}}, \pi_{\mathrm{single}}, \bar p_{nc}, \bar t$$

## Verification algorithm

### Transcript initialization

The first step is to initialize the transcript in the same way the prover did, adding to it the following elements.
$$\left[ S_{ \sigma 1 } \right]_1, \left[ S_{ \sigma 2 } \right]_1, \left[ S_{ \sigma 3 }\right]_1, \left[ q_L \right]_1, \left[ q_R \right]_1, \left[ q_M \right]_1, \left[ q_O \right]_1, \left[ q_C \right]_1$$

### Extraction of values and commitments

#### Challenges

Firstly, the verifier needs to compute all the challenges. For that, he follows these steps:

- Add $\left[a\right]_1, \left[b\right]_1, \left[c\right]_1$ to the transcript.
- Sample two challenges $\beta, \gamma$.
- Add $\left[z\right]_1$ to the transcript.
- Sample a challenge $\alpha$.
- Add $\left[ t_{lo} \right]_1, \left[ t_{mid} \right]_1, \left[ t_{hi} \right]_1$ to the transcript.
- Sample a challenge $\zeta$.
- Add $\bar a, \bar b, \bar c, \bar s_{\sigma 1}, \bar s_{\sigma 2}, \bar z_\omega$ to the transcript.
- Sample a challenge $\upsilon$.

#### Compute $p.i ( \zeta )$

Also he needs compute a few values off all these data. First, he computes the $PI$ matrix with the public inputs and outputs. He needs to compute $pi(\zeta)$, where $pi$ is the interpolation of $PI$ at the domain $H$. But he doesn't need to compute $pi$. He can instead compute $pi(\zeta)$ as
$$\sum_{i = 0 }^n L_i(\zeta) (PI)_i,$$
where $n$ is the number of public inputs and $L_i$ is the Lagrange basis at the domain $H$.

#### Compute claimed values of $p( \zeta )$ and $t( \zeta )$

He computes $\bar p_{c} := pi(\zeta) + \alpha \bar z_\omega (\bar c + \gamma) (\bar a + \beta \bar s_{\sigma 1} + \gamma) (\bar b + \beta \bar s_{\sigma 2} + \gamma) - \alpha^2 L_1(\zeta)$

This is the _constant_ part of the linearization of $p$. So adding it to what the prover claims to be $\bar p_{nc}$, he obtains
$$p(\zeta) = \bar p_{c} + \bar p_{nc}$$

With respect to $t(\zeta)$, this is actually already $\bar{t}$.

#### Compute $\left[t_{\mathrm{partial}} \right]_1$ and $\left[p_{nc} \right]_1$

He computes these off the commitments in the proof as follows
$$\left[t_{\mathrm{partial}}\right]_1 = \left[t_{lo}\right]_1 + \zeta^{ N + 2 }\left[t_{mid}\right]_1 + \zeta^{ 2 (N + 2) }\left[t_{hi}\right]_1$$

For $\left[p_{nc}\right]_1$, first compute

$$
\begin{aligned}
\left[\hat p_{nc1} \right]_1 &= \bar a\left[q_L\right]_1 + \bar b\left[q_R\right]_1 + (\bar a\bar b)\left[q_M\right]_1 + \bar c\left[q_O\right]_1 + \left[q_C\right]_1 \\
\left[\hat p_{nc2} \right]_1 &= (\bar a + \beta\zeta + \gamma)(\bar b + \beta k_1 \zeta + \gamma)(\bar c + \beta k_2 \zeta + \gamma)\left[z\right]_1 - (\bar a + \beta \bar s_{\sigma 1} + \gamma)(\bar b + \beta \bar s_{\sigma 2} + \gamma)\beta \bar z_\omega \left[S_{\sigma 3}\right]_1 \\
\left[\hat p_{nc3} \right]_1 &= L_1(\zeta)\left[z\right]_1
\end{aligned}
$$

Then $\left[p_{nc} \right]_1 = \left[p_{nc1} \right]_1 + \left[p_{nc2} \right]_1 + \left[p_{nc3} \right]_1$.

#### Compute claimed value $f_{\text{batch} } (\zeta)$ and $\left[f_{{batch} }\right]_1$

Compute $f_{\text{batch}} (\zeta)$ as

$$
f_{\mathrm{batch}} (\zeta) =
\bar t +\upsilon \bar p_{nc} + \upsilon^2 \bar a + \upsilon^3 \bar b + \upsilon^4 \bar c + \upsilon^5 \bar s_{\sigma1} + \upsilon^6 \bar s_{\sigma2}
$$

Also, the commitment of the polynomial $f_{\text{batch}}$ is
$$\left[f_{\mathrm{batch} }\right]_1 = \left[t_{\mathrm{partial}}\right]_1 + \upsilon \left[p_{nc}\right]_1 + \upsilon^2 \left[a\right]_1 + \upsilon^3 \left[b\right]_1 + \upsilon^4 \left[c\right]_1 + \upsilon^5 \left[S_{\sigma 1}\right]_1 + \upsilon^6 \left[S_{\sigma 2}\right]_1$$

### Proof check

Now the verifier has all the necessary values to proceed with the checks.

- Check that $p(\zeta)$ equals $(\zeta^N - 1)t(\zeta)$.
- Verify the opening of $f_{\text{batch}}$ at $\zeta$. That is, check that $\mathrm{Verify}(\left[f_{\mathrm{batch} } \right]_1, \pi_{\mathrm{batch}}, \zeta, f_{\mathrm{batch} } (\zeta))$ outputs _Accept_.
- Verify the opening of $z$ at $\zeta\omega$. That is, check the validity of the proof $\pi_{single}$ using the commitment $\left[z\right]_1$ and the value $\bar z_\omega$.
That is, check that $\mathrm{Verify}(\left[z\right]_1, \pi_{\mathrm{single}}, \zeta \omega, \bar z_\omega)$ outputs _Accept_.

If all checks pass, he outputs _Accept_. Otherwise outputs _Reject_.