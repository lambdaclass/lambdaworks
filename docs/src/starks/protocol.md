# STARKs protocol

In this section we describe precisely the STARKs protocol used in Lambdaworks.

We begin with some additional considerations and notation for most of the relevant objects and values to refer to them later on.

### Transcript

The Fiat-Shamir heuristic is used to make the protocol noninteractive. We assume there is a transcript object to which values can be added and from which challenges can be sampled.

### Grinding
This is a technique to increase the soundness of the protocol by adding proof of work. It works as follows. At some fixed point in the protocol, a value $x$ is derived in a deterministic way from all the interactions between the prover and the verifier up to that point (the state of the transcript). The prover needs to find a string $y$ such that $H(x || y)$ begins with a predefined number of zeroes. Here $x || y$ denotes the concatenation of $x$ and $y$, seen as bit strings.
The number of zeroes is called the *grinding factor*. The hash function $H$ can be any hash function, independent of other hash functions used in the rest of the protocol. In Lambdaworks we use Keccak256.


## General notation

- $\mathbb{F}$ denotes a finite field.
- Given a vector $D=(y_1,\dots,y_L)$ and a function $f:\text{set}(D) \to \mathbb{F}$, denote by $f(D)$ the vector $(f(y_1),\dots,f(y_L))$. Here $\text{set}(D)$ denotes the underlying set of $A$.
- A polynomial $p \in \mathbb{F}[X]$ induces a function $f:A \to \mathbb{F}$ for every subset $A$ of $\mathbb{F}$, where $f(a) := p(a)$.
- Let $p, q \in \mathbb{F}[X]$ be two polynomials. A function $f: A \to \mathbb{F}$ can be induced from them for every subset $A$ disjoint from the set of roots of $q$, defined by $f(a) := p(a) q(a)^{-1}$. We abuse notation and denote $f$ by $p/q$.

## Definitions 

We assume the prover has already obtained the trace of the execution of the program. This is a matrix $T$ with entries in a finite field $\mathbb{F}$. We assume the number of rows of $T$ is $2^n$ for some $n$ in $\mathbb{N}$.

#### Values known by the prover and verifier prior to the interactions

These values are determined the program,  the specifications of the AIR being used and the security parameters chosen.

- $m'$ is the number of columns of the trace matrix $T$.
- $r$ the number of RAP challenges.
- $m''$ is the number of extended columns of the trace matrix in the (optional) second round of RAP.
- $m$ is the total number of columns: $m := m' + m''$.
- $P_k^T$ denote the transition constraint polynomials for $k=1,\dots,n_T$. We are assuming these are of degree at most 2.
- $Z_j^T$ denote the transition constraint zerofiers for $k=1,\dots,n_T$.
- $b=2^l$ is the *[blowup factor](/starks/protocol_overview.html#fri)*.
- $c$ is the *grinding factor*.
- $Q$ is number of FRI queries.
- We assume there is a fixed hash function from $\mathbb{F}$ to binary strings. We also assume all Merkle trees are constructed using this hash function.

#### Values computed by the prover
These values are computed by the prover from the execution trace and are sent to the verifier along with the proof.
- $2^n$ is the number of rows of the trace matrix after RAP.
- $\omega$ a primitive $2^{n+l}$-th root of unity.
- $g = \omega^{b}$.
- An element $h\in\mathbb{F} \setminus \{\omega^i\}_{i \geq 0}$. This is called the *coset factor*.
- Boundary constraints polynomials $P_j^B$ for $j=1,\dots,m$.
- Boundary constraint zerofiers $Z_j^B$ for $j=1,\dots,m$..

#### Derived values
Both prover and verifier compute the following.

- The interpolation domain: the vector $D_S=(1, g, \dots, g^{2^n-1})$.
- The Low Degree Extension $D_{\text{LDE}} =(h, h\omega, h\omega^2,\dots, h\omega^{2^{n+l} - 1})$. Recall $2^l$ is the blowup factor.

### Notation of important operations
#### Vector commitment scheme

Given a vector $A=(y_0, \dots, y_L)$. The operation $\text{Commit}(A)$ returns the root $r$ of the Merkle tree that has the hash of the elements of $A$ as leaves.

For $i\in[0,2^{n+k})$, the operation $\text{Open}(A, i)$ returns the pair $(y_i, s)$, where $s$ is the authentication path to the Merkle tree root.

The operation $\text{Verify}(i,y,r,s)$ returns _Accept_ or _Reject_ depending on whether the $i$-th element of $A$ is $y$. It checks whether the authentication path $s$ is compatible with $i$, $y$ and the Merkle tree root $r$.


In our cases the sets $A$ will be of the form $A=(f(a), f(ab), f(ab^2), \dots, f(ab^L))$ for some elements $a,b\in\mathbb{F}$. It will be convenient to use the following abuse of notation. We will write $\text{Open}(A, ab^i)$ to mean $\text{Open}(A, i)$. Similarly, we will write $\text{Verify}(ab^i, y, r, s)$ instead of $\text{Verify}(i, y, r, s)$. Note that this is only notation and $\text{Verify}(ab^i, y, r, s)$ is only checking that the $y$ is the $i$-th element of the commited vector. 

##### Batch
As we mentioned in the [protocol overview](protocol_overview.html#batch). When committing to multiple vectors $A_1, \dots, A_k$, where $A_i = (y_0^{(i), \dots, y_L^{(i)}})$ one can build a single Merkle tree. Its $j$-th leaf is the concatenation of all the $j$-th coordinates of all vectors, that is, $(y_j^{(1)}||\dots||y_j^{(k)})$. The commitment to this batch of vectors is the root of this Merkle tree.

## Protocol

### Prover

#### Round 0: Transcript initialization

- Start a new transcript.
- (Strong Fiat Shamir) Add to it all the public values.

#### Round 1: Arithmetization and commitment of the execution trace
 
##### Round 1.1: Commit main trace

- For each column $M_j$ of the execution trace matrix $T$, interpolate its values at the domain $D_S$ and obtain polynomials $t_j$ such that $t_j(g^i)=T_{i,j}$.
- Compute $[t_j] := \text{Commit}(t_j(D_{\text{LED}}))$ for all $j=1,\dots,m'$ (*Batch commitment optimization applies here*).
- Add $[t_j]$ to the transcript in increasing order.

##### Round 1.2: Commit extended trace

- Sample random values $a_1,\dots,a_l$ in $\mathbb{F}$ from the transcript.
- Use $a_1,\dots,a_l$ to build $M_{\text{RAP2}}\in\mathbb{F}^{2^n\times m''}$ following the specifications of the RAP process.
- For each column $\hat M_j$ of the matrix $M_{\text{RAP2}}$, interpolate its values at the domain $D_S$ and obtain polynomials $t_{m'+1}, \dots, t_{m' + m''}$ such that $t_j(g^i)=\hat M_{i,j}$.
- Compute $[t_j] := \text{Commit}(t_j(D_{\text{LED}}))$ for all $j=m'+1,\dots,m'+m''$ (*Batch commitment optimization applies here*).
- Add $[t_j]$ to the transcript in increasing order for all $j=m'+1,\dots,m'+m''$.

#### Round 2: Construction of composition polynomial $H$
 
- Sample $\beta_1^B,\dots,\beta_{m}^B$ in $\mathbb{F}$ from the transcript.
- Sample $\beta_1^T,\dots,\beta_{n_T}^T$ in $\mathbb{F}$ from the transcript.
- Compute $B_j := \frac{t_j - P^B_j}{Z_j^B}$.
- Compute $C_k := \frac{P^T_k(t_1, \dots, t_m, t_1(gX), \dots, t_m(gX))}{Z_k^T}$.
- Compute the _composition polynomial_
  $$H := \sum_{k} \beta_k^TC_k + \sum_j \beta_j^BB_j$$
- Decompose $H$ as
  $$H = H_1(X^2) + XH_2(X^2)$$
- Compute commitments $[H_1]$ and $[H_2]$ (*Batch commitment optimization applies here*).
- Add $[H_1]$ and $[H_2]$ to the transcript.

#### Round 3: Evaluation of polynomials at $z$
 
- Sample from the transcript until obtaining $z\in\mathbb{F}\setminus D_{\text{LDE}}$.
- Compute $H_1(z^2)$, $H_2(z^2)$, and $t_j(z)$ and $t_j(gz)$ for all $j$.
- Add $H_1(z^2)$, $H_2(z^2)$, and $t_j(z)$ and $t_j(gz)$ for all $j$ to the transcript.

#### Round 4: Run batch open protocol 
 
- Sample $\gamma$, $\gamma'$, and $\gamma_1,\dots,\gamma_m$, $\gamma_1',\dots,\gamma_m'$ in $\mathbb{F}$ from the transcript.
- Compute $p_0$ as $$\gamma\frac{H_1 - H_1(z^2)}{X - z^2} + \gamma'\frac{H_2 - H_2(z^2)}{X - z^2} + \sum_j \gamma_j\frac{t_j - t_j(z)}{X - z} + \gamma_j'\frac{t_j - t_j(gz)}{X - gz}$$

##### Round 4.1.k: FRI commit phase

- Let $D_0:=D_{\text{LDE}}$.
- For $k=1,\dots,n$ do the following:
  - Sample $\zeta_{k-1}$ from the transcript.
  - Decompose $p_{k-1}$ into even and odd parts, that is, $p_{k-1}=p_{k-1}^{odd}(X^2)+ X p_{k-1}^{even}(X^2)$.
  - Define $p_k:= p_{k-1}^{odd}(X) + \zeta_{k-1}p_{k-1}^{even}(X)$.
  - If $k < n$:
    - Let $L = |D_{k-1}|/2$. Define $D_{k}:=(d_0^2, \dots, d_{L-1}^2)$, where $D_{k-1}=(d_0, \dots, d_{2L-1})$.
    - Let $[p_k]:=\text{Commit}(p_k(D_k))$.
    - Add $[p_k]$ to the transcript.
- $p_n$ is a constant polynomial and therefore $p_n\in\mathbb{F}$. Add $p_n$ to the transcript.

##### Round 4.2: Grinding
- Let $x$ be the internal state of the transcript.
- Compute $y$ such that $\text{Keccak256}(x || y)$ has $c$ leading zeroes.
- Add $y$ to the transcript.

##### Round 4.3: FRI query phase

- For $s=0,\dots,Q-1$ do the following:
  - Sample random index $\iota_s \in [0, 2^{n+l-1}]$ from the transcript and let $\upsilon_s := h\omega^{\iota_s}$.
  - Compute $\text{Open}(t_j(D_{\text{LDE}}), \upsilon_s)$ and $\text{Open}(t_j(D_{\text{LDE}}), -\upsilon_s)$ for all $j=1,\dots, m$.
  - Compute $\text{Open}(H_1(D_{\text{LDE}}), \upsilon_s)$ and $\text{Open}(H_1(D_{\text{LDE}}), -\upsilon_s)$.
  - Compute $\text{Open}(H_2(D_{\text{LDE}}), \upsilon_s)$ and $\text{Open}(H_2(D_{\text{LDE}}), -\upsilon_s)$.
  - Compute $\text{Open}(p_k(D_k), \upsilon_s^{2^k})$ and $\text{Open}(p_k(D_k), -\upsilon_s^{2^k})$ for all $k=1,\dots,n-1$.

#### Build proof

- Send the proof to the verifier:
$$
\begin{align}
\Pi = ( &\\
&\{[t_j], t_j(z), t_j(gz): 0\leq j < m\}, \\
&[H_1], H_1(z^2),[H_2], H_2(z^2), \\
&\{[p_k]: 1\leq k < n\}, \\
&p_n, \\
&y, \\
&\{\text{Open}(t_j(D_{\text{LDE}}), \upsilon_s): 0 \leq j< m, 0 \leq s < Q\}, \\
&\{\text{Open}(H_1(D_{\text{LDE}}), \upsilon_s): 0 \leq s < Q\}, \\
&\{\text{Open}(H_2(D_{\text{LDE}}), \upsilon_s): 0 \leq s < Q\}, \\
&\{\text{Open}(p_k(D_k), \upsilon_s^{2^k}): 1\leq k< n, 0\leq s < Q\}, \\
&\{\text{Open}(t_j(D_{\text{LDE}}), -\upsilon_s): 0 \leq j< m, 0 \leq s < Q\}, \\
&\{\text{Open}(H_1(D_{\text{LDE}}), -\upsilon_s): 0 \leq s < Q\}, \\
&\{\text{Open}(H_2(D_{\text{LDE}}), -\upsilon_s): 0 \leq s < Q\}, \\
&\{\text{Open}(p_k(D_k), -\upsilon_s^{2^k}): 1\leq k< n, 0\leq s < Q\}, \\
) &
\end{align}
$$

### Verifier
From the point of view of the verifier, the proof they receive is a bunch of values that may or may not be what they claim to be. To make this explicit, we avoid denoting values like $t_j(z)$ as such, because that implicitly assumes that the value was obtained after evaluating a polynomial $t_j$ at $z$. And that's something the verifier can't assume. We use the following convention.

- Bold capital letters refer to commitments. For example $\mathbf{T}_j$ is the claimed commitment $[t_j]$.
- Greek letters with superscripts refer to claimed function evaluations. For example $\tau_j^z$ is the claimed evaluation $t_j(z)$ and $\tau_j^{gz}$ is the claimed evaluation of $t_j(gz)$. Note that field elements in superscripts never indicate powers. They are just notation.
- Gothic letters refer to authentication paths. For example $\mathfrak{T}_j$ is the authentication path of a opening of $t_j$.
- Recall that every opening $\text{Open}(A, i)$ is a pair $(y, s)$, where $y$ is the claimed value at index $i$ and $s$ is the authentication path. So for example, $\text{Open}(t_j(D_{\text{LDE}}), \upsilon_s)$ is denoted as $(\tau_j^{\upsilon_s}, \mathfrak{T}_j)$ from the verifier's end.

#### Input
This is the proof using the notation described above. The elements appear in the same exact order as they are in the [Prover](#build-proof) section, serving also as a complete reference of the meaning of each value.

$$
\begin{align}
\Pi = ( &\\
&\{\mathbf{T}_j, \tau_j^z, \tau_j^{gz}: 0\leq j < m\}, \\
&\mathbf{H}_1, \eta_1^{z^2},\mathbf{H}_2, \eta_2^{z^2}, \\
&\{\mathbf{P}_k: 1\leq k < n\}, \\
&\pi, \\
&y, \\
&\{(\tau_j^{\upsilon_s}, \mathfrak{T}_j): 0 \leq j< m, 0 \leq s < Q\}, \\
&\{(\eta_1^{\upsilon_s}, \mathfrak{H}_1): 0 \leq s < Q\}\\
&\{(\eta_2^{\upsilon_s}, \mathfrak{H}_2): 0 \leq s < Q\},\\
&\{(\pi_k^{\upsilon_s^{2^k}}, \mathfrak{P}_k): 1\leq k< n, 0\leq s < Q\}, \\
&\{(\tau_j^{-\upsilon_s}, \mathfrak{T}_j'): 0 \leq j< m, 0 \leq s < Q\}, \\
&\{(\eta_1^{-\upsilon_s}, \mathfrak{H}_1'): 0 \leq s < Q\}\\
&\{(\eta_2^{-\upsilon_s}, \mathfrak{H}_2'): 0 \leq s < Q\},\\
&\{(\pi_k^{-\upsilon_s^{2^k}}, \mathfrak{P}_k'): 1\leq k< n, 0\leq s < Q\}, \\
) &
\end{align}
$$

#### Step 1: Replay interactions and recover challenges

- Start a transcript
- (Strong Fiat Shamir) Add all public values to the transcript.
- Add $\mathbf{T}_j$ to the transcript for all $j=1, \dots, m'$.
- Sample random values $a_1, \dots, a_l$ from the transcript.
- Add $\mathbf{T}_j$ to the transcript for $j=m' +1, \dots, m' + m''$.
- Sample $\alpha_1^B,\dots,\alpha_{m}^B$ and $\beta_1^B,\dots,\beta_{m}^B$ in $\mathbb{F}$ from the transcript.
- Sample $\alpha_1^T,\dots,\alpha_{n_T}^T$ and $\beta_1^T,\dots,\beta_{n_T}^T$ in $\mathbb{F}$ from the transcript.
- Add $\mathbf{H}_1$ and $\mathbf{H}_2$ to the transcript.
- Sample $z$ from the transcript.
- Add $\eta_1^{z^2}$, $\eta_2^{z^2}$, $\tau_j^z$ and $\tau_j^{gz}$ to the transcript.
- Sample $\gamma$, $\gamma'$, and $\gamma_1, \dots, \gamma_m, \gamma'_1, \dots,  \gamma'_m$ from the transcript.
- For $k=1, \dots, n$ do the following:
  - Sample $\zeta_{k-1}$
  - If $k < n$: add $\mathbf{P}_k$ to the transcript
- Add $\pi$ to the transcript.
- Add $y$ to the transcript.
- For $s=0, \dots, Q-1$:
  - Sample random index $\iota_s \in [0, 2^{n+l-1}]$ from the transcript and let $\upsilon_s := h\omega^{\iota_s}$.

#### Verify grinding:
Check that $\text{Keccak256}(x || y)$ has $c$ leading zeroes.


#### Step 2: Verify claimed composition polynomial

- Compute $h := \eta_1^{z^2} + z \eta_2^{z^2}$
- Compute $b_j := \frac{\tau_j^z - P^B_j(z)}{Z_j^B(z)}$
- Compute $c_k := \frac{P^T_k(\tau_1^z, \dots, \tau_m^z, \tau_1^{gz}, \dots, \tau_m^{gz})}{Z_k^T(z)}$
- Verify
  $$h = \sum_{k} \beta_k^Tc_k + \sum_j \beta_j^Bb_j$$

#### Step 3: Verify FRI

- Reconstruct the deep composition polynomial values at $\upsilon_s$ and $-\upsilon_s$. That is, define
        $$\begin{align}\pi_0^{\upsilon_s}&:=
        \gamma\frac{\eta_1^{\upsilon_s} - \eta_1^{z^2}}{\upsilon_s - z^2} + \gamma'\frac{\eta_2^{\upsilon_s} - \eta_2^{z^2}}{\upsilon_s - z^2} + \sum_j \gamma_j\frac{\tau_j^{\upsilon_s} - \tau_j^{z}}{\upsilon_s - z} + \gamma_j'\frac{\tau_j^{\upsilon_s} - \tau_j^{gz}}{\upsilon_s - gz}, \\
        \pi_0^{-\upsilon_s}&:=
        \gamma\frac{\eta_1^{-\upsilon_s} - \eta_1^{z^2}}{-\upsilon_s - z^2} + \gamma'\frac{\eta_2^{-\upsilon_s} - \eta_2^{z^2}}{-\upsilon_s - z^2} + \sum_j \gamma_j\frac{\tau_j^{-\upsilon_s} - \tau_j^{z}}{-\upsilon_s - z} + \gamma_j'\frac{\tau_j^{-\upsilon_s} - \tau_j^{gz}}{-\upsilon_s - gz}.
        \end{align}
        $$
- For all $s=0,\dots,Q-1$:
  - For all $k=0,\dots,n-1$:
    - Check that $\text{Verify}((\upsilon_s^{2^k}, \pi_k^{-\upsilon_s^{2^k}}), \mathbf{P}_k, \mathfrak{P}_k)$ and $\text{Verify}((-\upsilon_s^{2^k}, \pi_k^{-\upsilon_s^{2^k}}), \mathbf{P}_k, \mathfrak{P}_k')$ are _Accept_.
    - Solve the following system of equations on the variables $G, H$
  $$
  \begin{aligned}
  \pi_k^{\upsilon_s^{2^{k}}} &= G + \upsilon_s^{2^k}H \\
  \pi_k^{-\upsilon_s^{2^{k}}} &= G - \upsilon_s^{2^k}H
  \end{aligned}
  $$
    - If $k < n - 1$, check that $\pi_{k+1}^{\upsilon_s^{2^{k+1}}}$ equals $G + \zeta_{k}H$
    - If $k = n$, check that $\pi$ equals $G + \zeta_{k}H$.
 
#### Step 4: Verify trace and composition polynomials openings

- For $s=0,\dots,Q-1$ do the following:
    - Check that the following are all _Accept_:
        - $\text{Verify}((\upsilon_s, \tau_j^{\upsilon_s}), \mathbf{T}_j, \mathfrak{T}_j)$ for all $0\leq j < m$.
        - $\text{Verify}((\upsilon_s, \eta_1^{\upsilon_s}), \mathbf{H}_1, \mathfrak{h}_1)$.
        - $\text{Verify}((\upsilon_s, \eta_2^{\upsilon_s}), \mathbf{H}_2, \mathfrak{h}_2)$.
        - $\text{Verify}((-\upsilon_s, \tau_j^{\upsilon_s}), \mathbf{T}_j, \mathfrak{T}_j')$ for all $0\leq j < m$.
        - $\text{Verify}((-\upsilon_s, \eta_1^{\upsilon_s}), \mathbf{H}_1, \mathfrak{h}_1')$.
        - $\text{Verify}((-\upsilon_s, \eta_2^{\upsilon_s}), \mathbf{H}_2, \mathfrak{h}_2')$.


## Notes on Optimizations and variants
### Sampling of challenges variant
To build the composition the prover samples challenges $\beta_k^T$ and $\beta_j^B$ for $k = 1,\dots,n_T$ and $j=1,\dots,m$. A variant of this is sampling a single challenge $\beta$ and defining $\beta_k^T$ and $\beta_j^B$ as powers of $\beta$. That is, define $\beta_k^T := \beta^{k-1}$ for $k=1,\dots,n_T$ and $\beta_j^B := \beta^{j + n_T - 1}$ for $j =1, \dots, m$.

The same variant applies for the challenges $\gamma, \gamma', \gamma_j, \gamma_j'$ for $j = 1, \dots, m$ used to build the deep composition polynomial. In this case the variant samples a single challenge $\alpha$ and defines $\gamma_j := \alpha^j$, $\gamma_j' := \alpha^{j + m - 1}$ for all $j=1,\dots,m$, and $\gamma := \alpha^{2m}, \gamma' := \alpha^{2m+1}$.

### Batch inversion
Inversions of finite field elements are slow. There is a very well known trick to batch invert many elements at once replacing inversions by multiplications. See [here](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Multiple_inverses) for the algorithm.

### FFT
One of the most computationally intensive operations performed is polynomial division. These can be optimized by utilizing [Fast Fourier Transform](http://web.cecs.pdx.edu/~maier/cs584/Lectures/lect07b-11-MG.pdf) (FFT) to divide each field element in Lagrange form.

### Ruffini's rule
In specific scenarios, such as dividing by a polynomial of the form $X-a$, for example when building the deep composition polynomial, [Ruffini's rule](https://en.wikipedia.org/wiki/Ruffini%27s_rule) can be employed to further enhance performance.

### Bit-reversal ordering of Merkle tree leaves
As one can see from inspecting the protocol, there are multiple times where, for a polynomial $p$, the prover sends both openings $\text{Open}(p(D), h\omega^i)$ and $\text{Open}(p(D), -h\omega^i)$. This implies, a priori, sending two authentication paths. Domains can be indexed using bit-reverse ordering to reduce this to a single authentication path for both openings, as follows.

The natural way of building a Merkle tree to commit to a vector $(p(h), p(h\omega), p(h\omega^2), \dots, p(h\omega^{2^k-1}))$, is assigning the value $p(h\omega^i)$ to leaf $i$. If this is the case, the value $p(h\omega^i)$ is at position $i$ and the value $p(-h\omega^i)$ is at position $i + 2^{k-1}$. This is because $-1$ equals $\omega{2^{k-1}}$ for the value $\omega$ used in the protocol.

Instead of this naive approach, a better solution is to assign the value $p(h\omega^{\sigma(i)})$ to leaf $i$, where $\sigma$ is the bit-reversal permutation. This is the permutation that maps $i$ to the index whose binary representation (padded to $k$ bits), is the binary representation of $i$ but in reverse order. For example, if $k=3$ and $i=1$, then its binary representation is $001$, which reversed is $100$. Therefore $\sigma(1) = 8$. In the same way $\sigma(0) = 0$ and $\sigma(2) = 4$. Check out the [wikipedia](https://en.wikipedia.org/wiki/Bit-reversal_permutation) article. With this ordering of the leaves, if $i$ is even, element $p(h\omega^{\sigma(i)})$ is at index $i$ and $p(-h\omega^{\sigma(i)})$ is at index $i + 1$. Which means that a single authentication path serves to validate both points simultaneously.

### Redundant values in the proof
The prover opens the polynomials $p_k$ of the FRI layers at $\upsilon_s^{2^k}$ and $-\upsilon_s^{2^k}$ for all $k>1$. Later on, the verifier uses each of those pairs to reconstruct one of the values of the next layer, namely $p_{k+1}(\upsilon^{2^{k+1}})$. So there's no need to add the value $p_k(\upsilon^{2^{k+1}})$ to the proof, as the verifier reconstructs them. The prover only needs to send the authentication paths $\mathfrak{P}_k$ for them.

The protocol is only modified at Step 3 of the verifier as follows. Checking that $\text{Verify}((\upsilon_s^{2^k}, \pi_k^{\upsilon_s^{2^k}}), \mathbf{P}_k, \mathfrak{P}_k)$ is skipped. After computing $x := G + \zeta_{k}H$, the verifier uses it to check that $\text{Verify}((\upsilon_s^{2^k}, x), \mathbf{P}_k, \mathfrak{P}_k)$ is _Accept_, which proves that $x$ is actually $\pi_k^{\upsilon_s^{2^k}}$, and continues to the next iteration of the loop.
