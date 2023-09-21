# STARKs protocol

In this section we describe precisely the STARKs protocol used in Lambdaworks.

We begin with some additional considerations and notation for most of the relevant objects and values to refer to them later on.

### Grinding
This is a technique to increase the soundness of the protocol by adding proof of work. It works as follows. At some fixed point in the protocol, the verifier sends a challenge $x$ to the prover. The prover needs to find a string $y$ such that $H(x || y)$ begins with a predefined number of zeroes. Here $x || y$ denotes the concatenation of $x$ and $y$, seen as bit strings.
The number of zeroes is called the *grinding factor*. The hash function $H$ can be any hash function, independent of other hash functions used in the rest of the protocol. In Lambdaworks we use Keccak256.

### Transcript

The Fiat-Shamir heuristic is used to make the protocol noninteractive. We assume there is a transcript object to which values can be added and from which challenges can be sampled.

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
- $b=2^l$ is the *[blowup factor](/starks/protocol_overview.html#general-case-the-blowup-factor)*.
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

For $i\in[0,2^{n+k})$, the operation $\text{open}(A, i)$ returns $y_i$ and the authentication path $s$ to the Merkle tree root.

The operation $\text{Verify}(i,y,r,s)$ returns _Accept_ or _Reject_ depending on whether the $i$-th element of $A$ is $y$. It checks whether the authentication path $s$ is compatible with $i$, $y$ and the Merkle tree root $r$.


In our cases the sets $A$ will be of the form $A=(f(a), f(ab), f(ab^2), \dots, f(ab^L))$ for some elements $a,b\in\mathbb{F}$. It will be convenient to use the following abuse of notation. We will write $\text{Open}(A, ab^i)$ to mean $\text{Open}(A, i)$. Similarly, we will write $\text{Verify}(ab^i, y, r, s)$ instead of $\text{Verify}(i, y, r, s)$. Note that this is only notation and $\text{Verify}(ab^i, y, r, s)$ is only checking that the $y$ is the $i$-th element of the commited vector. 

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
 
- Sample $\alpha_1^B,\dots,\alpha_{m}^B$ and $\beta_1^B,\dots,\beta_{m}^B$ in $\mathbb{F}$ from the transcript.
- Sample $\alpha_1^T,\dots,\alpha_{n_T}^T$ and $\beta_1^T,\dots,\beta_{n_T}^T$ in $\mathbb{F}$ from the transcript.
- Compute $B_j := \frac{t_j - P^B_j}{Z_j^B}$.
- Compute $C_k := \frac{P^T_k(t_1, \dots, t_m, t_1(gX), \dots, t_m(gX))}{Z_k^T}$.
- Compute the _composition polynomial_
  $$H := \sum_{k} \beta_k^TC_k + \sum_j \beta_j^BB_j$$
- Decompose $H$ as
  $$H = H_1(X^2) + XH_2(X^2)$$
- Compute commitments $[H_1]$ and $[H_2]$.
- Add $[H_1]$ and $[H_2]$ to the transcript.

#### Round 3: Evaluation of polynomials at $z$
 
- Sample from the transcript until obtaining $z\in\mathbb{F}\setminus D_{\text{LDE}}$.
- Compute $H_1(z^2)$, $H_2(z^2)$, and $t_j(z)$ and $t_j(gz)$ for all $j$.
- Add $H_1(z^2)$, $H_2(z^2)$, and $t_j(z)$ and $t_j(gz)$ for all $j$ to the transcript.

#### Round 4: Run batch open protocol 
 
- Sample $\gamma$, $\gamma'$, and $\gamma_1,\dots,\gamma_m$, $\gamma_1',\dots,\gamma_m'$ in $\mathbb{F}$ from the transcript.
- Compute $p_0$ as $$\gamma\frac{H_1 - H_1(z^2)}{X - z^2} + \gamma'\frac{H_2 - H_2(z^2)}{X - z^2} + \sum_j \gamma_j\frac{t_j - t_j(z)}{X - z} + \gamma_j'\frac{t_j - t_j(gz)}{X - gz}$$

##### Round 4.1.k: FRI commit phase

- Let $D_0:=D_{\text{LDE}}$, and $[p_0]:=\text{Commit}(p_0(D_0))$.
- Add $[p_0]$ to the transcript.
- For $k=1,\dots,n$ do the following:
  - Sample $\zeta_{k-1}$ from the transcript.
  - Decompose $p_{k-1}$ into even and odd parts, that is, $p_{k-1}=p_{k-1}^{odd}(X^2)+ X p_{k-1}^{even}(X^2)$.
  - Define $p_k:= p_{k-1}^{odd}(X) + \zeta_{k-1}p_{k-1}^{even}(X)$.
  - If $k < n$:
    - Let $L$ such that $|D_{k-1}|=2L$. Define $D_{k}:=(d_1^2, \dots, d_L^2)$, where $D_{k-1}=(d_1, \dots, d_{2L})$.
    - Let $[p_k]:=\text{Commit}(p_k(D_k))$.
    - Add $[p_k]$ to the transcript.
- $p_n$ is a constant polynomial and therefore $p_n\in\mathbb{F}$. Add $p_n$ to the transcript.

##### Round 4.2: Grinding
- Sample $x$ from the transcript.
- Compute $y$ such that $\text{Keccak256}(x || y)$ has $c$ leading zeroes.
- Add $y$ to the transcript.

##### Round 4.3: FRI query phase

- For $s=0,\dots,Q-1$ do the following:
  - Sample random index $\iota_s \in [0, 2^{n+l}]$ from the transcript and let $\upsilon_s := \omega^{\iota_s}$.
  - Compute $\text{Open}(p_0(D_0), \upsilon_s)$.
  - Compute $\text{Open}(p_k(D_k), -\upsilon_s^{2^k})$ for all $k=0,\dots,n-1$.


##### Round 4.4: Open deep composition polynomial components
- For $s=0,\dots,Q-1$ do the following:
    - Compute $\text{Open}(H_1(D_{\text{LDE}}), \upsilon_s)$, $\text{Open}(H_2(D_{\text{LDE}}), \upsilon_s)$.
    - Compute $\text{Open}(t_j(D_{\text{LDE}}), \upsilon_s)$ for all $j=1,\dots, m$.

#### Build proof

- Send the proof to the verifier:
$$
\begin{align}
\Pi = ( &\\
&\{[t_j], t_j(z), t_j(gz): 0\leq j < m\}, \\
&[H_1], H_1(z^2),[H_2], H_2(z^2), \\
&\{[p_k]: 0\leq k < n\}, \\
&p_n, \\
&y, \\
&\{\text{Open}(p_0(D_0), \upsilon_s): 0\leq s < Q\}), \\
&\{\text{Open}(p_k(D_k), -\upsilon_s^{2^k}): 0\leq k< n, 0\leq s < Q\}, \\
&\{\text{Open}(H_1(D_{\text{LDE}}), \upsilon_s): 0 \leq s < Q\}, \\
&\{\text{Open}(H_2(D_{\text{LDE}}), \upsilon_s): 0 \leq s < Q\}, \\
&\{\text{Open}(t_j(D_{\text{LDE}}), \upsilon_s): 0 \leq j< m, 0 \leq s < Q\}, \\
) &
\end{align}
$$

### Verifier

#### Notation

- Bold capital letters refer to commitments. For example $\mathbf{H}_1$ is the claimed commitment $[H_1]$.
- Greek letters with superscripts refer to claimed function evaluations. For example $\tau_j^z$ is the claimed evaluation $t_j(z)$.
- Gothic letters refer to authentication paths (e.g. $\mathfrak{H}_1$ is the authentication path of the opening of $H_1$).

#### Input

$$
\begin{align}
\Pi = ( &\\
&\{\mathbf{T}_j, \tau_j^z, \tau_j^{gz}: 0\leq j < m\}, \\
&\mathbf{H}_1, \eta_1^{z^2},\mathbf{H}_2, \eta_2^{z^2}, \\
&\{\mathbf{P}_k: 0\leq k < n\}, \\
&\pi, \\
&y, \\
&\{(\pi_0^{\upsilon_s}, \mathfrak{P}_0): 0\leq s < Q\}, \\
&\{(\pi_k^{-\upsilon_s^{2^k}}, \mathfrak{P}_k): 0\leq k< n, 0\leq s < Q\}, \\
&\{(\eta_1^{\upsilon_s}, \mathfrak{H}_1): 0 \leq s < Q\}\\
&\{(\eta_2^{\upsilon_s}, \mathfrak{H}_2): 0 \leq s < Q\},\\
&\{(\tau_j^{\upsilon_s}, \mathfrak{T}_j): 0 \leq j< m, 0 \leq s < Q\}, \\
) &
\end{align}
$$

#### Step 1: Replay interactions and recover challenges

- Start a transcript
- (Strong Fiat Shamir) Commit to the set of coefficients of the transition and boundary polynomials, and add the commitments to the transcript.
- Add $\mathbf{T}_j$ to the transcript for all $j=1, \dots, m'$.
- Sample random values $a_1, \dots, a_l$ from the transcript.
- Add $\mathbf{T}_j$ to the transcript for $j=m' +1, \dots, m' + m''$.
- Sample $\alpha_1^B,\dots,\alpha_{m}^B$ and $\beta_1^B,\dots,\beta_{m}^B$ in $\mathbb{F}$ from the transcript.
- Sample $\alpha_1^T,\dots,\alpha_{n_T}^T$ and $\beta_1^T,\dots,\beta_{n_T}^T$ in $\mathbb{F}$ from the transcript.
- Add $\mathbf{H}_1$ and $\mathbf{H}_2$ to the transcript.
- Sample $z$ from the transcript.
- Add $\eta_1^{z^2}$, $\eta_2^{z^2}$, $\tau_j^z$ and $\tau_j^{gz}$ to the transcript.
- Sample $\gamma$, $\gamma'$, and $\gamma_1, \dots, \gamma_m, \gamma'_1, \dots,  \gamma'_m$ from the transcript.
- Add $\mathbf{P}_0$ to the transcript
- For $k=1, \dots, n$ do the following:
  - Sample $\zeta_{k-1}$
  - If $k < n$: add $\mathbf{P}_k$ to the transcript
- Add $\pi$ to the transcript.
- Sample $x$ from the transcript.
- Add $y$ to the transcript.
- For $s=0, \dots, Q-1$:
  - Sample random index $\iota_s \in [0, 2^{n+l}]$ from the transcript and let $\upsilon_s := \omega^{\iota_s}$.

#### Verify grinding:
Check that $\text{Keccak256}(x || y)$ has $c$ leading zeroes.


#### Step 2: Verify claimed composition polynomial

- Compute $h := \eta_1^{z^2} + z \eta_2^{z^2}$
- Compute $b_j := \frac{\tau_j^z - P^B_j(z)}{Z_j^B(z)}$
- Compute $c_k := \frac{P^T_k(\tau_1^z, \dots, \tau_m^z, \tau_1^{gz}, \dots, \tau_m^{gz})}{Z_k^T(z)}$
- Verify
  $$h = \sum_{k} \beta_k^Tc_k + \sum_j \beta_j^Bb_j$$

#### Step 3: Verify FRI

- Check that the following are all _Accept_:
  - $\text{Verify}((\upsilon_s, \pi_0^{\upsilon_s}), \mathbf{P}_0, \mathfrak{P}_0)$ for all $0\leq s < Q$.
  - $\text{Verify}((-\upsilon_s^{2^k}, \pi_k^{-\upsilon_s^{2^k}}), \mathbf{P}_k, \mathfrak{P}_k)$ for all $0\leq k < n$, $0\leq s < Q$.
- For all $s=0,\dots,Q-1$: - For all $k=0,\dots,n-1$: - Solve the following system of equations on the variables $G, H$
  $$
  \begin{aligned}
  \pi_k^{\upsilon_s^{2^{k}}} &= G + \upsilon_s^{2^k}H \\
  \pi_k^{-\upsilon_s^{2^{k}}} &= G - \upsilon_s^{2^k}H
  \end{aligned}
  $$
          - Define $\pi_{k+1}^{\upsilon_s^{2^{k+1}}}:=G + \zeta_{k}H$
      - Check that $\pi_n^{\upsilon_s^{2^n}}$ is equal to $\pi$.

#### Step 4: Verify deep composition polynomial is FRI first layer

- For $s=0,\dots,Q-1$ do the following:
    - Check that the following are all _Accept_:
        - $\text{Verify}((\upsilon_s, \eta_1^{\upsilon_s}), \mathbf{H}_1, \mathfrak{h}_1)$.
        - $\text{Verify}((\upsilon_s, \eta_2^{\upsilon_s}), \mathbf{H}_2, \mathfrak{h}_2)$.
        - $\text{Verify}((\upsilon_s, \tau_j^{\upsilon_s}), \mathbf{T}_j, \mathfrak{T}_j)$ for all $0\leq j < m$.
    - Check that $\pi_0^{\upsilon_s}$ is equal to the following:
        $$
        \gamma\frac{\eta_1^{\upsilon_s} - \eta_1^{z^2}}{\upsilon_s - z^2} + \gamma'\frac{\eta_2^{\upsilon_s} - \eta_2^{z^2}}{\upsilon_s - z^2} + \sum_j \gamma_j\frac{\tau_j^{\upsilon_s} - \tau_j^{z}}{\upsilon_s - z} + \gamma_j'\frac{\tau_j^{\upsilon_s} - \tau_j^{gz}}{\upsilon_s - gz}
        $$

# Other

## Notes on Optimizations
- Inversions of finite field elements are slow. There is a very well known trick to batch invert many elements at once replacing inversions by multiplications. See [here](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse#Multiple_inverses) for the algorithm.
- One of the most computationally intensive operations performed is polynomial division. These can be optimized by utilizing [Fast Fourier Transform](http://web.cecs.pdx.edu/~maier/cs584/Lectures/lect07b-11-MG.pdf) (FFT) to divide each field element in Lagrange form.
- In specific scenarios, such as dividing by a polynomial of the form (x-a), [Ruffini's rule](https://en.wikipedia.org/wiki/Ruffini%27s_rule) can be employed to further enhance performance.

