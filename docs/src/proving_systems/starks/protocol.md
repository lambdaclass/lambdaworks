# STARKs protocol
In this section we describe precisely the STARKs protocol used in Lambdaworks.

### Definitions of values known to both prover and verifier
- $m'$ is the number of columns of the trace matrix in first round of RAP.
- $m''$ is the number of columns of the trace matrix in the second round of RAP.
- $m:=m' + m''$.
- $2^n$ is the number of rows of the trace matrix after RAP.
- Boundary constraints polynomials $P_j^B$ for $j=1,\dots,m$.
- Boundary constraint zerofiers $Z_j^B$ for $j=1,\dots,m$..
- Transition constraint polynomials $P_k^T$ for $k=1,\dots,n_T$ of degree at most 3.
- Transition constraint zerofiers $Z_j^T$ for $k=1,\dots,n_T$.
- $b=2^l$ the blowup factor
- $\omega$ a primitive $2^{n+l}$-th root of unity.
- $g = \omega^{2^l}$.
- The interpolation domain is the vector $D_S=(1, g, \dots, g^{2^n-1})$.
- $h$ the coset factor.
- $Q$ number of FRI queries.
- Low Degree Extension is the vector $D_{\text{LDE}} =(h, h\omega, h\omega^2,\dots, h\omega^{2^{n+l} - 1})$.
- Let $d_k^T := 2^n (\deg(P_k^T) - 1)$ and let $d^B := 2^n$. Let $d := 2^{n + 1}$. Notice that $d^B \leq d$ and $d_k^T \leq d$ for all $k$. This holds because we assume all transition constraint polynomials are at most cubic.
- We assume we have a hash function from $\mathbb{F}$ to binary strings of fixed length.

### Notation
- Given a vector $A=(y_1,\dots,y_L)$ and a function $f:\text{set}(A) \to \mathbb{F}$, denote by $f(A)$ the vector $(f(y_1),\dots,f(y_L))$. Here $\text{set}(A)$ denotes the underlying set of $A$.
- A polynomial $p \in \mathbb{F}[X]$ induces a function $f:A \to \mathbb{F}$ for every subset $A$ of $\mathbb{F}$, where $f(a) := p(a)$.
- Two polynomials $p, q \in \mathbb{F}[X]$ induce a function $f: A \to \mathbb{F}$ for every subset $A$ disjoint from the set of roots of $q$, where $f(a) := p(a) q(a)^{-1}$. Denote $f$ by $p/q$.

### Randomized AIR with Preprocessing (RAP)
This the process in which the prover uses randomness from the verifier to complete the program trace with additional columns. This is specific to each RAP. See [here](https://hackmd.io/@aztec-network/plonk-arithmetiization-air) for more details.

#### Cairo's RAP
The execution of a Cairo program produces a memory vector $V$ and a matrix $M$ of size $L \times 3$ with the evolution of the three registers `pc`, `ap`, `fp`. All of them with entries in $\mathbb{F}$.

##### First round of RAP:
1. Augment each row of $M$ with information about the pointed instruction as follows: For each entry $(\text{pc}_i, \text{ap}_i, \text{fp}_i)$ of $M$, unpack the $\text{pc}_i$-th value of $V$. The result is a new matrix $M \in \mathbb{F}^{L\times 33}$ with the following layout
```
 A.  flags     (16) : Decoded instruction flags
 B.  res       (1)  : Res value
 C.  pointers  (2)  : Temporary memory pointers (ap and fp)
 D.  mem_a     (4)  : Memory addresses (pc, dst_addr, op0_addr, op1_addr)
 E.  mem_v     (4)  : Memory values (inst, dst, op0, op1)
 F.  offsets   (3)  : (off_dst, off_op0, off_op1)
 G.  derived   (3)  : (t0, t1, mul)

 A                B C  D    E    F   G
|xxxxxxxxxxxxxxxx|x|xx|xxxx|xxxx|xxx|xxx|
```
2. Let $R$ be the last row of $M$, and let $R'$ be the vector that's equal to $R$ except that it has zeroes in entries corresponding to the set of columns `mem_a` and `mem_v`. Let $L_{\text{pub}}$ be the length of the public input (program code). Extend $M$ with additional $L':=\lceil L_{\text{pub}}/4 \rceil$ rows to obtain a matrix $M \in \mathbb{F}^{(L + L')\times 33}$ by appending copies of $R'$ at the bottom (the notation $\lceil x \rceil$ means the _ceiling function_, defined as the smallest integer that is not smaller than $x$).
3. Let $r_\text{min}$ and $r_\text{max}$ be respectively the minimum and maximum values of the entries of the submatrix $M_\text{offsets}$ defined by the columns of the group `offsets`. Let $v$ be the vector of all the values between $r_\text{min}$ and $r_\text{max}$ that are not in $M_\text{offsets}$. If the length of $v$ is not a multiple of three, extend it to the nearest multiple of three using one arbitrary value of $v$.

4. Let $L_v$ be the length of $v$. Add $L_v$ rows repeating the last row of $M$ in all the columns except in the group `offsets`. In that section place the values of $v$. The result is a matrix $M$ in $\mathbb{F}^{(L + L' + L_v) \times 33}$

5. Pad $M$ with copies of its last row until it has a power of two number of rows. As a result we obtain a matrix $M_{\text{RAP1}}\in\mathbb{F}^{2^n\times 33}$.
##### Second round of RAP:

The verifier sends challenges $\alpha, z \in \mathbb{F}$ (or the prover samples them from the transcript). Additional columns are added to incorporate the memory constraints. To define them the prover follows these steps:
1. Stack the rows of the submatrix of $M_{\text{RAP1}}$ defined by the columns `pc, dst_addr, op0_addr, op1_addr` into a vector `a` of length $2^{n+2}$ (this means that the first entries of `a` are `pc[0], dst_addr[0], op0_addr[0], op1_addr[0], pc[1], dst_addr[1],...`).
2. Stack the the rows of the submatrix defined by the columns `inst, dst, op0, op1` into a vector `v` of length $2^{n+2}$.
3. Define $M_{\text{Mem}}\in\mathbb{F}^{2^{n+2}\times 2}$ to be the matrix with columns $a$, $v$.
4. Define $M_{\text{MemRepl}}\in\mathbb{F}^{2^{n+2}\times 2}$ to be the matrix that's equal to $M_{\text{Mem}}$ in the first $2^{n+2} - L_{\text{pub}}$ rows, and its last $L_{\text{pub}}$ entries are the addresses and values of the actual public memory (program code).
5. Sort $M_{\text{MemRepl}}$ by the first column in increasing order. The result is a matrix $M_{\text{MemReplSorted}}$ of size $2^{n+2}\times 2$. Denote its columns by $a'$ and $v'$.
6. Compute the vector $p$ of size $2^{n+2}$ with entries 
$$ p_i := \prod_{j=0}^i\frac{z - (a_i' + \alpha v_i')}{z - (a_i + \alpha v_i)}$$
7. Reshape the matrix $M_{\text{MemReplSorted}}$ into a $2^n\times 8$ in row-major. Reshape the vector $p$ into a $2^n \times 4$ matrix in row-major.
8. Concatenate these 12 rows. The result is a matrix $M_\text{MemRAP2}$ of size $2^n \times 12$

The verifier sends challenge $z' \in \mathbb{F}$. Further columns are added to incorporate the range check constraints following these steps:

9. Stack the rows of the submatrix of $M_\text{RAP1}$ defined by the columns in the group `offsets` into a vector $b$ of length $3\cdot 2^n$.
10. Sort the values of $b$ in increasing order. Let $b'$ be the result.
11. Compute the vector $p'$ of size $3\cdot 2^n$ with entries
$$ p_i' := \prod_{j=0}^i\frac{z' - b_i'}{z' - b_i}$$
12. Reshape $b'$ and $p'$ into matrices of size $2^n \times 3$ each and concatenate them into a matrix $M_\text{RangeCheckRAP2}$ of size $2^n \times 6$.
13. Concatenate $M_\text{MemRAP2}$ and $M_\text{RangeCheckRAP2}$ into a matrix $M_\text{RAP2}$ of size $2^n \times 18$.


Using the notation described at the beginning, $m'=33$, $m''=18$ and $m=52$. They are respectively the columns of the first and second part of the rap, and the total number of columns.


Putting all together, the final layout of the trace is the following

```
 A.  flags      (16) : Decoded instruction flags
 B.  res        (1)  : Res value
 C.  pointers   (2)  : Temporary memory pointers (ap and fp)
 D.  mem_a      (4)  : Memory addresses (pc, dst_addr, op0_addr, op1_addr)
 E.  mem_v      (4)  : Memory values (inst, dst, op0, op1)
 F.  offsets    (3)  : (off_dst, off_op0, off_op1)
 G.  derived    (3)  : (t0, t1, mul)
 H.  mem_a'     (4)  : Sorted memory addresses
 I.  mem_v'     (4)  : Sorted memory values
 J.  mem_p      (4)  : Memory permutation argument columns
 K.  offsets_b' (3)  : Sorted offset columns
 L.  offsets_p' (3)  : Range check permutation argument columns

 A                B C  D    E    F   G   H    I    J    K   L
|xxxxxxxxxxxxxxxx|x|xx|xxxx|xxxx|xxx|xxx|xxxx|xxxx|xxxx|xxx|xxx|
```

### Vector commitment scheme
Given a vector $A=(y_0, \dots, y_L)$. The operation $\text{Commit}(A)$ returns the root $r$ of the Merkle tree that has the hash of the elements of $A$ as leaves.

For $i\in[0,2^{n+k})$, the operation $\text{open}(A, i)$ returns $y_i$ and the authentication path $s$ to the Merkle tree root.

The operation $\text{Verify}(i,y,r,s)$ returns _Accept_ or _Reject_ depending on whether the $i$-th element of $A$ is $y$. It checks whether the authentication path $s$ is compatible with $i$, $y$ and the Merkle tree root $r$.

#### Notation
In our cases the sets $A$ will be of the form $A=(f(a), f(ab), f(ab^2), \dots, f(ab^L))$ for some elements $a,b\in\mathbb{F}$. It will be convenient to use the following abuse of notation. We will write $\text{Open}(A, ab^i)$ to mean $\text{Open}(A, i)$. Similarly, we will write $\text{Verify}(ab^i, y, r, s)$ instead of $\text{Verify}(i, y, r, s)$. Note that this is only notation and $\text{Verify}(ab^i, y, r, s)$ is only checking that the $y$ is the $i$-th element of the commited vector. It is not checking that it is an evaluation of a function at $ab^i$.

## Protocol
### Prover
#### Round 0: Transcript initialization
- Start a transcript.
- (Strong Fiat Shamir) Commit to the set of coefficientes of the transition and boundary polynomials, and add the commitments to the transcript.
#### Round 1: Build RAP
##### Round 1.1: Interpolate original trace
- Obtain the trace of the program $M_{\text{RAP1}} \in \mathbb{F}^{2^n \times m'}$
- For each column $M_j$ of the matrix $M_{\text{RAP1}}$, interpolate its values at the domain $D_S$ and obtain polynomials $t_j$ such that $t_j(g^i)=M_{i,j}$.
- Compute $[t_j] := \text{Commit}(t_j(D_{\text{LED}}))$ for all $j=1,\dots,m'$.
- Add $[t_j]$ to the transcript in increasing order.
##### Round 1.2: Commit extended trace
- Sample random values $a_1,\dots,a_l$ in $\mathbb{F}$ from the transcript.
- Use $a_1,\dots,a_l$ to build $M_{\text{RAP2}}\in\mathbb{F}^{2^n\times m''}$ following the specifications of the RAP process.
- For each column $\hat M_j$ of the matrix $M_{\text{RAP2}}$, interpolate its values at the domain $D_S$ and obtain polynomials $t_{m'+1}, \dots, t_{m' + m''}$ such that $t_j(g^i)=\hat M_{i,j}$.
- Compute $[t_j] := \text{Commit}(t_j(D_{\text{LED}}))$ for all $j=m'+1,\dots,m'+m''$.
- Add $[t_j]$ to the transcript in increasing order for all $j=m'+1,\dots,m'+m''$.
#### Round 2: Compute composition polynomial
- Sample $\alpha_1^B,\dots,\alpha_{m}^B$ and $\beta_1^B,\dots,\beta_{m}^B$ in $\mathbb{F}$ from the transcript.
- Sample $\alpha_1^T,\dots,\alpha_{n_T}^T$ and $\beta_1^T,\dots,\beta_{n_T}^T$ in $\mathbb{F}$ from the transcript.
- Compute $B_j := \frac{t_j - P^B_j}{Z_j^B}$.
- Compute $C_k := \frac{P^T_k(t_1, \dots, t_m, t_1(gX), \dots, t_m(gX))}{Z_k^T}$.
- Compute the _composition polynomial_
$$H := \sum_{k} (\alpha_k^T X^{d - d_k^T} + \beta_k^T)C_k + \sum_j (\alpha_j^BX^{d - d^B}+\beta_j^B)B_j$$
- Decompose $H$ as 
$$H = H_1(X^2) + XH_2(X^2)$$
- Compute commitments $[H_1]$ and $[H_2]$.
- Add $[H_1]$ and $[H_2]$ to the transcript.
#### Round 3: Evaluate polynomials in out of domain element
- Sample from the transcript until obtaining $z\in\mathbb{F}\setminus D_{\text{LDE}}$.
- Compute $H_1(z^2)$, $H_2(z^2)$, and $t_j(z)$ and $t_j(gz)$ for all $j$.
- Add $H_1(z^2)$, $H_2(z^2)$, and $t_j(z)$ and $t_j(gz)$ for all $j$ to the transcript.

#### Round 4: Compute and run FRI on the Deep composition polynomial
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

##### Round 4.2: FRI query phase

- For $s=0,\dots,Q-1$ do the following:
    - Sample random index $\iota_s \in [0, 2^{n+l}]$ from the transcript and let $\upsilon_s := \omega^{\iota_s}$.
    - Compute $\text{Open}(p_0(D_0), \upsilon_s)$.
    - Compute $\text{Open}(p_k(D_k), -\upsilon_s^{2^k})$ for all $k=0,\dots,n-1$.

##### Round 4.3: Open deep composition polynomial components
- Compute $\text{Open}(H_1(D_{\text{LDE}}), \upsilon_0)$, $\text{Open}(H_2(D_{\text{LDE}}), \upsilon_0)$.
- Compute $\text{Open}(t_j(D_{\text{LDE}}), \upsilon_0)$ for all $j=1,\dots, m$.

#### Build proof
- Send the proof to the verifier:
$$
\begin{align}
\Pi = ( &\\
&\{[t_j], t_j(z), t_j(gz): 0\leq j < m\}, \\
&[H_1], H_1(z^2),[H_2], H_2(z^2), \\
&\{[p_k]: 0\leq k < n\}, \\
&p_n, \\
&\{\text{Open}(p_0(D_0), \upsilon_s): 0\leq s < Q\}), \\
&\{\text{Open}(p_k(D_k), -\upsilon_s^{2^k}): 0\leq k< n, 0\leq s < Q\}, \\
&\text{Open}(H_1(D_{\text{LDE}}), \upsilon_0), \\
&\text{Open}(H_2(D_{\text{LDE}}), \upsilon_0), \\
&\{\text{Open}(t_j(D_{\text{LDE}}), \upsilon_0): 0 \leq j< m\}, \\
) &
\end{align}
$$

### Verifier
#### Notation
- Bold capital letters refer to commitments. For example $\mathbf{H}_1$ is the claimed commitment $[H_1]$.
- Greek letters with superscripts refer to claimed function evaluations. For example $\tau_j^z$ is the claimed evaluation $t_j(z)$.
- Gothic letters refer to authentication paths (e.g. $\mathfrak{H}_1$  is the authentication path of the opening of $H_1$).

#### Input
$$
\begin{align}
\Pi = ( &\\
&\{\mathbf{T}_j, \tau_j^z, \tau_j^{gz}: 0\leq j < m\}, \\
&\mathbf{H}_1, \eta_1^{z^2},\mathbf{H}_2, \eta_2^{z^2}, \\
&\{\mathbf{P}_k: 0\leq k < n\}, \\
&\pi, \\
&\{(\pi_0^{\upsilon_s}, \mathfrak{P}_0): 0\leq s < Q\}, \\
&\{(\pi_k^{-\upsilon_s^{2^k}}, \mathfrak{P}_k): 0\leq k< n, 0\leq s < Q\}, \\
&(\eta_1^{\upsilon_0}, \mathfrak{H}_1)\\
&(\eta_2^{\upsilon_0}, \mathfrak{H}_2)\\
&\{(\tau_j^{\upsilon_0}, \mathfrak{T}_j): 0 \leq j< m\}, \\
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
- For $s=0, \dots, Q-1$:
-- Sample random index $\iota_s \in [0, 2^{n+l}]$ from the transcript and let $\upsilon_s := \omega^{\iota_s}$.

#### Step 2: Verify claimed composition polynomial
- Compute $h := \eta_1^{z^2} + z \eta_2^{z^2}$
- Compute $b_j := \frac{\tau_j^z - P^B_j(z)}{Z_j^B(z)}$
- Compute $c_k := \frac{P^T_k(\tau_1^z, \dots, \tau_m^z, \tau_1^{gz}, \dots, \tau_m^{gz})}{Z_k^T(z)}$
- Verify 
$$h = \sum_{k} (\alpha_k^T z^{d - d_k^T} + \beta_k^T)c_k + \sum_j (\alpha_j^B z^{d - d^B}+\beta_j^B)b_j$$


#### Step 3: Verify FRI

- Check that the following are all _Accept_:
    - $\text{Verify}((\upsilon_s, \pi_0^{\upsilon_s}), \mathbf{P}_0, \mathfrak{P}_0)$ for all $0\leq s < Q$.
    - $\text{Verify}((-\upsilon_s^{2^k}, \pi_k^{-\upsilon_s^{2^k}}), \mathbf{P}_k, \mathfrak{P}_k)$  for all $0\leq k < n$, $0\leq s < Q$.
- For all $s=0,\dots,Q-1$:
    - For all $k=0,\dots,n-1$:
        - Solve the following system of equations on the variables $G, H$
$$
\begin{aligned}
\pi_k^{\upsilon_s^{2^{k}}} &= G + \upsilon_s^{2^k}H \\
\pi_k^{-\upsilon_s^{2^{k}}} &= G - \upsilon_s^{2^k}H
\end{aligned}
$$
        - Define $\pi_{k+1}^{\upsilon_s^{2^{k+1}}}:=G + \zeta_{k}H$
    - Check that $\pi_n^{\upsilon_s^{2^n}}$ is equal to $\pi$.


#### Step 4: Verify deep composition polynomial is FRI first layer

- Check that the following are all _Accept_:
    - $\text{Verify}((\upsilon_0, \eta_1^{\upsilon_0}), \mathbf{H}_1, \mathfrak{h}_1)$.
    - $\text{Verify}((\upsilon_0, \eta_2^{\upsilon_0}), \mathbf{H}_2, \mathfrak{h}_2)$.
    - $\text{Verify}((\upsilon_0, \tau_j^{\upsilon_0}), \mathbf{T}_j, \mathfrak{T}_j)$ for all $0\leq j < m$.
- Check that $\pi_0^{\upsilon_0}$ is equal to the following:

$$
\gamma\frac{\eta_1^{\upsilon_0} - \eta_1^{z^2}}{\upsilon_0 - z^2} + \gamma'\frac{\eta_2^{\upsilon_0} - \eta_2^{z^2}}{\upsilon_0 - z^2} + \sum_j \gamma_j\frac{\tau_j^{\upsilon_0} - \tau_j^{z}}{\upsilon_0 - z} + \gamma_j'\frac{\tau_j^{\upsilon_0} - \tau_j^{gz}}{\upsilon_0 - gz}
$$

