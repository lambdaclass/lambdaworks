# Lambdaworks Groth16 Prover

An under-optimized implementation of [Groth16](https://eprint.iacr.org/2016/260) protocol. To use it with Circom, check the [examples](./circom-adapter/src/README.md).

## Introduction

Over the last decade, SNARKs (succinct, non-interactive arguments of knowledge) and STARKs (scalable, transparent arguments of knowledge) have been gaining attention due to their applications in verifiable private computation and scalability of blockchains.

Groth introduced this [proof system](https://eprint.iacr.org/2016/260.pdf) in 2016 and saw an early application in ZCash. The protocol relies on pairing-friendly elliptic curves, such as BN254, BLS12-381, and BLS12-377 (more later). Its proof size is among the smallest (consisting of only three elliptic curve elements) and fastest to verify. The main drawback is that it needs a trusted setup per program. In other words, we need to regenerate all the parameters whenever we want to prove a new program (or change the original one).

## Arithmetization

To prove the execution of a given program, we have to transform it to a SNARK (succinct, non-interactive argument of knowledge) friendly form. One of such forms is arithmetic circuit satisfiability, where one can prove knowledge of a valid circuit assignment. This first step, known as arithmetization, is the program's transformation into an arithmetic circuit or equivalent form.

## R1CS

Arithmetic circuits can be expressed equivalently as (quadratic) rank one constraint systems (R1CS), which are systems of equations of the form:
$$(Az)\times (Bz) = Cz$$
where $A, B, C$ are matrices of size $m + 1$ rows by $n + 1$ columns, $z$ is a (column) vector of size $n + 1$ and $\times$ indicates the componentwise product of the resulting vectors.

We can alternatively view this compact form as
$\left( \sum_k a_{0k} z_k \right) \left( \sum_k b_{0k} z_k \right) - \left( \sum_k c_{0k} z_k \right) = 0$
$\left( \sum_k a_{1k} z_k \right) \left( \sum_k b_{1k} z_k \right) - \left( \sum_k c_{1k} z_k \right) = 0$
$\left( \sum_k a_{2k} z_k \right) \left( \sum_k b_{2k} z_k \right) - \left( \sum_k c_{2k} z_k \right) = 0$
$\vdots$
$\left( \sum_k a_{mk} z_k \right) \left( \sum_k b_{mk} z_k \right) - \left( \sum_k c_{mk} z_k \right) = 0$

We could express these equations more compactly by using polynomials and prove the solution of the R1CS system more concisely. To this end, we will introduce quadratic arithmetic programs, [QAP](https://vitalik.eth.limo/general/2016/12/10/qap.html).

## Quadratic Arithmetic Program

We can interpret each column of the $A$ matrix as evaluations of some polynomial over some suitable domain. This is a common practice in many SNARKs, where we try to encode a vector as a polynomial; see, for example, our [post about STARKs](https://blog.lambdaclass.com/diving-deep-fri/). We sample $D_0 = \{ x_0 , x_1 , ... , x_n \}$ over the finite field and define the polynomial $A_i (x)$ as the polynomial of at most degree $n$ such that $A_i ( x_k ) = a_{ki}$.

For performance reasons, it is convenient to select as interpolation domain $D_0$ the n-th roots of unity since we can use the Fast Fourier Transform to interpolate. Similarly, we can interpret the columns of $B$ and $C$ as polynomials $B_k (x)$ and $C_k (x)$. Taking advantage of these polynomials, we can express the R1CS system in polynomial form,
$P (x) = \left( \sum_k A_{k} (x) z_k \right) \left( \sum_k B_{k} (x) z_k \right) - \left( \sum_k C_{k} (x) z_k \right)$

We can see that if we have a valid solution for the R1CS, the polynomial $P (x)$ evaluates to $0$ over $D_0$ (since we require the polynomial to interpolate the values of the columns of the matrices). Therefore, we can express the condition as 
$P (x) = 0$ for $x \in D_0$
We now introduce the vanishing polynomial over the set $D_0$, $Z_D (x) = \prod_k (x - x_k )$
So, if the polynomial $P (x)$ evaluates to $0$ over $D_0$, it is divisible by $Z_D (x)$. This can be written as there is some polynomial $h (x)$ such that
$P (x) = h(x) Z_D (x)$
The degree of the polynomial $h(x)$ is the degree of $P$ minus the degree of $Z_D$. An honest prover should be able to find the resulting quotient and use it to show that he correctly executed the program.

## Transforming QAP into a zero-knowledge proof

We need to make some transformation to the above problem if we want to turn it into a zero-knowledge proof. For a more detailed description of this process, see [here](https://www.rareskills.io/post/groth16). We must ensure that the prover cannot cheat and that the verifier cannot learn anything about the private input or witness. One key ingredient is a polynomial commitment scheme (PCS): we can make the prover commit to a given polynomial so that he cannot change it later. One such commitment scheme is the [KZG commitment](https://dankradfeist.de/ethereum/2020/06/16/kate-polynomial-commitments.html), where we use [pairing-friendly elliptic curves](https://static1.squarespace.com/static/5fdbb09f31d71c1227082339/t/5ff394720493bd28278889c6/1609798774687/PairingsForBeginners.pdf) to bind the prover to a polynomial. The scheme's security relies on the hardness of the discrete logarithm problem over the curve. Pairings can be considered an operation that allows a one-time multiplication between points in an elliptic curve. In our case, we will work over type3 III pairings, $\dagger : G_1 \times G_2 \rightarrow G_t$, which have the following nice property (bilinearity):
$(a g_1 ) \dagger (b g_2 ) = (ab) (g_1 \dagger g_2)$
To commit to a polynomial using KZG, we need to sample a random scalar $\tau$ (which is considered toxic waste and should be forgotten, or we could forge proofs) and generate the following sequence of points in the elliptic curve, whose generator is $g_1$,
$P_0 = g_1$, 
$P_1 = \tau g_1$
$P_2 = \tau^2 g_1$
$\vdots$
$P_n = \tau^n g_1$
Then, given a polynomial $p(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n$ we compute the commitment as
$\mathrm{cm} (p) = a_0 P_0 + a_1 P_1 + ... + a_n P_n$
which is the same as $\mathrm{cm} (p) = p(\tau) g_1$, that is, hiding the evaluation of $p(x)$ inside the elliptic curve. Because the discrete log problem is hard, we cannot use our knowledge of $g_1$ and $\mathrm{cm} (p)$ to obtain $p(\tau)$.

To check that the polynomial $p(x)$ evaluates to $v$ at $z$ we can use the fact that
$p(x) - v = (x - z)q(x)$
where $q(x)$ is the quotient polynomial of the division of $p(x)$ by $x - z$. The prover can produce proof of such evaluation by committing to $q(x)$ using the same trick. Still, the verifier will need some additional information (included in the verifying key), $g_2$ (the generator of the group $G_2$), and $\tau g_2$ (remember, nobody must know $\tau$). Then, using pairings, the verifier can check the evaluation using the points in the elliptic curves,
$(\mathrm{cm} (p) - vg_1 \dagger g_2) = a = p(\tau) (g_1 \dagger g_2)$
$\mathrm{cm} (q) \dagger (\tau g_2 - z g_2) = b = q(\tau) ( \tau - z)(g_1 \dagger g_2)$
If $a$ and $b$ are the same, and since $\tau$ is a random point with high probability, we assume that $p(z) = v$ (This depends on the Schwartz-Zippel lemma).

Remember that we want to prove that the verifier knows some $w$ and a polynomial $h(x)$ of degree $m - 1$ such that if $z= (1, x, w)$, the following condition holds
$\left( \sum_k A_{k} (x) z_k \right) \left( \sum_k B_{k} (x) z_k \right) = \left( \sum_k C_{k} (x) z_k \right) + h(x)Z_D (x)$

If we force the prover first to commit to the polynomials $A_k (x)$ and $B_k (x)$ and then produce the quotient polynomial, we have to make sure that he cannot forge $C_k (x)$ to fulfill the previous condition. To do so, we are going to introduce random shifts ($\alpha$ and $\beta$) to the evaluations:
$\mathrm{cm} (\sum A_i z_i ) = \sum (A_i (\tau) z_i) g_1 + \alpha g_1$ 
$\mathrm{cm} (\sum B_i z_i) = \sum (B_i (\tau) z_i) g_2 + \beta g_2$
The $B_i (x)$ are committed to using group $G_2$ so that we can compute the product on the left-hand side through a pairing,
$(\mathrm{cm} (\sum A_i  z_i )) \dagger ( \mathrm{cm} (\sum B_i  z_i )) = (\sum A_i (\tau) z_i )(\sum B_i (\tau) z_i ) (g_1 \dagger g_2)$

Because we introduce these shifts, we need to modify the $C_k$ term accordingly,
$\begin{equation}\left( \alpha + \sum_k A_{k} (x) z_k \right) \left( \beta + \sum_k B_{k} (x) z_k \right) = \\ \alpha \beta + \left( \sum_k (C_{k} (x) + \beta A_k (x) + \alpha B_k (x)) z_k \right) + h(x)Z_D (x) \end{equation}$
Since the prover cannot know $\alpha$ and $\beta$, we need to provide them hidden as part of the trusted setup, as $\alpha g_1$ and $\beta g_2$, so that we can compute
$(\alpha g_1) \dagger (\beta g_2) = \alpha \beta (g_1 \dagger g_2)$
so that we can compare this result to the pairing between the shifted $A_i$ and $B_i$.

Also, since the prover does not have $\alpha$ and $\beta$, he needs to be supplied with all the elements of the form $C_{k} (x) + \beta A_k (x) + \alpha B_k (x)$. However, when we want to calculate the product between these terms and $z$, we must recall that $z$ contains both the public input and the witness. The verifier cannot learn anything about the witness (therefore, the evaluations involving the witness should be provided by the prover). We introduce two additional variables, $\gamma$, and $\delta$, to split the variable $z$ between public input and witness. The first $k$ terms correspond to the public input, and these are encoded as
$K_i^v = \gamma^{- 1} (C_{i} (\tau) + \beta A_i (\tau) + \alpha B_i (\tau)) g_1$
for $i = 0, 1, 2 ... , k$. For the witness, we have
$K_i^p = \delta^{- 1} (C_{i} (\tau) + \beta A_i (\tau) + \alpha B_i (\tau)) g_1$
With these new parameters, we get
$\begin{equation}\left( \alpha + \sum_j A_{j} (x) z_j \right) \left( \beta + \sum_j B_{j} (x) z_j \right) = \\ \alpha \beta + \gamma \left( \sum_i^k \gamma^{- 1} (C_{i} (x) + \beta A_i (x) + \alpha B_i (x)) x_i \right) + \\
\delta \left( \sum_{j = k + 1}^n \delta^{- 1} (C_{i} (x) + \beta A_i (x) + \alpha B_i (x)) x_i \right) + h(x)Z_D (x) \end{equation}$
We can combine the last two terms into one (since they contain all the information that the verifier must not learn)
$D = \left( \sum_{j = k + 1}^n \delta^{- 1} (C_{i} (x) + \beta A_i (x) + \alpha B_i (x)) x_i \right) + h(x)Z_D (x)\delta^{- 1}$

Since we want to compute the product $h(x) Z_D(x)$ with the help of one pairing, we can compute the following group elements,
$Z_0 = \delta^{ - 1} Z_D (\tau)$
$Z_1 = \delta^{ - 1} \tau Z_D (\tau)$
$Z_2 = \delta^{ - 1} \tau^2 Z_D (\tau)$
$\vdots$
$Z_{m - 1} = \delta^{ - 1} \tau^{ m - 1 } Z_D (\tau)$

With these changes, the right-hand side of the QAP is the sum of 3 terms:
A constant (related to the random shifts).
A term involving the public input.
A term that contains the secret terms (known only to the prover).

## Setup

Groth16 requires sampling five random field elements to generate the proving and verifying key, $t, \alpha, \beta, \gamma, \delta$. These are toxic waste and should be discarded and wholly forgotten once the keys have been generated.

We will use a pairing-friendly elliptic curve (with type III pairing), with subgroups $G_1$ and $G_2$ of prime order $r$. We will call the generators $g_1$ and $g_2$, respectively. To make notation easier, we will write
$[x]_1 = x g_1$
$[x]_2 = x g_2$
to denote points in $G_1$ and $G_2$, where $x g$ means the scalar product of $x$ and the generator of the group (i.e., applying x times the elliptic curve group operation to the generator). We will follow the notation given by [DIZK](https://eprint.iacr.org/2018/691.pdf). First, we compute the following vectors,
$K_i^v (t) = \gamma^{-1} \left( \beta A_i(t) + \alpha B_i (t) + C_i (t)\right)$
for $i = 0, 1, 2 , ... k$,
$K_i^p (t) = \delta^{-1} \left( \beta A_i(t) + \alpha B_i (t) + C_i (t)\right)$
for $i = k+1, 1, 2 , ... n$ and
$Z_k (t) = t^k Z_D (t) \delta^{-1}$
for $k = 0, 1, 2, ... m - 1$.
The proving key consists of the following elements:
1. $[\alpha]_1$
2. $[\beta]_1$
3. $[\beta]_2$
4. $[\delta]_1$
5. $[\delta]_2$
6. $[A_0 (t) ]_1, [A_1 (t) ]_1 , ... , [A_n (t) ]_1$
7. $[B_0 (t) ]_1, [B_1 (t) ]_1 , ... , [B_n (t) ]_1$
8. $[B_0 (t) ]_2, [B_1 (t) ]_2 , ... , [B_n (t) ]_2$
9. $[K_{ k + 1 }^p (t)] , [ K_{ k + 2 }^p (t)] , ... , [K_n^p (t)]$
10. $[Z_0 (t)] , [Z_1 (t)] , ... , [ Z_{ m - 1 } (t)]$

The verifying key is much shorter and will contain in addition the value of one pairing because that value is constant:
1. $[\alpha]_1 \dagger [\beta]_2$
2. $[\gamma]_2$
3. $[\delta]_2$
4. $[K_0^v (t)]_1 , [K_1^v (t)]_1 , ... , [K_k^v (t)]_1$ 

## Proof generation

The prover receives the proving key and knows the polynomials representing the program and the public input, and he wants to prove that he has a witness satisfying that program. First, the prover needs to calculate the quotient polynomial $h(x)$ or, more precisely, its coefficients. The prover has to calculate
$$h(x) = \frac{\sum A_k(x) z_k \sum B_k (x) z_k - \sum C_k (x) z_k}{Z_D (X) }$$

The best way to evaluate this quotient is by choosing a domain $D_{ev}$, of size at least the degree of the quotient polynomial plus one and not containing elements from $D_0$ (the interpolation domain) and evaluating numerator and denominator at all the elements of $D_{ev}$. Since we have at least as many evaluations of the polynomial $h (x)$ as its degree plus one, we can reconstruct $h(x)$ via interpolation. In practice, the fastest way to do this is by using the Fast Fourier Transform for evaluation and interpolation. The prover now possesses a vector of coefficients $h_0 , h_1 , h_2 , ... , h_m$.

To ensure that the proof is zero-knowledge, the prover sample two random scalars, $r$ and $s$. 

The prover can compute the three elements of the proof, $\pi = ([\pi_1 ]_1 ,  [\pi_2 ]_2 , [\pi_3 ]_1)$ by doing the following calculations,
$[\pi_1 ]_1 = [\alpha]_1 + \sum z_k [A_k (t) ]_1 + r [\delta]_1$
$[\pi_2 ]_2 = [\beta]_2 + \sum z_k [B_k (t) ]_2 + s[\delta]_2$
$[\pi_2 ]_1 = [\beta]_1 + \sum z_k [B_k (t) ]_1 + s[\delta]_1$
$[h(t)z(t)]_1 = \sum h_i [Z_i (t)]_1$
$[\pi_3 ]_1 = \sum w_i [K_i^p ]_1 + [h(t)z(t)]_1 + s[\pi_1 ]_1 + r [\pi_2 ]_1 - rs [\delta]_1$

## Verification

The verifier has the verifying key, the public input and parses the proof as $[\pi_1 ]_1, [\pi_2 ]_2, [\pi_3 ]_1$ and computes the following:
$[\pi_1 ]_1 \dagger [\pi_2 ]_2 = P_1$
$[\pi_3 ]_1 \dagger [\delta]_2 + [\alpha]_1 \dagger [\beta]_2 + \left(\sum x_i [K_i^v ]_1 \right) \dagger [\gamma]_2 = P_2$

The proof is valid if $P_1$ and $P_2$ coincide. This is equivalent to checking the modified QAP.
