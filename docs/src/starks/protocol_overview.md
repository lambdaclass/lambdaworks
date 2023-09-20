# Protocol Overview

In this section, we start diving deeper before showing the formal protocol. If you haven't done so, we recommend reading the "Recap" section first.

At a high level, the protocol works as follows. The starting point is a matrix $T$ that encodes the trace of a valid execution of the program. This matrix needs to be in a particular format so that its correctness is equivalent to checking a finite number of polynomial equations on its rows. Transforming the execution to this matrix is what's called the arithmetization process.

Then a single polynomial $F$ is constructed that encodes the set of all the polynomial constraints. The satisfiability of all these constraints is equivalent to $F$ being divisible by some public polynomial $G$. So the prover constructs $H$ as the quotient $F/G$ called the composition polynomial.

Then the verifier chooses a random point $z$ and challenges the prover to reveal the values $F(z)$ and $H(z)$. Then the verifier checks that $H(z) = F(z)/G(z)$, which convinces him that the same relation holds at a level of polynomials and, in consequence, convinces the verifier that the private trace $T$ of the prover is valid.

In summary, at a very high level, the STARK protocol can be organized into three major parts:
- Arithmetization and commitment of execution trace.
- Construction of composition polynomial $H$.
- Opening of polynomials at random $z$.

# Arithmetization
As the Recap mentions, the trace is a table containing the system's state at every step. In this section, we will denote the trace as $T$. A trace can have several columns to store different aspects or features of a particular state at a specific moment. We will refer to the $j$-th column as $T_j$. You can think of a trace as a matrix $T$ where the entry $T_{ij}$ is the $j$-th element of the $i$-th state.

Most proving systems' primary tool is polynomials over a finite field  $\mathbb{F}$. Each column $T_j$ of the trace $T$ will be interpreted as evaluations of such a polynomial $t_j$. Consequently, any information about the states must be encoded somehow as an element in $\mathbb{F}$.

To ease notation, we will assume here and in the protocol that the constraints encoding transition rules depend only on a state and the previous one. Everything can be easily generalized to transitions that depend on many preceding states. Then, constraints can be expressed as multivariate polynomials in $2m$ variables
$$P_k^T(X_1, \dots, X_m, Y_1, \dots, Y_m)$$
A transition from state $i$ to state $i+1$ will be valid if and only if when we plug row $i$ of $T$ in the first $m$ variables and row $i+1$ in the second $m$ variables of $P_k^T$, we get $0$ for all $k$. In mathematical notation, this is
$$P_k^T(T_{i, 0}, \dots, T_{i, m}, T_{i+1, 0}, \dots, T_{i+1, m}) = 0 \text{    for all }k$$

These are called *transition constraints* and check the trace's local properties, where local means relative to specific rows. There is another type of constraint, called *boundary constraint*, and denoted $P_j^B$. These enforce parts of the trace to take particular values. It is helpful, for example, to verify the initial states.

So far, these constraints can only express the local properties of the trace. There are situations where the global properties of the trace need to be checked for consistency. For example, a column may need to take all values in a range but not in any predefined way. Several methods exist to express these global properties as local by adding redundant columns. Usually, they need to involve randomness from the verifier to make sense, and they turn into an interactive protocol called *Randomized AIR with Preprocessing*.

# Polynomial commitment scheme
To make interactions possible, a crucial cryptographic primitive is the Polynomial Commitment Scheme. This prevents the prover from changing the polynomials to adjust them to what the verifier expects.

Such a scheme consists of the commit and the open protocols. STARK uses a univariate polynomial commitment scheme that internally combines a vector commitment scheme and a protocol called FRI. Let's begin with these two components and see how they build up the polynomial commitment scheme.

## Vector commitments
Given a vector $Y = (y_0, \dots, y_M)$, commiting to $Y$ means the following. The prover builds a Merkle tree out of it and sends its root to the verifier. The verifier can then ask the prover to reveal, or *open*, the value of the vector $Y$ at some index $i$. The prover won't have any choice except to send the correct value. The verifier will expect the corresponding value $y_i$ and the authentication path to the tree's root to check its authenticity. The authentication path also encodes the vector's position $i$ and its length $M$.

In STARKs, all commited vectors are of the form $Y = (p(d_1), \dots, p(d_M))$ for some polynomial $p$ and some domain fixed domain $D = (d_1, \dots, d_M)$. The domain is always known to the prover and the verifier.

## FRI
The FRI protocol is a potent tool to prove that the commitment of a vector $(p(d_1), \dots, p(d_M))$ corresponds to the evaluations of a polynomial $p$ of a certain degree. The degree is a configuration of the protocol.

## Polynomial commitments
STARK uses a univariate polynomial commitment scheme. The following is what is expected from the **commit** and **open** protocols:
- *Commit*: given a polynomial $p$, the prover produces a sort of hash of it. We denote it here by $[p]$, called the *commitment* of $p$. This hash is unique to $p$. The prover usually sends $[p]$ to the verifier.
- *Open*: this is an interactive protocol between the prover and the verifier. The prover holds the polynomial $p$. The verifier only has the commitment $[p]$. The verifier sends a value $z$ to the prover at which he wants to know the value $y=p(z)$. The prover sends a value $y$ to the verifier, and then they engage in the *Open* protocol. As a result, the verifier gets convinced that the polynomial corresponding to the hash $[p]$ evaluates to $y$ at $z$.

Let's see how both of these protocols work in detail. Some configuration parameters are needed: 
- Powers of two $N = 2^n$ and $M = 2^m$ with $n < m$.
- A vector $D=(d_1,\dots,d_M)$, with $d_i$ in $\mathbb{F}$ for all $i$ and $d_i\neq d_j$ for all $i\neq j$.
- An integer $k > 0$.

The commitment scheme will only work for polynomials of degree at most $N$. This means anyone can commit to any polynomial, but the Open protocol will pass only for polynomials satisfying that degree bound.

### Commit
Given a polynomial $p$, the commitment $[p]$ is just the commitment of the vector $(p(d_1), \dots, p(d_M))$. That is, $[p]$ is the root of the Merkle tree of the vector of evaluations of $p$ at $D$.

### Open
It is an interactive protocol. So assume there is a prover and a verifier. We describe the process considering an honest prover. In the next section, we analyze what happens for malicious provers.

The prover holds the polynomial $p$, and the verifier only the commitment $[p]$ of it. There is also an element $z$. The prover evaluates $p(z)$ and sends the result to the verifier. As we mentioned, the goal is to generate proof of the validity of the evaluation. Let us denote $y := p(z)$. This is a value now both the prover and verifier have.

The protocol has three steps:

- **FRI on $p$**: First, the prover and the verifier engage in a FRI protocol for polynomials of degree at most $N$ to check that $[p]$ is the commitment of such a polynomial. We will assume from now on that the degree of $p$ is at most $N$. There is an optimization that completely removes this step—more on that [later](#optimize-the-open-protocol-reusing-fri-internal-challenges).

- **FRI on $(p-y)/(x-z)$**: Since $p(z) = y$, the polynomial $p$ can be written as $p = y + (x - z) q$ for some polynomial $q$. The prover computes the commitment $[q]$ and sends it to the verifier. Now they engage in a FRI protocol for polynomials of degree at most $N-1$, which convinces the verifier that $[q]$ is the commitment of a polynomial of degree at most $N-1$. 

- **Point checks**: From the point of view of the verifier the commitments $[p]$ and $[q]$ are still potentially unrelated. Therefore, there is a check to ensure that $q$ was computed properly from $p$ following the formula $q = (p-y)/(x-z)$ and that $[q]$ is its commitment. To do this, the verifier challenges the prover to open $[p]$ and $[q]$ as vectors. They use the open protocol of the vector commitment scheme to reveal the values $p(d_i)$ and $q(d_i)$ for some random point $d_i \in D$ chosen by the verifier. Next he checks that $p(d_i) = y + (d_i - z) q(d_i) $. They repeat this last part $k$ times and, as we'll analyze in the next section, this whole thing will convince the verifier that $p = y + (x -z) q$ as polynomials with overwhelming probability (about $(N/M)^k$). Finally, the verifier deduces that $p(z) = y$ from this equality.

### Soundness
Let's analyze why the open protocol is sound. Keep in mind that the prover always has to provide a commitment $[q]$ of a polynomial of degree at most $N-1$ that satisfies $p(d_i) = y + (d_i - z) q(d_i)$ for the chosen values of the verifier. That's the goal. An essential but subtle part of the soundness is that $D$ is a vector of length $M>N$. To understand why let's see what would happen if $N = M$.

#### Case $M=N$
Suppose the prover tries to cheat the verifier by sending a value $y$ different from $p(z)$. This means that $p - y$ is not divisible by $X-z$ as polynomials. But as long as $z$ is not in $D$ the prover can interpolate the values $$\frac{p(d_1) - y}{d_1 - z}, \dots, \frac{p(d_N) - y}{d_N - z},$$
at $D$ and obtain some polynomial $q$ of degree at most $N-1$. This polynomial satisfies $p(d_i) = y + (d_i - z) q(d_i)$ for all $d_i \in D$, but the equality of polynomials $p = y + (x - z) q$ does not hold. Mainly because that would imply that $p(z) = y$, and we are assuming the opposite case. Nevertheless, if they continue with the open protocol, FRI will pass since $q$ is a polynomial of degree at most $N-1$. All subsequent point checks will pass since $q$ was crafted especially for that. 

So how come $p$ is different from $y + (x - z) q$ but has the property that $p(d_i) = y + (d_i - z) q(d_i)$ for all $d_i \in D$? FRI guarantees $q$ is of degree at most $N-1$, which implies that $y + (x - z) q$ is of degree at most $N$. Then, since $p$ is of degree at most $N$, the difference $$f := y + (X-z) q - p$$ is again of degree at most $N$. A polynomial $f$ of degree at most $N$ is zero if and only if we can prove that $f(d) = 0$ for $N+1$ distinct points. But the construction of $q$ only shows that $f(d_i) = 0$ for all $d_i \in D$, a set of $N$ points. It's not enough. One extra point and the equality $f=0$ would hold. Of course, that point doesn't exist, again, because that would imply that $p(z) = y$, which we assume is not true.

#### Case $M=2N$
Let's see one more example where we double the de size of $D$ but keep the same bound for the degree of the polynomials. So polynomials are of degree at most $N$, and the length of $D$ is $M = 2N$.

Again let's assume the prover wants to cheat the verifier similarly and claims that $y$ is $p(z)$ but, in reality, $y \neq p(z)$. Inevitably, for the Open protocol to succeed, the prover has to send a commitment $[q]$ of some polynomial of degree at most $N-1$. The strategy of the prover is pretty much constrained to be the same. But now he can't interpolate $q$ as before in every element of $D$ because that would produce a polynomial of degree at most $2N-1$. This most likely won't pass the FRI check.

Alternatively, he can choose some subset of $D' \subset D$ of size $N$ and interpolate only there to get a polynomial $q$ of degree at most $N-1$. That will succeed in the FRI phase, but what happens with the point checks? The check will pass if the verifier chooses a challenge $d_i$ that belongs to $D'$. If $d_i \notin D'$, then the check will inevitably fail. This is because, as we saw in the previous case, one extra successful point to the already $N$ points where $q$ was interpolated was enough to prove that $p(z) = y$, which we assume now not to hold. This means, if $d_i \notin D'$, then $p(d_i)$ does not coincide with $y + (d_i - z) q(d_i)$ and the point check fails. So if the verifier chooses the challenges following a uniform distribution, the chances of the prover being caught cheating are $1/2$ for only one challenge. If the verifier performs $k$ challenges, then the chance of the prover not getting caught is $1/2^{k}$.

#### General case: the blowup factor
If the ratio between $M$ and $N$ is $2$, then $k$ challenges give $1/2^{k}$ of probability for a malicious prover to succeed. If the ratio is $4$, that is, if $M = 4N$, then that probability is $1/4^k$ for the same number of point checks. This provides another way of improving the soundness error. The larger the ratio, the more complex it is to cheat in the open protocol. This ratio is what's called *the blowup factor*. It is a security parameter, and finding a balance between the number of challenges $k$ and the size of $D$ is part of the configuration of the protocol. Increasing the size of $D$ makes committing an expensive operation since it involves building a Merkle tree with a vector of the size of $D$. But increasing the number of challenges makes the size of the final proof larger.

We denote the blowup factor by $b$, and it's always assumed to be a power of $2$.

### Batch
During proof generation, polynomials are committed and opened several times. Computing these for each polynomial independently is costly. In this section, we'll see how batching polynomials can reduce the amount of computation. Let $P=\{p_1, \dots, p_L\}$ be a set of polynomials. We will commit and open $P$ as a whole. We note this batch commitment as $[P]$.

We need the same configuration parameters as before: $N=2^n$, $M=2^m$ with $N<M$, a vector $D=(d_1, \dots, d_M)$ and $k>0$.

As described earlier, to commit to a single polynomial $p$, a Merkle tree is built over the vector $(p(d_1), \dots, p(d_m))$. When committing to a batch of polynomials $P=\{p_1, \dots, p_n\}$, the leaves of the Merkle tree are instead the concatenation of the polynomial evaluations. That is, in the batch setting, the Merkle tree is built for the vector $(p_1(d_1)||\dots||p_L(d_1), \dots, p_L(d_m)||\dots||p_n(d_m))$. The commitment $[P]$ is the root of this Merkle tree. This reduces the proof size: we only need one Merkle tree for $L$ polynomials. The verifier can then only ask for values in batches. When the verifier chooses an index $i$, the prover sends $p_1 (d_i) , \dots , p_L (d_i)$ along with one authentication path. The verifier on his side computes the concatenation $p_1(d_i)||\dots||p_L(d_i)$ and validates it with the authentication path and $[P]$. This also reduces the computational time. By traversing the Merkle tree one time, it can reveal several components simultaneously.

The batch open protocol proceeds similarly to the case of a single polynomial. The prover will try to convince the verifier that the committed polynomials $P$ at $z$ evaluate to some values $y_i = p_i(z)$. In the batch case, the prover will construct the following polynomial:

$$
Q:=\sum_{i=1}^{L}\gamma_i\frac{p_i-y_i}{X-z}
$$

Where $\gamma_i$ are challenges provided by the verifier. The prover commits to $Q$ and sends $[Q]$ to the verifier. Then the prover and verifier continue similarly to the normal open protocol for $Q$ only. This means they engage in a FRI protocol for polynomials of degree at most $N-1$ for $Q$. Then they engage in the point checks for $Q$. Here, for each challenge $d_i$, the prover uses one authentication path for $[Q]$ to reveal $Q(d_i)$ and use one authentication path for $[P]$ to batch reveal values $p_1(d_i),\dots, p_L(d_i)$. Successful point checks here mean that $Q(d_i) = \sum_i \gamma_i(p_i(d_i) - y_i) / (d_i - z)$.

This is equivalent to running the open protocol $L$ times, one for each term $p_i$ and $y_i$. Note that this optimization makes a huge difference, as we only need to run the FRI protocol once instead of running it once for each polynomial.

### Optimize the open protocol reusing FRI internal challenges

There is an optimization for the open protocol to avoid running FRI to check that $p$ is of degree at most $N$. The idea is as follows. Part of FRI protocol for $[q]$, to check that it is of degree at most $N-1$, involves revealing values of $q$ at other random points $d_i$ also chosen by the verifier. These are part of the internal workings of FRI. These challenges are unrelated to what we mentioned before. So if one removes the FRI check for $p$, the point checks of the open protocol need to be performed on these challenges $d_i$ of the FRI protocol for $[q]$. This optimization is included in the formal description of the protocol.

## References

- [Transparent Polynomial Commitment Scheme with Polylogarithmic Communication Complexity](https://eprint.iacr.org/2019/1020)
- [Summary on FRI low degree test](https://eprint.iacr.org/2022/1216)
- [DEEP FRI](https://eprint.iacr.org/2019/336)
- [Thank goodness it's FRIday](https://vitalik.ca/general/2017/11/22/starks_part_2.html)
- [Diving DEEP FRI](https://blog.lambdaclass.com/diving-deep-fri/)

# High-level description of the protocol
The protocol is split into rounds. Each round more or less represents an interaction with the verifier. Each round will generally start by getting a challenge from the verifier. 

The prover will need to interpolate polynomials, and he will always do it over the set $D_S = \{g^i \}_{i=0}^{2^n-1} \subseteq \mathbb{F}$, where $g$ is a $2^n$ root of unity in $\mathbb{F}$. Also, the vector commitments will be performed over the set $D_{LDE} = (h, h \omega, h \omega^2, \dots, h \omega^{2^{n + l}})$ where $\omega$ is a $2^{n + l}$ root of unity and $h$ is some field element. This is the set we denoted $D$ in the commitment scheme section. The specific choices for the shapes of these sets are motivated by optimizations at a code level.

## Round 1: Arithmetization and commitment of the execution trace
In **round 1**, the prover commits to the columns of the trace $T$. He does so by interpolating each column $j$ and obtaining univariate polynomials $t_j$.
Then the prover commits to $t_j$ over $D_{LDE}$. In this way, we have $T_{i,j}=t_j(g^i)$.
From now on, the prover won't be able to change the trace values $T$. The verifier will leverage this and send challenges to the prover. The prover cannot know in advance what these challenges will be. Thus he cannot handcraft a trace to deceive the verifier.

As mentioned before, if some constraints cannot be expressed locally, more columns can be added to make a constraint-friendly trace. This is done by committing to the first set of columns, then sampling challenges from the verifier and repeating round 1. The sampling of challenges serves to add new constraints. These constraints will ensure the new columns have some common structure with the original trace. In the protocol, extended columns are referred to as the *RAP2* (Randomized AIR with Preprocessing). The matrix of the extended columns is denoted $M_{\text{RAP2}}$.

## Round 2: Construction of composition polynomial $H$
**round 2** aims to build the composition polynomial $H$. This function will have the property that it is a polynomial if and only if the trace that the prover committed to at **round 1** is valid and satisfies the agreed polynomial constraints. That is, $H$ will be a polynomial if and only if $T$ is a trace that satisfies all the transition and boundary constraints.

Note that we can compose the polynomials $t_j$, the ones that interpolate the columns of the trace $T$, with the multivariate constraint polynomials as follows.
$$Q_k^T(x) = P_k^T(t_1(x), \dots, t_m(x), t_1(g x), \dots, t_m(\omega x))$$
These result in univariate polynomials. And the same can be done for the boundary constraints. Since $T_{i,j} = t_j(g^i)$, these univariate polynomials vanish at every element of $D$ if and only if the trace $T$ is valid.

As we already mentioned, this is assuming that transitions only depend on the current and previous state. But it can be generalized to include *frames* with three or more rows or more context for each constraint. For example, in the Fibonacci case, the most natural way is to encode it as one transition constraint that depends on a row and the two preceding it, as we already did in the Recap section. The STARK protocol checks whether the function $\frac{Q_k^T}{X^{2^n} - 1}$ is a polynomial instead of checking that the polynomial is zero over the domain $D =\{g_i\}_{i=0}^{2^n-1}$. The two statements are equivalent.

The verifier could check that all $\frac{Q_k^T}{X^{2^n} - 1}$ are polynomials one by one, and the same for the polynomials coming from the boundary constraints. But this is inefficient; the same can be obtained with a single polynomial. To do this, the prover samples challenges and obtains a random linear combination of these polynomials. The result of this is denoted by $H$ and is called the composition polynomial. It integrates all the constraints by adding them up. So after computing $H$, the prover commits to it and sends the commitment to the verifier. The rest of the protocol aims to prove that $H$ was constructed correctly and is a polynomial, which can only be true if the prover has a valid extension of the original trace.

## Round 3: Evaluation of polynomials at $z$
The verifier must check that $H$ was constructed according to the protocol rules. That is, $H$ has to be a linear combination of all the functions $\frac{Q_k^T}{X^{2^n}-1}$ and similar terms for the boundary constraints. To do so, in **round 3** the verifier chooses a random point $z\in\mathbb{F}$ and the prover computes $H(z)$, $t_j(z)$ and $t_j(g z)$ for all $j$. With all these, the verifier can check that $H$ and the expected linear combination coincide, at least when evaluated at $z$. Since $z$ was chosen randomly, this proves with overwhelming probability that $H$ was properly constructed.

## Round 4: Run batch open protocol 
In this round, the prover and verifier engage in the batch open protocol of the polynomial commitment scheme described above to validate all the evaluations at $z$ from the previous round.
