# Reed-Solomon Codes: Theory and Implementation

This example provides an introduction to Reed-Solomon codes, their mathematical foundations, and decoding algorithms. It serves as a companion to the [blog post](https://blog.lambdaclass.com/a-sharper-look-at-fri/) on FRI.

## Table of Contents

1. [Introduction to Error-Correcting Codes](#introduction-to-error-correcting-codes)
2. [Reed-Solomon Codes](#reed-solomon-codes)
3. [The Singleton Bound and MDS Property](#the-singleton-bound-and-mds-property)
4. [Unique Decoding: Berlekamp-Welch](#unique-decoding-berlekamp-welch)
5. [List Decoding](#list-decoding)
   - [Sudan's Algorithm](#sudans-algorithm)
   - [Guruswami-Sudan Algorithm](#guruswami-sudan-algorithm)
6. [Connection to STARKs and Proximity Testing](#connection-to-starks-and-proximity-testing)
7. [Usage](#usage)

---

## Introduction to Error-Correcting Codes

When transmitting data over noisy channels or storing it on unreliable media, errors are inevitable. Error-correcting codes add structured redundancy to data, enabling detection and correction of errors.

### Key Concepts

- **Alphabet**: A finite set of symbols (for us, elements of a finite field $\mathbb{F}_q$)
- **Block code**: A code that encodes fixed-length messages into fixed-length codewords
- **Parameters**: A code is described by $[n, k, d]$ where:
  - $n$ = *code length* (codeword size)
  - $k$ = *dimension* (message size)
  - $d$ = *minimum distance* (smallest Hamming distance between distinct codewords)

### Hamming Distance

The **Hamming distance** between two vectors is the number of positions where they differ:

$$d_H(x, y) = |\{i : x_i \neq y_i\}|$$

The **minimum distance** of a code $C$ is:

$$d = \min_{c_1 \neq c_2 \in C} d_H(c_1, c_2)$$

### Error Correction Capability

A code with minimum distance $d$ can:
- **Detect** up to $d - 1$ errors
- **Correct** up to $t = \lfloor(d - 1)/2\rfloor$ errors

This is because if we receive a word with at most $t$ errors, there's a unique closest codeword.

---

## Reed-Solomon Codes

Reed-Solomon codes are a family of **linear codes** over finite fields with remarkable properties.

### Definition

An *RS[n, k] code* over $\mathbb{F}_q$ is defined by:
1. An evaluation domain $D = \{\alpha_0, \alpha_1, \ldots, \alpha_{n-1}\} \subset \mathbb{F}_q$ of $n$ distinct points
2. A dimension $k \leq n$

**Encoding**: A message $(m_0, m_1, \ldots, m_{k-1}) \in \mathbb{F}_q^k$ is interpreted as coefficients of a polynomial:

$$p(x) = m_0 + m_1 x + m_2 x^2 + \cdots + m_{k-1} x^{k-1}$$

The codeword is the evaluation of $p$ at all points in $D$:

$$\text{Encode}(m) = (p(\alpha_0), p(\alpha_1), \ldots, p(\alpha_{n-1}))$$

### Example

Consider RS[8, 4] over $\mathbb{F}_{17}$ with domain $D = \{0, 1, 2, 3, 4, 5, 6, 7\}$.

Message: $(1, 2, 3, 4)$ represents polynomial $p(x) = 1 + 2x + 3x^2 + 4x^3$

Codeword: $(p(0), p(1), \ldots, p(7)) = (1, 10, 16, 7, 3, 14, 7, 7)$ (mod 17)

### Key Properties

1. *Linearity*: The sum of two codewords is a codeword
2. *Systematic form*: With proper choice of domain, the message appears in the codeword
3. *Minimum distance*: $d = n - k + 1$ (proved below)

---

## The Singleton Bound and MDS Property

### The Singleton Bound

**Theorem (Singleton Bound)**: For any $[n, k, d]$ code:

$$d \leq n - k + 1$$

**Proof idea**: A degree $k - 1$ polynomial has at most $k - 1$ roots. Since these are the evaluations of polynomials, two different polynomials can coincide at most in $k - 1$ points, being different in the remaining $n - k + 1$ points.

### Maximum Distance Separable (MDS) Codes

A code achieving $d = n - k + 1$ with equality is called Maximum Distance Separable (MDS).

**Theorem**: Reed-Solomon codes are MDS.

### Implications

For RS[n, k]:
- Minimum distance: $d = n - k + 1$
- Unique decoding radius: $t = \lfloor(n - k)/2\rfloor$
- Any $k$ symbols suffice to recover the message (erasure correction)

| Code | n | k | d | Rate | Unique Decoding |
|------|---|---|---|------|-----------------|
| RS[16, 8] | 16 | 8 | 9 | 0.50 | 4 errors |
| RS[32, 8] | 32 | 8 | 25 | 0.25 | 12 errors |
| RS[16, 4] | 16 | 4 | 13 | 0.25 | 6 errors |

---

## Unique Decoding: Berlekamp-Welch

When the number of errors is at most $t = \lfloor(n - k)/2\rfloor$, we can uniquely recover the original polynomial.

### The Key Equation Approach

Given received word $(r_0, r_1, \ldots, r_{n-1})$ with at most $t$ errors, we seek:
- The original polynomial $P(x)$ of degree $< k$
- The error locator polynomial $E(x)$ with roots at error positions

*Key insight*: $E(\alpha_i) \cdot P(\alpha_i) = E(\alpha_i) \cdot r_i$ for all $i$.

This is because:
- If $\alpha_i$ is not an error position: $r_i = P(\alpha_i)$, so equality holds trivially
- If $\alpha_i$ is an error position: $E(\alpha_i) = 0$, so both sides are zero

### The Algorithm

1. **Set up the linear system**: Define $N(x) = E(x) \cdot P(x)$. We have:
   - $\deg(E) \leq t$ with $E$ monic
   - $\deg(P) < k$
   - $\deg(N) < k + t$

2. **Write constraints**: For each $i \in \{0, \ldots, n-1\}$:
   $$N(\alpha_i) = E(\alpha_i) \cdot r_i$$

3. **Solve**: This gives $n$ linear equations in $(t) + k + (k + t) = 2k + 2t - 1$ unknowns. For $n \geq 2k + 2t - 1$ (which holds when $t \leq (n-k)/2$), the system is solvable.

4. **Recover P**: Compute $P(x) = N(x) / E(x)$.

### Complexity

- Naive Gaussian elimination: $O(n^3)$
- With structured matrix methods: $O(n^2)$ or better

### Example

RS[8, 4] with $t = 2$ errors.

Original: $P(x) = 1 + x + x^2 + x^3$

Codeword: $(1, 4, 15, 40, 85, 156, 259, 400)$ over suitable field

Received (errors at positions 1, 3): $(1, 7, 15, 50, 85, 156, 259, 400)$

Berlekamp-Welch finds $E(x) = (x-1)(x-3) = x^2 - 4x + 3$ and recovers $P(x)$.

---

## List Decoding

Beyond the unique decoding radius, multiple codewords may be equally close to the received word. List decoding returns all codewords within a specified radius.

### The List Decoding Problem

Given received word $r$ and radius $\tau$, find all polynomials $P$ of degree $< k$ such that:

$$|\{i : P(\alpha_i) \neq r_i\}| \leq \tau$$

### Why List Decoding?

- *Beyond unique decoding*: Can correct more than $(n - k)/2$ errors
- *Probabilistic applications*: If errors are random, usually only one candidate
- *Complexity theory*: List-decodable codes have applications in hardness amplification
- *STARKs*: Proximity testing relies on list decoding bounds

---

### Sudan's Algorithm

Sudan (1997) gave the first polynomial-time list decoding algorithm for RS codes beyond the unique decoding radius.

#### Decoding Radius

Sudan's algorithm corrects up to:

$$\tau_{Sudan} = n - \sqrt{2nk}$$

For RS[32, 8]: $\tau_{Sudan} = 32 - \sqrt{512} \approx 9$ errors (vs. 12 for unique decoding).

Note: Sudan's radius is better than unique decoding only when $k < n/2$.

#### The Algorithm

**Phase 1 - Interpolation**: Find a bivariate polynomial $Q(x, y) = A(x) + B(x) \cdot y$ such that:
- $Q(\alpha_i, r_i) = 0$ for all $i$
- $\deg(A) < n - \tau$, $\deg(B) < n - \tau - k + 1$

**Phase 2 - Root Finding**: Find all polynomials $f(x)$ of degree $< k$ such that $Q(x, f(x)) = 0$.

For the degree-1 case in $y$: $f(x) = -A(x)/B(x)$ when $B(x)$ divides $A(x)$.

#### Why It Works

If $P(x)$ agrees with $r$ on at least $n - \tau$ positions, then $Q(x, P(x))$ is a polynomial of degree $< n - \tau$ that vanishes at all agreement points. If $n - \tau > \deg(Q(x, P(x)))$, then $Q(x, P(x)) \equiv 0$, so $P$ is a root.

---

### Guruswami-Sudan Algorithm

Guruswami and Sudan (1999) improved the decoding radius to the optimal algebraic bound.

#### Decoding Radius

The Guruswami-Sudan algorithm corrects up to:

$$\tau_{GS} = n - \sqrt{nk}$$

For RS[32, 8]: $\tau_{GS} = 32 - \sqrt{256} = 16$ errors.

This is optimal: no polynomial-time algorithm can do better (under standard assumptions).

#### Key Innovation: Multiplicities

Instead of requiring $Q(\alpha_i, r_i) = 0$, we require $Q$ to vanish with multiplicity $m$ at each point.

**Definition**: $Q(x, y)$ vanishes with multiplicity $m$ at $(a, b)$ if all partial derivatives of order $< m$ vanish:

$$\frac{\partial^{i+j} Q}{\partial x^i \partial y^j}(a, b) = 0 \quad \text{for all } i + j < m$$

#### The Algorithm

**Parameter Selection**: Choose multiplicity $m$ and $(1, k-1)$-weighted degree bound $D$ to maximize decoding radius. Optimal choice:

$$m \approx \frac{n}{\sqrt{nk}}, \quad D \approx m \cdot \sqrt{nk}$$

**Phase 1 - Interpolation**: Find nonzero $Q(x, y)$ such that:
- $Q$ has $(1, k-1)$-weighted degree $< D$
- $Q$ vanishes with multiplicity $m$ at each $(\alpha_i, r_i)$

This is a linear algebra problem: the multiplicity constraint at each point gives $\binom{m+1}{2}$ linear conditions.

**Phase 2 - Root Finding (Roth-Ruckenstein)**: Find all polynomials $f(x)$ of degree $< k$ with $Q(x, f(x)) \equiv 0$.

The **Roth-Ruckenstein algorithm** builds $f$ coefficient by coefficient:
1. Start with $f_0$ = constant term of any root
2. Substitute $f(x) = f_0 + x \cdot g(x)$ and recurse on $Q'(x, y) = Q(x, f_0 + xy)/x^{\text{something}}$
3. The recursion depth is $k$ (degree of $f$)

#### Complexity

- Interpolation: $O(n^2 m^2)$ using linear algebra
- Root finding: $O(nk^2)$ using Roth-Ruckenstein

---

### Johnson Bound

The **Johnson bound** limits the list size in list decoding.

**Theorem**: For RS[n, k] with at most $\tau$ errors, the list size is bounded by:

$$|L| \leq \frac{n}{n - \tau - \sqrt{nk}}$$

when $\tau < n - \sqrt{nk}$.

| Code | Errors | List Bound |
|------|--------|------------|
| RS[16, 4] | 6 | 2.7 |
| RS[32, 8] | 12 | 2.9 |
| RS[64, 16] | 24 | 3.2 |

The list size grows slowly, making list decoding practical.

---

## Connection to STARKs and Proximity Testing

Reed-Solomon codes are fundamental to STARK proof systems.

### Low-Degree Testing

A function $f: D \to \mathbb{F}$ is a codeword of RS[n, k] if and only if it equals the evaluation of some polynomial of degree $< k$.

Low-degree testing asks: Given oracle access to $f$, determine if $f$ is close to a low-degree polynomial.

### Proximity Testing

*Problem*: Given oracle access to $f: D \to \mathbb{F}$, test whether $f$ is $\delta$-close to some RS codeword.

*Definition*: $f$ is $\delta$-close to $C$ if there exists $c \in C$ with:

$$\frac{d_H(f, c)}{n} \leq \delta$$

### The FRI Protocol

Fast Reed-Solomon IOP of Proximity (FRI) is a protocol for proximity testing with:
- Logarithmic proof size
- Logarithmic verifier time
- Quasi-Linear prover time ($\mathcal{O}(n\log n$)

FRI works by:
1. Folding: Reduce degree-$d$ polynomial to degree-$d/2$ using random challenge
2. Recursion: Repeat until constant degree
3. Verification: Check consistency of folding and final polynomial

### Proximity Gaps

A key ingredient in FRI soundness is proximity gaps:

*Theorem (Proximity Gap)*: If $f$ is $\delta$-far from all degree-$k$ polynomials, then with high probability over random folding, the folded function is $\delta'$-far from degree-$k/2$ polynomials, where $\delta' \approx \delta$.

This gap amplification ensures that "bad" functions are caught with high probability.

### Connection to List Decoding

List decoding bounds (especially Johnson bound) directly imply proximity gap results:
- If $f$ is far from RS codes, list decoding finds few/no candidates
- The gap between "close" and "far" is quantified by $n - \sqrt{nk}$

---

## Usage

### Running the Demo

```bash
cd examples/reed-solomon-codes
cargo run
```

This displays an interactive demonstration of:
1. Reed-Solomon encoding
2. Singleton bound verification
3. Berlekamp-Welch decoding with various error counts
4. List decoding comparison (Sudan vs Guruswami-Sudan)
5. Proximity testing concepts

### Running Tests

```bash
cargo test
```

### Code Examples

#### Basic Encoding

```rust
use reed_solomon_codes::reed_solomon::ReedSolomonCode;
use reed_solomon_codes::Babybear31PrimeField;
use lambdaworks_math::field::element::FieldElement;

type FE = FieldElement<Babybear31PrimeField>;

// Create RS[16, 8] code
let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);

// Message as polynomial coefficients
let message: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

// Encode
let codeword = code.encode(&message);
```

#### Unique Decoding

```rust
use reed_solomon_codes::berlekamp_welch;
use reed_solomon_codes::distance::introduce_errors_at_positions;

// Introduce 3 errors (within unique decoding radius of 4)
let corrupted = introduce_errors_at_positions(&codeword, &[0, 5, 10]);

// Decode
let result = berlekamp_welch::decode(&code, &corrupted, None).unwrap();
assert_eq!(result.polynomial.coefficients(), &message);
```

#### List Decoding

```rust
use reed_solomon_codes::guruswami_sudan::gs_list_decode;
use reed_solomon_codes::sudan::sudan_list_decode;
use lambdaworks_math::polynomial::Polynomial;

// Introduce errors beyond unique decoding radius
let corrupted = introduce_errors_at_positions(&codeword, &[0, 2, 4, 6, 8, 10]);

// Guruswami-Sudan list decoding
let gs_result = gs_list_decode(&code, &corrupted);
let original = Polynomial::new(&message);

// The list should contain the original polynomial
assert!(gs_result.candidates.contains(&original));
```

---

## References

1. **Reed, I. S., & Solomon, G.** (1960). Polynomial codes over certain finite fields. *Journal of SIAM*, 8(2), 300-304.

2. **Berlekamp, E. R., & Welch, L. R.** (1986). Error correction for algebraic block codes. U.S. Patent 4,633,470.

3. **Sudan, M.** (1997). Decoding of Reed-Solomon codes beyond the error-correction bound. *Journal of Complexity*, 13(1), 180-193.

4. **Guruswami, V., & Sudan, M.** (1999). Improved decoding of Reed-Solomon and algebraic-geometry codes. *IEEE Transactions on Information Theory*, 45(6), 1757-1767.

5. **Roth, R. M., & Ruckenstein, G.** (2000). Efficient decoding of Reed-Solomon codes beyond half the minimum distance. *IEEE Transactions on Information Theory*, 46(1), 246-257.

6. **Ben-Sasson, E., et al.** (2018). Fast Reed-Solomon interactive oracle proofs of proximity. *ICALP 2018*.

7. **LambdaClass Blog**: [STARKs series](https://blog.lambdaclass.com/)

---