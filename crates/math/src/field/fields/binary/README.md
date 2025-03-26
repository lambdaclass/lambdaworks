# Binary Fields

This module implements binary fields of the form $GF(2^{2^n})$ (i.e. a finite field with $2^{2^n}$ elements) by constructing a tower of field extensions. It doesn't implement `IsField` or `FieldElement`, because we wanted elements from different field extensions to coexist within the same struct, allowing us to optimize each operation to work simultaneously across all levels.

Binary fields are particularly useful for verifiable computing applications, including SNARKs and STARKs, due to their efficient implementation and favorable properties.

## Overview

The tower of binary fields provides a powerful alternative to prime fields for cryptographic applications. This implementation represents field elements as multivariable polynomials with binary coefficients in $GF(2) = \{0, 1\}$, where these coefficients are stored as bits in a `u128` integer. The tower structure is built recursively, with each level representing an extension of the previous field.

Key features of this implementation:

- Supports field extensions from level 0 $(GF(2))$ to level 7 $(GF(2^{128}))$.
- Efficient arithmetic operations optimized for binary fields.
- Karatsuba optimization for multiplication.

## Implementation Details

### Tower Construction

Let's explain the theory behind. To expand the binary field $GF(2) = \{0, 1\}$, we construct a tower of field extensions in the following way:

**Level 0:** The tower starts at level 0 with the base field of two elements $GF(2^{2^0}) = \{0, 1\}$. 

**Level 1:** Then at level 1, we define the field extension $GF(2^{2^1})$ whose elements are univariate polynomials with binary coefficients and variable $x_0$ such that ${x_0}^2 = x_0 + 1$. Note that this means that the polynomials are lineal (have degree at most 1). Therefore, this field extension has $2^{2^1}$ elements. We represent them as the binary expression of integers:

$$\begin{aligned}
00 &= 0 \\
01 &= 1 \\
10 &= x_0 \\
11 &= x_0 + 1
\end{aligned}$$

**Level 2:** At level 2, we define the field extension $GF(2^{2^2})$. In this case the elements are polynomials with binary coefficients and two variables, $x_0$ and $x_1$. The first one keeps satisfying ${x_0}^2 = x_0 + 1$ and in addition the second one satisfies ${x_1}^2 = x_1 \cdot x_0 + 1$. This means that the polynomials are lineal in each variable. Therefore, this field extension has $2^{2^2}$ elements:

$\begin{array}{llll}
0000 = 0  & 0100 = x_1  & 1000 = x_1x_0 & 1100 = x_1x_0 + x_1 \\
0001 = 1  & 0101 = x_1 + 1  & 1001 = x_1x_0 + 1  & 1101 = x_1x_0 + x_1 + 1 \\
0010 = x_0  & 0110 = x_1 + x_0  & 1010 = x_1x_0 + x_0  & 1110 = x_1x_0 + x_1 + x_0 \\
0011 = x_0 + 1  & 0111 = x_1 + x_0 + 1  & 1011 = x_1x_0 + x_0 + 1  & 1111 = x_1x_0 + x_1 + x_0 + 1
\end{array}$

**Level 3:** At level 3, we define $GF(2^{2^3})$ in the same way. This time the polynomials have three variables $x_0$, $x_1$ and $x_2$. The first two variables satisfy the equations mentioned before and in addition the last one satisfies ${x_2}^2 = x_2 \cdot x_1 + 1$. This field extension has $2^{2^3}$ elements.

**Level $n$:** Continuing this argument, in each level $n$ we define the field extension $GF(2^{2^n})$ using polynomials of $i$ variables with ${x_n}^2 = x_n \cdot x_{n-1} + 1$.

Our implementation admits until level $n = 7$.



### Element Representation

A `TowerFieldElement` is represented by:

- `value`: A `u128` integer where the bits represent the coefficients of the polynomial.
- `num_level`: The level in the tower (0-7).

For example, if `value = 0b1101` and `num_level = 2`, this represents the polynomial $x_0\cdot x_1 + x_1 + 1$ in $GF(2^4)$.


### Field Operations

The implementation provides efficient algorithms for:

- Addition and subtraction (XOR operations).
- Multiplication using a recursive tower approach with Karatsuba optimization.
- Inversion using Fermat's Little Theorem.
- Exponentiation using square-and-multiply.

## API Usage

### Creating Field Elements

The method `new` handles possible overflows in this way:
- If the level input is greater than 7, then the element's `num_level` is set as 7.
- If the value input doesn't fit in the level given, then we take just the less significant bits that fit in that level and set it as the element's `value`.

```rust
use lambdaworks_math::field::fields::binary::field::TowerFieldElement;

// Create elements at different tower levels
let element_level_0 = TowerFieldElement::new(1, 0);  // Element '1' in GF(2)
let element_level_1 = TowerFieldElement::new(3, 1);  // Element '11' in GF(2^2)
let element_level_2 = TowerFieldElement::new(123, 2); // Element '1011' in GF(2^4)
let element_level_3 = TowerFieldElement::new(123, 3); // Element '01111011'in GF(2^8)
let element_level_7 = TowerFieldElement::new(123, 25); // Element '0..01111011'in GF(2^128)

// Create zero and one
let zero = TowerFieldElement::zero();
let one = TowerFieldElement::one();

// Create from integer values
let from_u64 = TowerFieldElement::from(42u64);
```

### Basic Operations

```rust
use lambdaworks_math::field::fields::binary::field::TowerFieldElement;

// Create two elements
let a = TowerFieldElement::new(5, 2);  // '0101' in GF(2^4)
let b = TowerFieldElement::new(3, 2);  // '0011' in GF(2^4)

// Addition (XOR operation)
let sum = a + b;  // '0110' = 6

// Subtraction (same as addition in binary fields)
let difference = a - b;  // '0110' = 6

// Multiplication
let product = a * b;  // '1111' = 15

// Inversion
let a_inverse = a.inv().unwrap();
assert_eq!(a * a_inverse, TowerFieldElement::one());

// Exponentiation
let a_cubed = a.pow(3);
```

### Working with Different Levels

```rust
use lambdaworks_math::field::fields::binary::field::TowerFieldElement;

// Elements at different levels
let a = TowerFieldElement::new(3, 1);  // Level 1: GF(2^2)
let b = TowerFieldElement::new(5, 2);  // Level 2: GF(2^4)

// Operations automatically promote to the higher level
let sum = a + b;  // Result is at level 2
assert_eq!(sum.num_level(), 2);

// Splitting and joining elements
let element = TowerFieldElement::new(0b1010, 2);  // Level 2
let (hi, lo) = element.split();  // Split into two level 1 elements
assert_eq!(hi.value(), 0b10);    // High part: '10'
assert_eq!(lo.value(), 0b10);    // Low part: '10'

// Join back
let rejoined = hi.join(&lo);
assert_eq!(rejoined, element);
```

## Applications

Binary tower fields are particularly useful for:

1. **SNARKs and STARKs**: These fields enable efficient proof systems, especially when working with binary operations.

2. **Binius**: A SNARK system that leverages binary fields for improved performance and simplicity.


## Performance Considerations

- Operations in binary fields are generally faster than in prime fields for many applications.
- XOR-based addition/subtraction is extremely efficient.
- The tower structure enables optimized implementations of multiplication and other operations.

## References

- [SNARKs on Binary Fields: Binius](https://blog.lambdaclass.com/snarks-on-binary-fields-binius/) - LambdaClass Blog
- [Binius: SNARKs on Binary Fields](https://vitalik.eth.limo/general/2024/04/29/binius.html) - Vitalik Buterin's explanation
- [Binary Tower Fields are the Future of Verifiable Computing](https://www.irreducible.com/posts/binary-tower-fields-are-the-future-of-verifiable-computing) - Irreducible
