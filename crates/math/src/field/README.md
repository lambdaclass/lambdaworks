# lambdaworks Fields

This folder contains the different field backends, including field extensions. To learn how to use our fields, see the [examples](https://github.com/lambdaclass/lambdaworks/blob/main/examples/README.md) under basic use of finite fields. Below we give a list of currently supported fields; if yours is not on the list, you can add it by implementing the traits and providing the constants.
- [Stark-252](./fields/fft_friendly/stark_252_prime_field.rs): the field currently used by Starknet and STARK Platinum prover. FFT-friendly.
- [Mini-Goldilocks](./fields/fft_friendly/u64_goldilocks.rs), also known as oxfoi prime ($2^{64} - 2^{32} + 1$). FFT-friendly.
- [Pallas base field](./fields/pallas_field.rs): this is also the scalar field of the Vesta elliptic curve.
- [Vesta base field](./fields/vesta_field.rs): this is also the scalar field of the Pallas elliptic curve.
- [Goldilocks-448](./fields/p448_goldilocks_prime_field.rs)
- [Mersenne-31](./fields/mersenne31/): $2^{31} - 1$ and its [quadratic extension](./fields/mersenne31/extensions.rs)
- [Baby Bear](./fields/fft_friendly/babybear_u32.rs) and its [quadratic extension](./fields/fft_friendly/quadratic_babybear.rs): FFT-friendly, $2^{31} - 2^{27} + 1$.
- [Scalar field of BN-254](../elliptic_curve/short_weierstrass/curves/bn_254/default_types.rs), and its quadratic extension, quartic, sextic and twelth degree extensions. This coincides with the base field of [Grumpkin](../elliptic_curve/short_weierstrass/curves/grumpkin/curve.rs)
- [Base field of BN-254](../elliptic_curve/short_weierstrass/curves/bn_254/field_extension.rs) and its quadratic extension. The base field coincides with the scalar field of [Grumpkin](../elliptic_curve/short_weierstrass/curves/grumpkin/curve.rs)
- [Scalar field of BLS12-381](../elliptic_curve/short_weierstrass/curves/bls12_381/default_types.rs), and its quadratic, sextic and twelth degree extensions. FFT-friendly.
- [Base field of BLS12-381](../elliptic_curve/short_weierstrass/curves/bls12_381/field_extension.rs) 
- [Scalar field of BLS12-377](../elliptic_curve/short_weierstrass/curves/bls12_377/curve.rs)
- [Base field of BLS12-377](../elliptic_curve/short_weierstrass/curves/bls12_377/field_extension.rs)
- [Base field of secp256k1](./fields/secp256k1_field.rs): the base field of Bitcoin's elliptic curve.
- [Scalar field of secp256k1](./fields/secp256k1_scalarfield.rs): the scalar field of Bitcoin's elliptic curve.

You also have the tooling to define quadratic and cubic extension fields.

## 📊 Benchmarks

Benchmark results are hosted [here](https://lambdaclass.github.io/lambdaworks/bench).

These are the results of execution of the benchmarks for finite field arithmetic using the STARK field prime (p = 3618502788666131213697322783095070105623107215331596699973092056135872020481). 

Differences of 3% are common for some measurements, so small differences are not statistically relevant.

ARM - M1

| Operation| N      | Arkworks  | lambdaworks |
| -------- | ---    | --------- | ----------- |
| `mul`    |   10k  | 112 μs | 115 μs   |
| `add`    |   1M   | 8.5 ms  | 7.0 ms    |
| `sub`    |   1M   | 7.53 ms   | 7.12 ms     |
| `pow`    |   10k  | 11.2 ms   | 12.4 ms    |
| `invert` |  10k   | 30.0 ms  | 27.2 ms   |

x86 - AMD Ryzen 7 PRO 

| Operation | N    | Arkworks (ASM)*  | lambdaworks |
| -------- | --- | --------- | ----------- |
| `mul`    |   10k  | 118.9 us | 95.7 us   |
| `add`    |   1M  | 6.8 ms  | 5.4 ms    |
| `sub`    |   1M  |  6.6 ms  |  5.2 ms   |
| `pow`    |   10k  |  10.6 ms   | 9.4 ms    |
| `invert` |  10k   | 34.2 ms  | 35.74 ms |

*assembly feature was enabled manually for that bench, and is not activated by default when running criterion

To run them locally, you will need `cargo-criterion` and `cargo-flamegraph`. Install it with:

```bash
cargo install cargo-criterion
```

Run the complete benchmark suite with:

```bash
make benchmarks
```

Run a specific benchmark suite with `cargo`, for example to run the one for `field`:

```bash
make benchmark BENCH=field
```

You can check the generated HTML report in `target/criterion/reports/index.html`

## Background on finite fields

Finite fields play a fundamental role in Cryptography. They work essentially as the rational or real numbers (where we have the operations of addition, subtraction, multiplication and division), except that the number of elements is finite (for example, 31, 101, but not infinite as real numbers). We will begin this explanation with the simplest types of finite fields, where the number of elements is given by a prime number (a prime number is an integer such that its only divisors are 1 and itself, like 7, 19, 31, but not 8, which is divisible by 1, 2, 4, and 8). We will denote the prime $p$ and the finite field whose size is equal to $p$, $\mathbb{F_p}$.

The elements of $\mathbb{F_p}$ are given by the possible remainders of the integer division of numbers by $p$. Remember that for any integer $n$, we can express $n = p q + r$, where $q$ is the quotient and $0 \leq r < p$ is the remainder. For example, $8 = 5 \times 1 + 3$. Expressing it in simpler terms, $\mathbb{F_p}$ is the set $\{ 0, 1, 2, 3, \dots , p - 1 \}$. We can define an addition operation $+ : \mathbb{F_p} \times \mathbb{F_p} \rightarrow \mathbb{F_p}$ with the following rule:

Whenever we add two integers $a, b$ such that $n = a + b$, if the result exceeds $p$, we take the remainder of the division of $n$ by $p$. For example, if $p = 7$, $\mathbb{F_7} = \{ 0, 1, 2, 3, 4, 5, 6 \}$, so
$6 + 5 = 11 \equiv 4 \pmod{7}$ 
$2 + 3 = 5 \equiv 5 \pmod{7}$
$3 + 4 = 7 \equiv 0 \pmod{7}$

The notation $a \equiv b \pmod{p}$ means that $a - b$ is divisible by $p$, or that $a$ and $b$ have the same remainder when divided by $p$. We can show that $\mathbb{F_p}$ together with this addition forms an [Abelian group](https://en.wikipedia.org/wiki/Abelian_group). You can check that for every element $k$ there is an element $m$ such that $k + m = 0$. We note that element $m = - k$. You can see that $p - k = - k$, and you can define subtraction.

We can also define multiplication in a similar way: whenever the product of two integers exceeds the modulus $p$, take the remainder. Except for $0$, we can show that for every element $x$, there is an element $y$, such that $x \times y = 1$ and we denote $y = x^{- 1}$. This allows us to define division as $n / x = n \times x^{- 1}$. We will use $\mathbb{F_p}^\star = \{ 1, 2, \dots p - 1 \}$, that is $\mathbb{F_p}$ without the element $0$. The number of elements in $\mathbb{F_p}^\star$ is $p - 1$ and we can show that is is also an Abelian group. We call it the multiplicative group of $\mathbb{F_p}$. We will say that the field is *FFT friendly* if $p - 1 = 2^m c$, where $c$ is an odd integer, and $m$ is *sufficiently large*. For example, $p = 2^{64} - 2^{32} + 1$ satisfies $p - 1 = 2^{32} (2^{32} - 1)$ is *FFT friendly* with $m = 32$, and this means we can use the radix-2 FFT algorithm for vectors of size up to $2^{32}$.

If we take a look at $\mathbb{F_p}$ with the operations $+$ and $\times$, we see that it satisfies the axioms for a [field](https://en.wikipedia.org/wiki/Field_(mathematics)). To define a field in lambdaworks, you need to specify the modulus $p$, then the library will handle all the operations. While it is possible to do operations by taking remainder, this is not performant in practice. lambdaworks relies on Montgomery arithmetic for general finite fields (unless they have some faster alternative, such as [Mersenne primes](https://en.wikipedia.org/wiki/Mersenne_prime)). Before jumping into how you define your finite field, we need to see how we are going to represent "big" numbers. Common types to represent integers are `u8`, `u16`, `u32`, `u64`, `u128`. For cryptographic applications, we need to work (in general) with larger integers, taking 256 bits or more. As these numbers do not fit into a single unsigned integer variable, we can express it using several limbs in a given base (in lambdaworks, we use base 64, where each limb is represent by `u64`). Mathematically,

$$n = \sum_{i = 0}^m a_k b^k$$

where $b = 2^{64}$. For a 256 bit number, we need $4$ limbs and the number looks like

$$n = a_0 + a_1 2^{64} + a_2 2^{128} + a_3 2^{192}$$

The number is expressed in little-endian form. Alternatively, we can also write things as

$$n = a_0 2^{192} + a_1 2^{128} + a_2 2^{64} + a_3$$

The number is expressed in big-endian form. This is the form we use in lambdaworks to work with big integers. To work with large integers, we provide `UnsignedInteger` types with variable number of limbs.

```rust
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct UnsignedInteger<const NUM_LIMBS: usize> {
    pub limbs: [u64; NUM_LIMBS],
}
```

You can create `UnsignedInteger` types whose size is a multiple of 64, such as 128, 192, 256, 320, 384. We provide `U128`, `U256` `U384` since these are commonly used with elliptic curves. There are several ways of assigning an `UnsignedInteger` a value. Some examples are:
- `UnsignedInteger::from(value: u128)`
- `UnsignedInteger::from(value: u64)`
- `UnsignedInteger::from(value: u16)`
- `UnsignedInteger::from(hex_str: &str)`
- `UnsignedInteger::from_hex_unchecked(hex_str: &str)`

For example,

```rust
    let modulus: U256 = U256::from_hex_unchecked(
        "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    );
```

Defines a value, modulus, whose hex representation is given by `0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001`. Inside, the function takes chunks of 8 bytes and interprets them as the limb. This represents the prime number of 77 decimal digits `28948022309329048855892746252171976963363056481941647379679742748393362948097`.

We can now jump to the definition of a finite field:
```rust
use crate::{
    field::fields::montgomery_backed_prime_fields::{IsModulus, MontgomeryBackendPrimeField},
    unsigned_integer::element::U256,
};

type VestaMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 4>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MontgomeryConfigVesta255PrimeField;
impl IsModulus<U256> for MontgomeryConfigVesta255PrimeField {
    const MODULUS: U256 = U256::from_hex_unchecked(
        "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    );
}

pub type Vesta255PrimeField = VestaMontgomeryBackendPrimeField<MontgomeryConfigVesta255PrimeField>;
```

We give a name to the field `Vesta255PrimeField`, which is going to use a Montgomery representation. Each element is composed of 4 limbs of 64 bits and the modulus is given by `MODULUS`. The compiler resolves all the necessary things to define the different field operations:
- Addition
- Doubling (as an optimization to addition when $a = b$)
- Multiplication
- Square (as an optimization to multiplication when $a = b$)
- Subtraction
- (Multiplicative) inversion/division
- Exponentiation/Power (using a square and multiply algorithm)
- Square root (In a finite field, we can compute the square root of an element if it is a [quadratic residue](https://en.wikipedia.org/wiki/Quadratic_residue) modulo $p$). When the element is a quadratic residue, the function returns the values $y$ and $- y$ such that $y^2 = (- y)^2 = x$.

It also resolves all the necessary constants. These involve:
- `pub const R2` : The square of the $R$ parameter
- `pub const MU` : $- \text{modulus}^{-1} \pmod{ 2^{64}}$
- `pub const ZERO` : the value of the neutral element for addition in Montgomery form (it is always 0).
- `pub const ONE` : the value of the unit in Montgomery form.
- `MODULUS_HAS_ONE_SPARE_BIT` : checks whether the highest bit in the modulus is set or not (this is useful for faster modular arithmetic).

Additionally, you have the following methods:
- `to_bytes_be` : transforms the element to bytes in big-endian form.
- `to_bytes_le` : transforms the element to bytes in little-endian form.
- `from_bytes_be` : creates an element from byte array in big-endian form.
- `from_bytes_le` : creates an element from byte array in little-endian form.
- `representative` : transforms the element from Montgomery form to standard form.
- `to_hex` : transforms the element to a hex string.

To practice some operations, we are going to define a new field and do some operations. The following is the base field for the `secp256k1` elliptic curve, best known as Bitcoin's curve:

```rust
 #[derive(Clone, Debug)]
struct SecpModulus;
impl IsModulus<U256> for SecpModulus {
    const MODULUS: U256 = UnsignedInteger::from_hex_unchecked(
        "0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
    );
}
type SecpMontField = U256PrimeField<SecpModulus>;
type SecpMontElement = FieldElement<SecpMontField>;
```

We will create some elements and perform operations:

```rust
let minus_3 = -SecpMontElement::from_hex_unchecked("0x3");
let three = SecpMontElement::from_hex_unchecked("0x3");
assert_eq!(three + minus_3, SecpMontElement::zero());
```

```rust
let two = SecpMontElement::from_hex_unchecked("0x2");
assert_eq!(three - two, SecpMontElement::one());
```

```rust
let minus_3_mul_minus_3 = &minus_3 * &minus_3;
let minus_3_squared = minus_3.square();
let minus_3_pow_2 = minus_3.pow(2_u32);
let nine = SecpMontElement::from_hex_unchecked("0x9");
        
assert_eq!(minus_3_mul_minus_3, nine);
assert_eq!(minus_3_squared, nine);
assert_eq!(minus_3_pow_2, nine);
```

This last part shows that we have three ways of computing the square of a number. The first one is using multiplication, the second is squaring and the third one is using the power/exponentiation function `pow`. The `pow` function needs as exponent an `UnsignedInteger`. The most efficient function in this context is `square()`, followed by `mul`.

```rust
let three_inv = three.inv();
assert_eq!(three * three_inv, SecpMontElement::one());
```

Note: if you need to invert several elements, you should use the `inplace_batch_inverse`, since computing field inversion is usually expensive.

If you print the hex representation of three, `three.to_hex()`, you will get `0x300000B73`. This is the Montgomery representation, which is different from the standard form `0x3`. If you perform `three.representative().to_hex()`, it will transform first to standard form, then give `0x3`. Let's look at the output of several functions:
- `three.to_hex()` : `0x300000B73`
- `three.representative().to_hex()` : `0x3`
- `three.to_bytes_be()`: Returns a vector of 32 bytes (256 bits), all of which are `0x0`, except for the last one, `0x3`
- `three.to_bytes_le()`: Returns a vector of 32 bytes (256 bits), all of which are `0x0`, except for the first one, `0x3`
- `SecpMontElement::from_bytes_be(&three.to_bytes_be())`: will return the Montgomery form of the number 3.
- `SecpMontElement::from_hex(&three.to_hex())`: this will not return the Montgomery form of 3!
- `SecpMontElement::from_hex(&three.representative().to_hex())`: this will return the Montgomery form of 3.

## Montgomery arithmetic

Addition and subtraction in Montgomery form follow the same rules as ordinary addition and subtraction over a field. There are different algorithms for multiplication (and squaring):
- Coarsely Integrated Operand Scanning (CIOS)
- Separated Operand Scanning Method (SOS)

Multiplication follows `cios`, unless there are spare bits in the modulus. For that case, multiplication changes to `cios_optimized_for_moduli_with_one_spare_bit`. Squaring uses the `sos_square` method.

Inversion is performed using Algorithm 16 (Binary Euclidean Algorithm) from [Guajardo, Kumar, Paar, Perzl](https://www.sandeep.de/my/papers/2006_ActaApplMath_EfficientSoftFiniteF.pdf).

## Extension fields

In some applications in Cryptography, it may be necessary to work over an *extension field*. For example, to compute pairings we need to work over a larger field. Similarly, in STARKs, when we need to sample a random number, we want to do it from a large set, and we can do this by working with an extension of the original field. What are extension fields? You may have heard about [complex numbers](https://en.wikipedia.org/wiki/Complex_number). We can view them as a pair of real numbers $c = (a, b)$ with a multiplication operation defined as $(a_0 , b_0 ) \times (a_1 , b_1 ) = (a_0 a_1 - b_0 b_1 , a_0 b_1 + a_1 b_0 )$ and the addition as $(a_0 , b_0 ) + (a_1 , b_1 ) = (a_0 + a_1 , b_0 + b_1 )$. We can see real numbers as a subset of $\mathbb{C}$ of the form $(a , 0)$. It is common to introduce the imaginary unit, $i$, and write them also as $a + b i$, with $i^2 = - 1$ (you can check that $(0 , 1)^2 = - 1$). We see that when work with complex numbers of the form $(a , 0)$ this works as ordinary real numbers, and we see that complex numbers extend the real numbers, allowing us to find solutions to equations that cannot be solved over the real numbers. For example, $x^2 + 1 = 0$ does not have a solution over $\mathbb{R}$, but over $\mathbb{C}$ both $i$ and $- i$ solve the equation. We will focus on how to build the complex numbers from the real numbers, and then explain how to adapt this to the setting of finite fields.

We will work with univariate polynomials over the real numbers. A univariate polynomial over the real numbers is an expression in an indeterminate, $x$, with coefficients taking their values over the real numbers, with the following form $p(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_n x^n$. For example, $p(x) = \sqrt{2} + \pi x + 2 x^2 + 5x^3 - 56 x^4$. We can add, subtract and multiply polynomials similar to what we do with integers, which means that the polynomials form a ring. We denote the ring of polynomials over real numbers as $\mathbb{R} [x]$, the ring of polynomials over the integers as $\mathbb{Z} [x]$ and the ring of polynomials over a finite field as $\mathbb{F_p} [x]$. The highest $k$ such that $a_k \neq 0$ is called the degree of the polynomial; for our previous example it is 4. Given polynomials $p(x)$ and $d(x)$, there are polynomials $q(x)$ and $r(x)$ such that $p(x) = d(x)q(x) + r(x)$, with the degree of $r(x)$ less than the degree of $d(x)$ (this in analogous to the integer division with remainder). Given a polynomial $p(x)$, we can define the ring of polynomials modulo $p(x)$, denoted by $\mathbb{R} [x] / p(x)$. Operations in this ring work similar to integers: whenever the degree of the result exceeds or equals the degree of $p(x)$, we take the remainder $r(x)$ of the division between the result and $p(x)$ (basically, $p(x)$ acts like the modulus in finite fields).

Many polynomials can be expressed in terms of lower degree polynomials. For example, $x^2 - 1 = (x + 1) (x - 1)$, $x^3 + 3x^2 + 3x + 1 = (x + 1)(x + 1)(x + 1)$. Over the real numbers, $I(x) = x^2 + 1$ cannot be expressed in terms of lower degree polynomials. We say that $I(x)$ is [irreducible](https://en.wikipedia.org/wiki/Irreducible_polynomial) over $\mathbb{R}$. If we consider the ring modulo an irreducible polynomial, $\mathbb{R} [x] / I(x)$, then $\mathbb{R} [x] / I(x)$ is a field. The degree of $I(x)$ is the degree of the extension. For example, in the case of the real numbers, $\mathbb{R} [x] / (x^2 + 1)$ coincides with our notion of the complex numbers. Every element there has the form $a + b x$. 

Addition (and subtraction) is done the usual way, $(a + b x) + (c + dx) = (a + c) + (b + d)x$. Multiplication is a bit more difficult, $(a + bx) \times (c + dx) = ac + (b c + a d) x + b d x^2$. But this polynomial has equal degree than $x^2 + 1$, so we take the remainder. The result is $ac - bd + (b c + a d)x$ (you can check that $x^2 = - 1$). 

In the case of complex numbers, we don't need to continue extending them further, since we can factor any polynomial over $\mathbb{C}$ (see the [fundamental theorem of algebra](https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra)).

The recipe to build extension fields over finite fields is the same. We will start with the simplest case, when $\mathbb{F_p}$ is a prime field. For example, we are working with $p = 2^{31} - 1$, a Mersenne prime (we can see that $p \equiv 3 \pmod{4}$). We consider the polynomials with coefficients over $\mathbb{F_p} [x]$. It can be shown that $- 1$ has no square root over $\mathbb{F_p}$: this means that there is no $x$ such that $x^2 = - 1$ over $\mathbb{F_p}$. The polynomial $I(x) = x^2 + 1$ is, therefore, irreducible over $\mathbb{F_p}$ (note, $x^2 - 1$ is not always irreducible over arbitrary fields!). We can consider then $\mathbb{F_p} [x] / I(x)$ and the elements there are represented by $a + b x$. One caveat with the case of complex numbers is that the operations involving $a$ and $b$ are the operations of $\mathbb{F_p }$. For example, $(5 + 2^{31} x) + ((2^{31} - 6) - x) = 0 + 0x$. Since the degree of $I(x)$ is $2$, we say that we have a quadratic extension of $\mathbb{F_p}$ and will denote it $\mathbb{F_{ p^2 } }$. You could have chosen a different degree $2$ irreducible polynomial, but we can show that the two extensions are isomorphic. 

You can build other extensions looking for higher-degree irreducible polynomials. For example, if you consider the field $\mathbb{F_2} = \{0 , 1 \}$, the polynomial $x^8 + x^4 + x^3 + x + 1$ is irreducible, and you can define a degree $8$ extension of $\mathbb{F_2}$.

There are different ways in which we can construct higher-degree extensions. For example, we can take our prime field and find an irreducible polynomial of degree $4$ and work with $\mathbb{F_p} / I(x)$. Each element in the field can be represented as $a + b x + c x^2 + dx^3$. We can also use a towered approach: we first find an irreducible polynomial $I(x)$ of degree $2$ and obtain $\mathbb{F_{ p^2 } } = \mathbb{F_p} / I(x)$. Each element is of the form $a + b x$. Since the extension is also a field, we can find an irreducible polynomial over ${F_{ p^2 } }$, $J(y)$, of degree $2$ and consider ${F_{ p^2 } } [y] / J(y)$. Then, each element there is of the form $a^\prime + b^\prime y$, where $a^\prime$ and $b^\prime$ live in ${F_{ p^2 } }$. Since every element in ${F_{ p^2 } }$ is of the form $a_0 + a_1 x$, we get that $a_0 + a_1 x + b_0 y + b_1 x y$. Even though the extensions look different, there is an isomorphism connecting the two. Depending on the application, one form or the other can be more efficient.

While some of the underlying concepts can be difficult to grasp, defining extension fields is simpler in lambdaworks. While we could allow you to define arbitrary extensions, we provide methods to define quadratic and cubic extensions over fields. To define a quadratic extension, you need to implement the following trait:

```rust
pub trait HasQuadraticNonResidue<F: IsField> {
    fn residue() -> FieldElement<F>;
}
```

Here, we assume that you want to define a quadratic extension of the form $x^2 - \mathrm{residue}$, where $\mathrm{residue}$ is not a square over your field. In the case of the Mersenne prime $2^{31} - 1$, $\mathrm{residue} = - 1$ (you could use other quadratic non-residues, but this can lead to slower field extension operations). Similarly, for cubic extensions we have

```rust
pub trait HasCubicNonResidue<F: IsField> {
    /// This function must return an element that is not a cube in Fp,
    /// that is, a cubic non-residue.
    fn residue() -> FieldElement<F>;
}
```

For example, the following code defines the default quadratic extension for BabyBear:

```rust
pub type QuadraticBabybearField =
    QuadraticExtensionField<Babybear31PrimeField, Babybear31PrimeField>;

impl HasQuadraticNonResidue<Babybear31PrimeField> for Babybear31PrimeField {
    fn residue() -> FieldElement<Babybear31PrimeField> {
        -FieldElement::one()
    }
}

/// Field element type for the quadratic extension of Babybear
pub type QuadraticBabybearFieldElement =
    QuadraticExtensionFieldElement<Babybear31PrimeField, Babybear31PrimeField>;
```

You can also create a separate quadratic extension by implementing the `IsField` trait for the quadratic extension,

```rust
#[derive(Clone, Debug)]
pub struct BLS12381FieldModulus;
impl IsModulus<U384> for BLS12381FieldModulus {
    const MODULUS: U384 = BLS12381_PRIME_FIELD_ORDER;
}

pub type BLS12381PrimeField = MontgomeryBackendPrimeField<BLS12381FieldModulus, 6>;

//////////////////
#[derive(Clone, Debug)]
pub struct Degree2ExtensionField;

impl IsField for Degree2ExtensionField {
    type BaseType = [FieldElement<BLS12381PrimeField>; 2];
    /// Returns the component wise addition of `a` and `b`
    fn add(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        [&a[0] + &b[0], &a[1] + &b[1]]
    }

    /// Returns the multiplication of `a` and `b` using the following
    /// equation:
    /// (a0 + a1 * t) * (b0 + b1 * t) = a0 * b0 + a1 * b1 * Self::residue() + (a0 * b1 + a1 * b0) * t
    /// where `t.pow(2)` equals `Q::residue()`.
    fn mul(a: &Self::BaseType, b: &Self::BaseType) -> Self::BaseType {
        let a0b0 = &a[0] * &b[0];
        let a1b1 = &a[1] * &b[1];
        let z = (&a[0] + &a[1]) * (&b[0] + &b[1]);
        [&a0b0 - &a1b1, z - a0b0 - a1b1]
    }
}
```

You should then implement the operations for the field, such as addition, multiplication, subtraction, inversion and so on. This is more convenient if you can avoid doing extra operations and defining the residue. You can also optimize the operations between elements of the base field and the extension field by implementing the trait `IsSubFieldOf`.

```rust
impl IsSubFieldOf<Degree2ExtensionField> for BLS12381PrimeField {
    fn mul(
        a: &Self::BaseType,
        b: &<Degree2ExtensionField as IsField>::BaseType,
    ) -> <Degree2ExtensionField as IsField>::BaseType {
        let c0 = FieldElement::from_raw(<Self as IsField>::mul(a, b[0].value()));
        let c1 = FieldElement::from_raw(<Self as IsField>::mul(a, b[1].value()));
        [c0, c1]
    }
}
```

Once you have the quadratic extension, you can build another extension (tower approach). For example, to define a degree 3 extension field over the quadratic extension of the BLS12-381 scalar field, we have

```rust
#[derive(Debug, Clone)]
pub struct LevelTwoResidue;
impl HasCubicNonResidue<Degree2ExtensionField> for LevelTwoResidue {
    fn residue() -> FieldElement<Degree2ExtensionField> {
        FieldElement::new([
            FieldElement::new(U384::from("1")),
            FieldElement::new(U384::from("1")),
        ])
    }
}

pub type Degree6ExtensionField = CubicExtensionField<Degree2ExtensionField, LevelTwoResidue>;
```

This defines a 6th degree extension over the scalar field of BLS12-381. We only need to define the cubic (non) residue, which is an element of $\mathbb{F_{ p^2 } }$. 

## Exercises

- Define the base field of the Ed25519 elliptic curve, defined by the prime $p$.
- Check whether $- 1$ is a quadratic residue.
- Compute $100^{65537} \pmod p$
- Define a degree 4 extension of the BabyBear field.

## References

- [An introduction to mathematical cryptography](https://books.google.com.ar/books/about/An_Introduction_to_Mathematical_Cryptogr.html?id=XLY9AnfDhsYC&source=kp_book_description&redir_esc=y)
- [High-Speed Algorithms & Architectures For Number-Theoretic Cryptosystems](https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf)
- [Developer math survival kit](https://blog.lambdaclass.com/math-survival-kit-for-developers/)
- [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf)
- [Guajardo, Kumar, Paar, Perzl - Efficient software implementation of finite fields with applications to Cryptography](https://www.sandeep.de/my/papers/2006_ActaApplMath_EfficientSoftFiniteF.pdf)