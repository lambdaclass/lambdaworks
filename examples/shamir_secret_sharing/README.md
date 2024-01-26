# Shamir's Secret Sharing

## Usage example

```rust
// Definition of the secret
// Creation of 6 shares
// 3 shares will be used to recover the secret
let sss = ShamirSecretSharing {
    secret: FE::new(1234),
    n: 6,
    k: 3,
};

// Polynomial calculation
let polynomial = sss.calculate_polynomial();

// Produce shares
let shares = sss.generating_shares(polynomial.clone());

// Specify the x and y coordinates of the shares to use
let shares_to_use_x = vec![shares.x[1], shares.x[3], shares.x[4]];
let shares_to_use_y = vec![shares.y[1], shares.y[3], shares.y[4]];

// Interpolation
let poly_2 = sss.reconstructing(shares_to_use_x, shares_to_use_y);

// Recover the free coefficient of the polynomial
let secret_recovered = sss.recover(&poly_2);

// Verifications
assert_eq!(polynomial, poly_2);
assert_eq!(sss.secret, secret_recovered);
```
