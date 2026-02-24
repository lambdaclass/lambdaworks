//! Additive NTT over binary field subspaces.
//!
//! Implements polynomial evaluation and interpolation over GF(2)-linear subspaces
//! of GF(2^128). This is the foundation for Reed-Solomon encoding and FRI folding
//! in the Binius proof system.
//!
//! # Subspace structure
//!
//! The evaluation domain is a GF(2)-linear subspace W of GF(2^128). We use the
//! standard GF(2)-basis {e_0 = 1, e_1 = 2, e_2 = 4, ...} where e_i has value 2^i.
//! A subspace of dimension k is span(e_0, ..., e_{k-1}), which consists of all
//! field elements whose u128 representation has at most k low bits set, i.e.,
//! the set {0, 1, 2, ..., 2^k - 1}.
//!
//! # Key operations
//!
//! - **Forward NTT**: Evaluate a polynomial at all points of a subspace
//! - **Inverse NTT**: Interpolate a polynomial from evaluations at subspace points
//! - **RS encoding**: Evaluate a polynomial at a larger subspace (for FRI)
//!
//! # References
//!
//! - LCH14: <https://arxiv.org/pdf/1708.09746>
//! - FRI-Binius: <https://eprint.iacr.org/2024/504>
//! - LambdaClass blog: <https://blog.lambdaclass.com/additive-fft-background/>

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::binary::tower_field::BinaryTowerField128;
use lambdaworks_math::polynomial::Polynomial;

type FE = FieldElement<BinaryTowerField128>;

/// Returns the GF(2)-standard basis vectors for a subspace of the given dimension.
///
/// The basis consists of {e_0, e_1, ..., e_{dimension-1}} where e_i = 2^i.
/// These are the canonical GF(2)-basis elements of GF(2^128).
///
/// # Panics
///
/// Panics if `dimension > 128` (GF(2^128) has at most 128 basis elements).
pub fn tower_basis(dimension: usize) -> Vec<FE> {
    assert!(dimension <= 128, "dimension must be <= 128");
    (0..dimension).map(|i| FE::new(1u128 << i)).collect()
}

/// Enumerates all 2^k elements of the GF(2)-linear subspace spanned by `basis`.
///
/// Elements are ordered by their "index": element i is the sum of basis[j]
/// for each bit j set in i. With the standard basis, this gives {0, 1, 2, 3, ...}.
pub fn enumerate_subspace(basis: &[FE]) -> Vec<FE> {
    let k = basis.len();
    let n = 1usize << k;
    let mut points = vec![FE::zero(); n];
    for i in 1..n {
        // Gray-code style: XOR with basis[j] where j is the lowest set bit of i
        let j = i.trailing_zeros() as usize;
        points[i] = points[i ^ (1 << j)] + basis[j];
    }
    points
}

/// Evaluates the subspace vanishing polynomial V_W(x) for the subspace W spanned
/// by the given basis vectors.
///
/// V_W(x) = prod_{w in W} (x + w), which is GF(2)-linear: V(a+b) = V(a) + V(b).
///
/// Uses the recursive identity:
///   V_{W+span(beta)}(x) = V_W(x) * (V_W(x) + V_W(beta))
///
/// where V_W(x+beta) = V_W(x) + V_W(beta) by linearity.
///
/// Note: this evaluates V_W at a single point x, not the polynomial itself.
pub fn subspace_vanishing_poly(x: &FE, basis: &[FE]) -> FE {
    if basis.is_empty() {
        return *x;
    }
    // Start with V_0(x) = x (vanishing poly of the trivial subspace {0})
    // At each step: V_{i+1}(x) = V_i(x) * (V_i(x) + V_i(basis[i]))
    // We need V_i(basis[i]) at each step, computed by the same recursion.
    let mut v_x = *x;
    for (i, beta) in basis.iter().enumerate() {
        // Compute V_i(beta) where V_i is vanishing poly of span(basis[0]..basis[i-1])
        let v_beta = subspace_vanishing_poly(beta, &basis[..i]);
        v_x = v_x * (v_x + v_beta);
    }
    v_x
}

/// Forward additive NTT: evaluates a polynomial at all points of a GF(2)-linear subspace.
///
/// Given coefficients of a polynomial p(x) = c_0 + c_1*x + ... + c_{n-1}*x^{n-1},
/// computes p(w) for all w in the subspace of dimension `log_size`.
///
/// The evaluation domain is {0, 1, 2, ..., 2^log_size - 1} viewed as elements
/// of GF(2^128).
///
/// After this call, `coeffs` is replaced by the evaluations.
pub fn additive_ntt(coeffs: &mut [FE], log_size: usize) {
    let n = 1usize << log_size;
    assert_eq!(coeffs.len(), n, "input length must be 2^log_size");

    let basis = tower_basis(log_size);
    let points = enumerate_subspace(&basis);

    let poly = Polynomial::new(coeffs);
    let evals: Vec<FE> = points.iter().map(|x| poly.evaluate(x)).collect();

    coeffs.copy_from_slice(&evals);
}

/// Inverse additive NTT: interpolates a polynomial from evaluations at subspace points.
///
/// Given evaluations evals[i] = p(i) for i = 0, ..., 2^log_size - 1, recovers the
/// polynomial coefficients.
///
/// After this call, `evals` is replaced by the polynomial coefficients.
pub fn inverse_additive_ntt(evals: &mut [FE], log_size: usize) {
    let n = 1usize << log_size;
    assert_eq!(evals.len(), n, "input length must be 2^log_size");

    let basis = tower_basis(log_size);
    let points = enumerate_subspace(&basis);

    let poly = Polynomial::interpolate(&points, evals)
        .expect("interpolation should succeed: subspace points are distinct");

    // Copy coefficients back, zero-padding if needed
    let mut coefficients = poly.coefficients;
    coefficients.resize(n, FE::zero());
    evals.copy_from_slice(&coefficients);
}

/// Reed-Solomon encoding: evaluates a message polynomial at a larger subspace.
///
/// Given a message of length 2^log_message_size (polynomial coefficients),
/// evaluates the polynomial at all 2^(log_message_size + log_blowup) points
/// of a larger subspace.
///
/// Returns the codeword of length 2^(log_message_size + log_blowup).
pub fn rs_encode(message: &[FE], log_blowup: usize) -> Vec<FE> {
    let log_msg = message.len().trailing_zeros() as usize;
    assert_eq!(
        message.len(),
        1 << log_msg,
        "message length must be a power of 2"
    );

    let log_codeword = log_msg + log_blowup;

    let basis = tower_basis(log_codeword);
    let points = enumerate_subspace(&basis);

    let poly = Polynomial::new(message);
    points.iter().map(|x| poly.evaluate(x)).collect()
}

/// Computes the folding of a codeword for binary FRI.
///
/// Given evaluations on a domain (list of field elements), folds the codeword
/// by pairing consecutive elements:
///   f_folded(V(w)) = f(w) + (w + alpha) * (f(w) + f(w + basis[0]))
///
/// where V(x) = x^2 + x is the vanishing polynomial of {0, 1}.
///
/// The `domain` contains the actual evaluation points of the codeword. After each
/// fold, the domain transforms via V(x) = x^2 + x, so subsequent rounds must use
/// the updated domain — NOT integer indices.
///
/// Returns (folded_codeword, new_domain).
pub fn fold_codeword(codeword: &[FE], challenge: &FE, domain: &[FE]) -> (Vec<FE>, Vec<FE>) {
    assert_eq!(
        codeword.len(),
        domain.len(),
        "codeword and domain must have same length"
    );
    assert!(codeword.len() >= 2, "need at least 2 elements to fold");

    let half = codeword.len() / 2;
    let mut folded = Vec::with_capacity(half);
    let mut new_domain = Vec::with_capacity(half);

    // Pair elements (2i, 2i+1) which differ by the first basis element of the current subspace
    for i in 0..half {
        let f_w = codeword[2 * i]; // f(w) where w is the "even" element
        let f_w1 = codeword[2 * i + 1]; // f(w + basis[0])
        let w = domain[2 * i]; // actual evaluation point

        // f_folded = f(w) + (w + alpha) * (f(w) + f(w + basis[0]))
        let diff = f_w + f_w1;
        let folded_val = f_w + (*challenge + w) * diff;
        folded.push(folded_val);

        // New domain point: V(w) = w^2 + w
        new_domain.push(w * w + w);
    }

    (folded, new_domain)
}

/// Builds the initial evaluation domain {0, 1, 2, ..., 2^log_size - 1}.
pub fn initial_domain(log_size: usize) -> Vec<FE> {
    let n = 1usize << log_size;
    (0..n).map(|i| FE::new(i as u128)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tower_basis_dimension_0() {
        let basis = tower_basis(0);
        assert!(basis.is_empty());
    }

    #[test]
    fn test_tower_basis_dimension_3() {
        let basis = tower_basis(3);
        assert_eq!(basis.len(), 3);
        assert_eq!(*basis[0].value(), 1u128);
        assert_eq!(*basis[1].value(), 2u128);
        assert_eq!(*basis[2].value(), 4u128);
    }

    #[test]
    fn test_enumerate_subspace_dim_0() {
        let basis = tower_basis(0);
        let points = enumerate_subspace(&basis);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0], FE::zero());
    }

    #[test]
    fn test_enumerate_subspace_dim_2() {
        let basis = tower_basis(2);
        let points = enumerate_subspace(&basis);
        assert_eq!(points.len(), 4);
        // {0, 1, 2, 3} since basis is {1, 2}
        assert_eq!(*points[0].value(), 0u128);
        assert_eq!(*points[1].value(), 1u128);
        assert_eq!(*points[2].value(), 2u128);
        assert_eq!(*points[3].value(), 3u128);
    }

    #[test]
    fn test_enumerate_subspace_dim_3() {
        let basis = tower_basis(3);
        let points = enumerate_subspace(&basis);
        assert_eq!(points.len(), 8);
        // Should be {0, 1, 2, 3, 4, 5, 6, 7}
        for (i, point) in points.iter().enumerate().take(8) {
            assert_eq!(*point.value(), i as u128);
        }
    }

    #[test]
    fn test_vanishing_poly_single_basis() {
        // V_{span(1)}(x) = x * (x + 1) for the subspace {0, 1}
        // V(0) = 0 * 1 = 0
        // V(1) = 1 * 0 = 0
        // V(2) should be nonzero
        let basis = tower_basis(1);
        assert_eq!(subspace_vanishing_poly(&FE::new(0u128), &basis), FE::zero());
        assert_eq!(subspace_vanishing_poly(&FE::new(1u128), &basis), FE::zero());

        let v2 = subspace_vanishing_poly(&FE::new(2u128), &basis);
        assert_ne!(v2, FE::zero());
    }

    #[test]
    fn test_vanishing_poly_linearity() {
        // V_W is GF(2)-linear: V(a + b) = V(a) + V(b)
        let basis = tower_basis(2);
        let a = FE::new(5u128);
        let b = FE::new(13u128);
        let v_a = subspace_vanishing_poly(&a, &basis);
        let v_b = subspace_vanishing_poly(&b, &basis);
        let v_ab = subspace_vanishing_poly(&(a + b), &basis);
        assert_eq!(v_ab, v_a + v_b);
    }

    #[test]
    fn test_vanishing_poly_vanishes_on_subspace() {
        let basis = tower_basis(3);
        let points = enumerate_subspace(&basis);
        for point in &points {
            let v = subspace_vanishing_poly(point, &basis);
            assert_eq!(
                v,
                FE::zero(),
                "V_W should vanish on subspace element {}",
                point.value()
            );
        }
    }

    #[test]
    fn test_ntt_roundtrip() {
        // Forward NTT then inverse NTT should recover original coefficients
        let log_size = 3;
        let n = 1usize << log_size;
        let original: Vec<FE> = (0..n).map(|i| FE::new((i + 1) as u128)).collect();

        let mut data = original.clone();
        additive_ntt(&mut data, log_size);
        inverse_additive_ntt(&mut data, log_size);

        for i in 0..n {
            assert_eq!(
                data[i],
                original[i],
                "Roundtrip failed at index {i}: got {:?}, expected {:?}",
                data[i].value(),
                original[i].value()
            );
        }
    }

    #[test]
    fn test_ntt_consistency_with_direct_eval() {
        // NTT output should match direct polynomial evaluation
        let log_size = 3;
        let n = 1usize << log_size;
        let coeffs: Vec<FE> = (0..n).map(|i| FE::new((i * 3 + 7) as u128)).collect();

        let poly = Polynomial::new(&coeffs);

        let mut ntt_result = coeffs.clone();
        additive_ntt(&mut ntt_result, log_size);

        // Check each evaluation manually
        for (i, ntt_val) in ntt_result.iter().enumerate().take(n) {
            let x = FE::new(i as u128);
            let direct = poly.evaluate(&x);
            assert_eq!(
                *ntt_val, direct,
                "NTT disagrees with direct eval at point {i}"
            );
        }
    }

    #[test]
    fn test_rs_encode_preserves_message() {
        // RS encoding at message points should give the same evaluations
        let log_blowup = 1;
        let msg: Vec<FE> = (0..4).map(|i| FE::new((i + 1) as u128)).collect();

        let codeword = rs_encode(&msg, log_blowup);
        assert_eq!(codeword.len(), 8); // 2^(2+1) = 8

        // Codeword entries should match direct polynomial evaluation
        let poly = Polynomial::new(&msg);
        for (i, cw_val) in codeword.iter().enumerate().take(8) {
            let x = FE::new(i as u128);
            let expected = poly.evaluate(&x);
            assert_eq!(*cw_val, expected, "RS codeword mismatch at index {i}");
        }
    }

    #[test]
    fn test_rs_encode_degree_check() {
        // A polynomial of degree < n should produce a codeword that, when
        // interpolated, gives back a polynomial of degree < n
        let log_blowup = 1;
        let msg: Vec<FE> = vec![
            FE::new(1u128),
            FE::new(3u128),
            FE::new(0u128),
            FE::new(5u128),
        ];

        let codeword = rs_encode(&msg, log_blowup);
        assert_eq!(codeword.len(), 8);

        // Interpolate the codeword
        let points: Vec<FE> = (0..8).map(|i| FE::new(i as u128)).collect();
        let poly = Polynomial::interpolate(&points, &codeword).unwrap();

        // The interpolated polynomial should have degree < 4 (= message size)
        assert!(
            poly.coefficients.len() <= 4,
            "RS encoded polynomial should have degree < message size, got {}",
            poly.coefficients.len()
        );
    }

    #[test]
    fn test_fold_codeword_halves_size() {
        let log_size = 3;
        let n = 1usize << log_size;
        let codeword: Vec<FE> = (0..n).map(|i| FE::new((i + 1) as u128)).collect();
        let domain = initial_domain(log_size);
        let challenge = FE::new(42u128);

        let (folded, new_domain) = fold_codeword(&codeword, &challenge, &domain);
        assert_eq!(folded.len(), n / 2);
        assert_eq!(new_domain.len(), n / 2);
    }

    #[test]
    fn test_fold_preserves_low_degree() {
        // If codeword is an RS encoding of a polynomial of degree < 2^k,
        // then after folding, the result should be consistent with a polynomial
        // of degree < 2^{k-1}.
        let log_blowup = 1;
        let log_codeword = 2 + log_blowup; // degree < 4, domain size 8

        // Create a polynomial and RS-encode it
        let coeffs: Vec<FE> = vec![
            FE::new(3u128),
            FE::new(7u128),
            FE::new(1u128),
            FE::new(5u128),
        ];
        let codeword = rs_encode(&coeffs, log_blowup);
        let domain = initial_domain(log_codeword);

        let challenge = FE::new(42u128);
        let (folded, new_domain) = fold_codeword(&codeword, &challenge, &domain);

        // The folded codeword should be consistent with a polynomial of degree < 2
        // Interpolate using the actual folded domain points
        let folded_poly = Polynomial::interpolate(&new_domain, &folded).unwrap();

        // The folded polynomial should have degree < 2 = 2^{log_msg - 1}
        assert!(
            folded_poly.coefficients.len() <= 2,
            "Folded polynomial should have degree < 2, got degree {}",
            folded_poly.coefficients.len().saturating_sub(1)
        );
    }

    #[test]
    fn test_fold_to_constant() {
        // Folding a degree-3 polynomial twice should produce a constant
        let log_blowup = 1;
        let log_codeword = 2 + log_blowup; // degree < 4, domain size 8

        let coeffs: Vec<FE> = vec![
            FE::new(5u128),
            FE::new(3u128),
            FE::new(7u128),
            FE::new(11u128),
        ];
        let codeword = rs_encode(&coeffs, log_blowup);
        let domain = initial_domain(log_codeword);

        // First fold
        let challenge1 = FE::new(42u128);
        let (folded1, domain1) = fold_codeword(&codeword, &challenge1, &domain);
        assert_eq!(folded1.len(), 4);

        // Second fold
        let challenge2 = FE::new(99u128);
        let (folded2, _domain2) = fold_codeword(&folded1, &challenge2, &domain1);
        assert_eq!(folded2.len(), 2);

        // After 2 folds of a degree-3 polynomial, result should be constant
        assert_eq!(
            folded2[0],
            folded2[1],
            "Folded polynomial should be constant: folded2[0] = {:?}, folded2[1] = {:?}",
            folded2[0].value(),
            folded2[1].value()
        );
    }
}
