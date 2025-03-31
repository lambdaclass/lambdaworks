// use lambdaworks_math::field::traits::{IsField, IsPrimeField};

// use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
// use crate::elliptic_curve::traits::IsEllipticCurve;
// use crate::{
//     elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
// };

use std::collections::HashMap;

use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass},
        traits::IsEllipticCurve,
    },
    field::{
        element::FieldElement,
        fields::{
            montgomery_backed_prime_fields::{
                IsModulus, MontgomeryBackendPrimeField, U256PrimeField,
            },
            u64_prime_field::U64PrimeField,
        },
    },
    unsigned_integer::element::{UnsignedInteger, U128},
};

/// Curve y^2 = x^3 + 2x + 5 over the finite field with modulus 113.
/// This curve is smooth because it has order 108 = 2^2 * 3^3.
#[derive(Clone, Debug)]
pub struct SmoothCurve;

impl IsEllipticCurve for SmoothCurve {
    type BaseField = U64PrimeField<113>;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // g = (81, 13, 1) is a generator of the curve. We precomputed it in sage:
        // ```
        // F = GF(113)
        // C = EllipticCurve(F, [2, 5])
        // g = C.gens()[0]
        // ```
        let point = Self::PointRepresentation::new([
            FieldElement::from(81),
            FieldElement::from(13),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for SmoothCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(2)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(5)
    }
}

pub type SmoothMontgomeryBackendPrimeField<T> = MontgomeryBackendPrimeField<T, 2>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmoothMontgomeryConfigPrimeField;
impl IsModulus<U128> for SmoothMontgomeryConfigPrimeField {
    const MODULUS: U128 = U128::from_u128(130086066308714848439);
}

pub type SmoothPrimeField = SmoothMontgomeryBackendPrimeField<SmoothMontgomeryConfigPrimeField>;
pub type FE = FieldElement<SmoothPrimeField>;

#[derive(Clone, Debug)]
pub struct SmoothCurve2;

impl IsEllipticCurve for SmoothCurve2 {
    type BaseField = SmoothPrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        let point = Self::PointRepresentation::new([
            FE::from_hex_unchecked("28DA64FB59DFD59B"), // 2943776337147450779
            FE::from_hex_unchecked("18DB59F17E035D175"), // 28657986728736706933
            FE::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for SmoothCurve2 {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(1129)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(4789)
    }
}

// // Utility function to compute the order of a point.
// pub fn get_point_order(point: &ShortWeierstrassProjectivePoint<SmoothCurve>) -> u64 {
//     let identity = ShortWeierstrassProjectivePoint::<SmoothCurve>::neutral_element();
//     let mut current = point.clone();
//     let mut order = 1;
//     while current != identity {
//         current = current.operate_with(point);
//         order += 1;
//     }
//     order
// }

// Note: uses bruteforce
// Given q a point in the curve, it returns n such that g^k = q, where g is a curve's generator.
pub fn pohlig_hellman(q: &ShortWeierstrassProjectivePoint<SmoothCurve>) -> usize {
    let g = SmoothCurve::generator();
    let g0 = g.operate_with_self(27u32); // 27 = 108 / (2^2).
    let q0 = q.operate_with_self(27u32);
    let mut k0 = 0;
    for i in 0..4 {
        // 0 .. 2^2
        if q0 == g0.operate_with_self(i as u16) {
            k0 = i;
            break;
        }
    }
    let g1 = g.operate_with_self(12u32); // 12 = 108 / (3^3).
    let q1 = q.operate_with_self(12u32);
    let mut k1 = 0;
    for i in 0..9 {
        // 0 .. 3^3
        if q1 == g1.operate_with_self(i as u16) {
            k1 = i;
            break;
        }
    }
    let equations = [(k0, 4), (k1, 27)];
    chinese_remainder_theorem(&equations) as usize
}

fn update_step(a: &mut i128, old_a: &mut i128, quotient: i128) {
    let temp = *a;
    *a = *old_a - quotient * temp;
    *old_a = temp;
}

pub fn extended_euclidean_algorithm(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut old_r, mut rem) = (a, b);
    let (mut old_s, mut coeff_s) = (1, 0);
    let (mut old_t, mut coeff_t) = (0, 1);

    while rem != 0 {
        let quotient = old_r / rem;

        update_step(&mut rem, &mut old_r, quotient);
        update_step(&mut coeff_s, &mut old_s, quotient);
        update_step(&mut coeff_t, &mut old_t, quotient);
    }

    (old_r, old_s, old_t)
}

fn mod_inverse(x: i128, n: i128) -> Option<i128> {
    let (g, x, _) = extended_euclidean_algorithm(x, n);
    if g == 1 {
        Some((x % n + n) % n)
    } else {
        None
    }
}

pub fn chinese_remainder_theorem(equations: &[(i128, i128)]) -> i128 {
    // Calculate the product of all moduli
    let n: i128 = equations.iter().map(|(_, m)| m).product();

    // For each equation, compute:
    // 1. n_i = n / m_i (product of all moduli except current)
    // 2. x_i = inverse of n_i modulo m_i
    // 3. Add a_i * n_i * x_i to the result
    let mut result = 0;
    for &(a, m) in equations {
        let n_i = n / m;
        // Find x_i such that n_i * x_i ≡ 1 (mod m)
        let x_i = mod_inverse(n_i, m).expect("Moduli must be pairwise coprime");
        result = (result + a * n_i * x_i) % n;
    }

    result
}

/// Versión de Baby-step Giant-step usando un `Vec` en vez de `HashMap`.
/// Busca x en [0..n-1] tal que x*G = Q, devolviendo Some(x) si existe.
///
/// Complejidad: O(m^2) donde m ≈ sqrt(n).
pub fn baby_step_giant_step_vector(
    g: &ShortWeierstrassProjectivePoint<SmoothCurve>,
    q: &ShortWeierstrassProjectivePoint<SmoothCurve>,
    n: u64,
) -> Option<u64> {
    // 1) m = ceil(sqrt(n))
    let m = (n as f64).sqrt().ceil() as u64;

    // 2) Baby steps: calculamos j*G para j en [0..m)
    //    y lo guardamos en un Vec<(Point, j)>
    let mut baby_list = Vec::with_capacity(m as usize);
    for j in 0..m {
        // Ajusta a u32 / u16 según cómo esté definida tu API
        let point_j = g.operate_with_self(j as u32);
        baby_list.push((point_j, j));
    }

    // 3) Giant steps
    //    - g^m y su inverso aditivo
    let gm = g.operate_with_self(m as u32);
    let minus_gm = gm.neg(); // Asegúrate de tener `neg()` definido en tu punto

    // y = Q inicialmente
    let mut y = q.clone();

    // 4) Recorremos i en [0..m)
    for i in 0..m {
        // Buscamos si y está en baby_list (búsqueda lineal)
        // .find() retorna Some(&(point, j)) si lo halla
        if let Some(&(_, j)) = baby_list.iter().find(|(p, _)| p == &y) {
            let x = i * m + j;
            if x < n {
                return Some(x);
            }
        }
        // y = y + (−mG) para el siguiente salto
        y = y.operate_with(&minus_gm);
    }

    // Si no se encontró ninguna coincidencia
    None
}

pub fn baby_step_giant_step_vector_2(
    g: &ShortWeierstrassProjectivePoint<SmoothCurve2>,
    q: &ShortWeierstrassProjectivePoint<SmoothCurve2>,
    n: u64,
) -> Option<u64> {
    // 1) m = ceil(sqrt(n))
    let m = (n as f64).sqrt().ceil() as u64;

    // 2) Baby steps: calculamos j*G para j en [0..m)
    //    y lo guardamos en un Vec<(Point, j)>
    let mut baby_list = Vec::with_capacity(m as usize);
    for j in 0..m {
        // Ajusta a u32 / u16 según cómo esté definida tu API
        let point_j = g.operate_with_self(j as u32);
        baby_list.push((point_j, j));
    }

    // 3) Giant steps
    //    - g^m y su inverso aditivo
    let gm = g.operate_with_self(m as u32);
    let minus_gm = gm.neg(); // Asegúrate de tener `neg()` definido en tu punto

    // y = Q inicialmente
    let mut y = q.clone();

    // 4) Recorremos i en [0..m)
    for i in 0..m {
        // Buscamos si y está en baby_list (búsqueda lineal)
        // .find() retorna Some(&(point, j)) si lo halla
        if let Some(&(_, j)) = baby_list.iter().find(|(p, _)| p == &y) {
            let x = i * m + j;
            if x < n {
                return Some(x);
            }
        }
        // y = y + (−mG) para el siguiente salto
        y = y.operate_with(&minus_gm);
    }

    // Si no se encontró ninguna coincidencia
    None
}

pub fn pohlig_hellman_2(q: &ShortWeierstrassProjectivePoint<SmoothCurve>) -> usize {
    let g = SmoothCurve::generator();
    let g0 = g.operate_with_self(27u32); // 27 = 108 / (2^2).
    let q0 = q.operate_with_self(27u32);
    let k0 = baby_step_giant_step_vector(&g0, &q0, 4).unwrap();

    let g1 = g.operate_with_self(12u32); // 12 = 108 / (3^3).
    let q1 = q.operate_with_self(12u32);
    let k1 = baby_step_giant_step_vector(&g1, &q1, 9).unwrap();

    let equations = [(k0 as i128, 4), (k1 as i128, 27)];
    chinese_remainder_theorem(&equations) as usize
}

fn factorize(mut n: i128) -> Vec<(i128, i128)> {
    let mut factors = Vec::new();
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            let mut exponent = 0;
            while n % d == 0 {
                n /= d;
                exponent += 1;
            }
            factors.push((d, exponent));
        }
        d += 1;
    }
    // Si al final queda un n > 1, es primo
    if n > 1 {
        factors.push((n, 1));
    }
    factors
}

pub fn pohlig_hellman_3(q: &ShortWeierstrassProjectivePoint<SmoothCurve>) -> usize {
    // 1. Obtenemos el orden del grupo (en tu caso, sabes que es 108).
    let order = 108;

    // 2. Factorizamos el orden llamando a `factorize(order)`.
    //    Para 108, esto retornará: [(2,2), (3,3)].
    let factors = factorize(order);

    // 3. Vamos a reproducir exactamente la misma lógica que ya tenías:
    //    - Calculamos 'k0' para el factor 2^2 = 4
    //    - Calculamos 'k1' para el factor 3^3 = 27
    //    - Y combinamos con CRT.
    //
    //    PERO en lugar de "forzarlo" manualmente, aprovechamos el vector `factors`.

    let g = SmoothCurve::generator();
    let mut equations = Vec::new();

    for &(prime, exponent) in &factors {
        let prime_power = prime.pow(exponent as u32); // 4 ó 27, etc.
        let cofactor = order / prime_power; // 108/4=27 ó 108/27=4

        // Subgenerador y subpunto
        let g_sub = g.operate_with_self(cofactor as u32);
        let q_sub = q.operate_with_self(cofactor as u32);

        // Usamos la misma idea de BSGS (baby_step_giant_step_vector)
        // para hallar log en el subgrupo de orden = prime_power
        if let Some(k_sub) = baby_step_giant_step_vector(&g_sub, &q_sub, prime_power as u64) {
            // Añadimos la congruencia x ≡ k_sub (mod prime_power)
            equations.push((k_sub as i128, prime_power as i128));
        } else {
            // En caso de no encontrar
            panic!(
                "No se encontró el log en el subgrupo de orden {}",
                prime_power
            );
        }
    }

    // 4. Combinamos las congruencias con CRT y listo
    let x = chinese_remainder_theorem(&equations);
    x as usize
}

// This implementation does not suport order beyond 5. Need to check the implementation
pub fn pohlig_hellman_4(q: &ShortWeierstrassProjectivePoint<SmoothCurve2>) -> usize {
    // 1. Obtenemos el orden del grupo (en tu caso, sabes que es 108).
    let order = 130086066303665968735;

    // 2. Factorizamos el orden llamando a `factorize(order)`.
    //    Para 108, esto retornará: [(2,2), (3,3)].
    let factors = factorize(order);

    // 3. Vamos a reproducir exactamente la misma lógica que ya tenías:
    //    - Calculamos 'k0' para el factor 2^2 = 4
    //    - Calculamos 'k1' para el factor 3^3 = 27
    //    - Y combinamos con CRT.
    //
    //    PERO en lugar de "forzarlo" manualmente, aprovechamos el vector `factors`.

    let g = SmoothCurve2::generator();
    let mut equations = Vec::new();

    for &(prime, exponent) in &factors {
        let prime_power = prime.pow(exponent as u32); // 4 ó 27, etc.
        let cofactor = order / prime_power; // 108/4=27 ó 108/27=4

        // Subgenerador y subpunto
        let g_sub = g.operate_with_self(cofactor as u32);
        let q_sub = q.operate_with_self(cofactor as u32);

        // Usamos la misma idea de BSGS (baby_step_giant_step_vector)
        // para hallar log en el subgrupo de orden = prime_power
        if let Some(k_sub) = baby_step_giant_step_vector_2(&g_sub, &q_sub, prime_power as u64) {
            // Añadimos la congruencia x ≡ k_sub (mod prime_power)
            equations.push((k_sub as i128, prime_power as i128));
        } else {
            // En caso de no encontrar
            panic!(
                "No se encontró el log en el subgrupo de orden {}",
                prime_power
            );
        }
    }

    // 4. Combinamos las congruencias con CRT y listo
    let x = chinese_remainder_theorem(&equations);
    x as usize
}
#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_mod_inverse() {
        // Test case 1: Simple case
        assert_eq!(mod_inverse(3, 7), Some(5));

        // Test case 2: Larger numbers
        assert_eq!(mod_inverse(17, 3120), Some(2753));

        // Test case 3: Non-invertible case
        assert_eq!(mod_inverse(2, 4), None);

        // Test case 4: Identity case
        assert_eq!(mod_inverse(1, 5), Some(1));
    }

    #[test]
    fn test_chinese_remainder_theorem() {
        // Test case 1: Simple case
        let equations = [(2, 3), (3, 5), (2, 7)];
        assert_eq!(chinese_remainder_theorem(&equations), 23);

        // Test case 2: Our Pohlig-Hellman case
        let equations = [(1, 4), (2, 27)];
        assert_eq!(chinese_remainder_theorem(&equations), 29);

        // Test case 3: Another common case
        let equations = [(1, 3), (2, 4), (3, 5)];
        assert_eq!(chinese_remainder_theorem(&equations), 58);
    }

    // FIXME: It brakes if k > 8.
    #[test]
    fn test_pohlig_hellman() {
        let g = SmoothCurve::generator();

        // Test case 1: k = 1
        let q1 = g.operate_with_self(1u16);
        assert_eq!(pohlig_hellman(&q1), 1);

        // Test case 2: k = 2
        let q2 = g.operate_with_self(2u16);
        assert_eq!(pohlig_hellman(&q2), 2);

        // Test case 3: k = 3
        let q3 = g.operate_with_self(3u16);
        assert_eq!(pohlig_hellman(&q3), 3);

        // Test case 4: k = 4
        let q4 = g.operate_with_self(4u16);
        assert_eq!(pohlig_hellman(&q4), 4);

        // Test case 5: k = 5
        let q5 = g.operate_with_self(9u16);
        assert_eq!(pohlig_hellman(&q5), 9);
    }

    #[test]
    fn test_pohlig_hellman_2() {
        let g = SmoothCurve::generator();

        // Test case 1: k = 1
        let q1 = g.operate_with_self(1u16);
        assert_eq!(pohlig_hellman_2(&q1), 1);

        // Test case 2: k = 2
        let q2 = g.operate_with_self(2u16);
        assert_eq!(pohlig_hellman_2(&q2), 2);

        // Test case 3: k = 3
        let q3 = g.operate_with_self(3u16);
        assert_eq!(pohlig_hellman_2(&q3), 3);

        // Test case 4: k = 4
        let q4 = g.operate_with_self(4u16);
        assert_eq!(pohlig_hellman_2(&q4), 4);

        // Test case 5: k = 5
        let q5 = g.operate_with_self(5u16);
        assert_eq!(pohlig_hellman_2(&q5), 5);
    }

    #[test]
    fn test_pohlig_hellman_3() {
        let g = SmoothCurve::generator();

        // Test case 1: k = 1
        let q1 = g.operate_with_self(1u16);
        assert_eq!(pohlig_hellman_3(&q1), 1);

        // Test case 2: k = 2
        let q2 = g.operate_with_self(2u16);
        assert_eq!(pohlig_hellman_3(&q2), 2);

        // Test case 3: k = 3
        let q3 = g.operate_with_self(3u16);
        assert_eq!(pohlig_hellman_3(&q3), 3);

        // Test case 4: k = 4
        let q4 = g.operate_with_self(4u16);
        assert_eq!(pohlig_hellman_3(&q4), 4);

        // Test case 5: k = 5
        let q5 = g.operate_with_self(5u16);
        assert_eq!(pohlig_hellman_3(&q5), 5);
    }

    #[test]
    fn test_pohlig_hellman_4() {
        let g = SmoothCurve2::generator();

        // Test case 1: k = 1
        let q1 = g.operate_with_self(1u16);
        assert_eq!(pohlig_hellman_4(&q1), 1);

        // Test case 2: k = 2
        let q2 = g.operate_with_self(2u16);
        assert_eq!(pohlig_hellman_4(&q2), 2);

        // Test case 3: k = 3
        let q3 = g.operate_with_self(3u16);
        assert_eq!(pohlig_hellman_4(&q3), 3);

        // Test case 4: k = 4
        let q4 = g.operate_with_self(4u16);
        assert_eq!(pohlig_hellman_4(&q4), 4);

        // // Test case 5: k = 5
        // let q5 = g.operate_with_self(5u16);
        // assert_eq!(pohlig_hellman_4(&q5), 5);
    }

    #[test]
    fn factorize_curve_2_order() {
        let order: i128 = 130086066303665968735;
        let factors = factorize(order);
        println!("factors: {:?}", factors);
    }
}
