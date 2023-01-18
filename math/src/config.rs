// The elliptic curve is taken from the book "Pairing for beginners", page 57.
// We use the short Weierstrass form equation: y^2 = x^3 + A * x  + B

pub const ORDER_R: u128 = 5; // Base coefficients for polynomials that encode the circuit structure.
pub const ORDER_P: u128 = 59; // Base coefficients for coordinates of points in elliptic curve.
pub const EMBEDDING_DEGREE: u32 = 2; // Degree to ensure that torsion group is contained in the elliptic curve over field extensions.
pub const ORDER_FIELD_EXTENSION: u128 = ORDER_P.pow(EMBEDDING_DEGREE);
pub const TARGET_NORMALIZATION_POWER: u128 = (ORDER_FIELD_EXTENSION - 1) / ORDER_R;
pub const ELLIPTIC_CURVE_A: u128 = 1;
pub const ELLIPTIC_CURVE_B: u128 = 0;
pub const GENERATOR_AFFINE_X: u128 = 35;
pub const GENERATOR_AFFINE_Y: u128 = 31;
