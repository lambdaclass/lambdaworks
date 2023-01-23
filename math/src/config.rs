// The elliptic curve is taken from the book "Pairing for beginners", page 57.
// We use the short Weierstrass form equation: y^2 = x^3 + A * x  + B

pub const ORDER_R: u64 = 5; // Base coefficients for polynomials that encode the circuit structure.
pub const ORDER_P: u64 = 59; // Base coefficients for coordinates of points in elliptic curve.
pub const EMBEDDING_DEGREE: u32 = 2; // Degree to ensure that torsion group is contained in the elliptic curve over field extensions.
pub const ORDER_FIELD_EXTENSION: u64 = ORDER_P.pow(EMBEDDING_DEGREE);
pub const TARGET_NORMALIZATION_POWER: u64 = (ORDER_FIELD_EXTENSION - 1) / ORDER_R;
pub const ELLIPTIC_CURVE_A: u64 = 1;
pub const ELLIPTIC_CURVE_B: u64 = 0;
pub const GENERATOR_AFFINE_X: u64 = 35;
pub const GENERATOR_AFFINE_Y: u64 = 31;
