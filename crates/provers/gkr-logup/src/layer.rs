use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;
use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

use crate::fraction::{Fraction, Reciprocal};

/// A layer in a binary-tree-structured GKR circuit.
///
/// Each layer has half the size of the layer below it. The leaves (input layer)
/// are at the bottom, and the root (output layer) has a single element.
#[derive(Debug, Clone)]
pub enum Layer<F: IsField>
where
    F::BaseType: Send + Sync,
{
    /// Product gate: `output[j] = input[2j] * input[2j+1]`.
    GrandProduct(DenseMultilinearPolynomial<F>),
    /// LogUp fraction gate with separate numerator/denominator columns.
    LogUpGeneric {
        numerators: DenseMultilinearPolynomial<F>,
        denominators: DenseMultilinearPolynomial<F>,
    },
    /// LogUp with base-field multiplicities that convert to Generic after first fix.
    LogUpMultiplicities {
        numerators: DenseMultilinearPolynomial<F>,
        denominators: DenseMultilinearPolynomial<F>,
    },
    /// LogUp where all numerators are implicitly 1.
    LogUpSingles {
        denominators: DenseMultilinearPolynomial<F>,
    },
}

impl<F: IsField> Layer<F>
where
    F::BaseType: Send + Sync,
{
    /// Number of variables used to interpolate the layer's gate values.
    pub fn n_variables(&self) -> usize {
        match self {
            Self::GrandProduct(mle) | Self::LogUpSingles { denominators: mle } => mle.num_vars(),
            Self::LogUpGeneric { denominators, .. }
            | Self::LogUpMultiplicities { denominators, .. } => denominators.num_vars(),
        }
    }

    /// Whether this is the root layer (single value, 0 variables).
    pub fn is_output_layer(&self) -> bool {
        self.n_variables() == 0
    }

    /// Computes the next (parent) layer by pairwise combining elements.
    /// Returns `None` if already at the output layer.
    pub fn next_layer(&self) -> Option<Self> {
        if self.is_output_layer() {
            return None;
        }
        Some(match self {
            Self::GrandProduct(mle) => next_grand_product_layer(mle),
            Self::LogUpGeneric {
                numerators,
                denominators,
            }
            | Self::LogUpMultiplicities {
                numerators,
                denominators,
            } => next_logup_layer(Some(numerators), denominators),
            Self::LogUpSingles { denominators } => next_logup_layer(None, denominators),
        })
    }

    /// Returns the column values at the output (root) layer.
    /// - GrandProduct: `[product]`
    /// - LogUp variants: `[numerator, denominator]`
    pub fn try_into_output_layer_values(&self) -> Option<Vec<FieldElement<F>>> {
        if !self.is_output_layer() {
            return None;
        }
        Some(match self {
            Layer::GrandProduct(col) => vec![col[0].clone()],
            Layer::LogUpGeneric {
                numerators,
                denominators,
            }
            | Layer::LogUpMultiplicities {
                numerators,
                denominators,
            } => vec![numerators[0].clone(), denominators[0].clone()],
            Layer::LogUpSingles { denominators } => {
                vec![FieldElement::one(), denominators[0].clone()]
            }
        })
    }

    /// Fixes the first variable of all inner MLEs to `x0`.
    ///
    /// `LogUpMultiplicities` converts to `LogUpGeneric` after the first fix,
    /// since the numerator field extends.
    pub fn fix_first_variable(self, x0: &FieldElement<F>) -> Self {
        match self {
            Self::GrandProduct(mle) => Self::GrandProduct(mle.fix_first_variable(x0)),
            Self::LogUpGeneric {
                numerators,
                denominators,
            } => Self::LogUpGeneric {
                numerators: numerators.fix_first_variable(x0),
                denominators: denominators.fix_first_variable(x0),
            },
            Self::LogUpMultiplicities {
                numerators,
                denominators,
            } => {
                // Convert to Generic after first variable fix
                Self::LogUpGeneric {
                    numerators: numerators.fix_first_variable(x0),
                    denominators: denominators.fix_first_variable(x0),
                }
            }
            Self::LogUpSingles { denominators } => Self::LogUpSingles {
                denominators: denominators.fix_first_variable(x0),
            },
        }
    }
}

/// Generates all layers from the input (leaves) to the output (root).
pub fn gen_layers<F: IsField>(
    input_layer: Layer<F>,
) -> Result<Vec<Layer<F>>, lambdaworks_sumcheck::ProverError>
where
    F::BaseType: Send + Sync,
{
    let n_variables = input_layer.n_variables();
    let mut layers = vec![input_layer];
    while let Some(next) = layers.last().unwrap().next_layer() {
        layers.push(next);
    }
    if layers.len() != n_variables + 1 {
        return Err(lambdaworks_sumcheck::ProverError::InvalidState(format!(
            "gen_layers: expected {} layers (n_variables={}) but produced {}",
            n_variables + 1,
            n_variables,
            layers.len()
        )));
    }
    Ok(layers)
}

/// GrandProduct: `output[j] = input[2j] * input[2j+1]`.
fn next_grand_product_layer<F: IsField>(layer: &DenseMultilinearPolynomial<F>) -> Layer<F>
where
    F::BaseType: Send + Sync,
{
    let half_n = layer.len() / 2;
    let mut res = Vec::with_capacity(half_n);
    for j in 0..half_n {
        res.push(&layer[j * 2] * &layer[j * 2 + 1]);
    }
    Layer::GrandProduct(DenseMultilinearPolynomial::new(res))
}

/// LogUp fraction addition: `num[j]/den[j] + num[j+1]/den[j+1]`.
/// If `numerators` is `None`, all numerators are implicitly 1 (singles case).
fn next_logup_layer<F: IsField>(
    numerators: Option<&DenseMultilinearPolynomial<F>>,
    denominators: &DenseMultilinearPolynomial<F>,
) -> Layer<F>
where
    F::BaseType: Send + Sync,
{
    let half_n = denominators.len() / 2;
    let mut next_numerators = Vec::with_capacity(half_n);
    let mut next_denominators = Vec::with_capacity(half_n);

    for j in 0..half_n {
        let result = match numerators {
            Some(nums) => {
                let a = Fraction::new(nums[j * 2].clone(), denominators[j * 2].clone());
                let b = Fraction::new(nums[j * 2 + 1].clone(), denominators[j * 2 + 1].clone());
                a + b
            }
            None => {
                let a = Reciprocal::new(denominators[j * 2].clone());
                let b = Reciprocal::new(denominators[j * 2 + 1].clone());
                a + b
            }
        };
        next_numerators.push(result.numerator);
        next_denominators.push(result.denominator);
    }

    Layer::LogUpGeneric {
        numerators: DenseMultilinearPolynomial::new(next_numerators),
        denominators: DenseMultilinearPolynomial::new(next_denominators),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn grand_product_next_layer() {
        // Input: [2, 3, 5, 7]
        // Next:  [2*3, 5*7] = [6, 35]
        // Next:  [6*35] = [210] = 210 mod 101 = 8
        let input = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![
            FE::from(2),
            FE::from(3),
            FE::from(5),
            FE::from(7),
        ]));

        let next = input.next_layer().unwrap();
        match &next {
            Layer::GrandProduct(mle) => {
                assert_eq!(mle.len(), 2);
                assert_eq!(mle[0], FE::from(6));
                assert_eq!(mle[1], FE::from(35));
            }
            _ => panic!("Expected GrandProduct"),
        }

        let root = next.next_layer().unwrap();
        match &root {
            Layer::GrandProduct(mle) => {
                assert_eq!(mle.len(), 1);
                assert_eq!(mle[0], FE::from(6) * FE::from(35));
            }
            _ => panic!("Expected GrandProduct"),
        }

        assert!(root.next_layer().is_none());
    }

    #[test]
    fn grand_product_gen_layers() {
        let input = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![
            FE::from(2),
            FE::from(3),
            FE::from(5),
            FE::from(7),
        ]));
        let layers = gen_layers(input).unwrap();
        assert_eq!(layers.len(), 3); // 2 vars + 1

        // Root should be product of all
        let product = FE::from(2) * FE::from(3) * FE::from(5) * FE::from(7);
        let root_vals = layers[2].try_into_output_layer_values().unwrap();
        assert_eq!(root_vals, vec![product]);
    }

    #[test]
    fn logup_generic_next_layer() {
        // Fractions: 1/2, 3/4 → (1*4 + 2*3)/(2*4) = 10/8
        let input = Layer::<F>::LogUpGeneric {
            numerators: DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(3)]),
            denominators: DenseMultilinearPolynomial::new(vec![FE::from(2), FE::from(4)]),
        };

        let next = input.next_layer().unwrap();
        match &next {
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                assert_eq!(numerators.len(), 1);
                assert_eq!(numerators[0], FE::from(10)); // 1*4 + 2*3
                assert_eq!(denominators[0], FE::from(8)); // 2*4
            }
            _ => panic!("Expected LogUpGeneric"),
        }
    }

    #[test]
    fn logup_singles_next_layer() {
        // 1/2 + 1/3 = (2+3)/(2*3) = 5/6
        let input = Layer::<F>::LogUpSingles {
            denominators: DenseMultilinearPolynomial::new(vec![FE::from(2), FE::from(3)]),
        };

        let next = input.next_layer().unwrap();
        match &next {
            Layer::LogUpGeneric {
                numerators,
                denominators,
            } => {
                assert_eq!(numerators[0], FE::from(5));
                assert_eq!(denominators[0], FE::from(6));
            }
            _ => panic!("Expected LogUpGeneric"),
        }
    }

    #[test]
    fn logup_sum_matches_tree() {
        // 4 fractions: 1/2, 1/3, 1/5, 1/7
        // Sum = 1/2 + 1/3 + 1/5 + 1/7
        let input = Layer::<F>::LogUpSingles {
            denominators: DenseMultilinearPolynomial::new(vec![
                FE::from(2),
                FE::from(3),
                FE::from(5),
                FE::from(7),
            ]),
        };
        let layers = gen_layers(input).unwrap();

        // Compute expected sum: 1/2 + 1/3 + 1/5 + 1/7
        let fracs: Vec<Fraction<F>> = [2u64, 3, 5, 7]
            .iter()
            .map(|&d| Fraction::new(FE::one(), FE::from(d)))
            .collect();
        let expected: Fraction<F> = fracs.into_iter().sum();

        let root_vals = layers
            .last()
            .unwrap()
            .try_into_output_layer_values()
            .unwrap();
        // Check numerator/denominator ratio matches
        let tree_ratio = root_vals[0] * expected.denominator.inv().unwrap();
        let expected_ratio = expected.numerator * root_vals[1].inv().unwrap();
        assert_eq!(tree_ratio, expected_ratio);
    }

    #[test]
    fn fix_first_variable_consistency() {
        let input = Layer::GrandProduct(DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(3),
            FE::from(4),
        ]));
        let r = FE::from(7);

        let fixed = input.fix_first_variable(&r);
        match &fixed {
            Layer::GrandProduct(mle) => {
                assert_eq!(mle.len(), 2);
                // f(r, 0) = f(0,0) + r*(f(1,0) - f(0,0)) = 1 + 7*(3-1) = 15
                assert_eq!(
                    mle[0],
                    FE::from(1) + FE::from(7) * (FE::from(3) - FE::from(1))
                );
                // f(r, 1) = f(0,1) + r*(f(1,1) - f(0,1)) = 2 + 7*(4-2) = 16
                assert_eq!(
                    mle[1],
                    FE::from(2) + FE::from(7) * (FE::from(4) - FE::from(2))
                );
            }
            _ => panic!("Expected GrandProduct"),
        }
    }

    #[test]
    fn multiplicities_converts_to_generic_on_fix() {
        let input = Layer::<F>::LogUpMultiplicities {
            numerators: DenseMultilinearPolynomial::new(vec![FE::from(1), FE::from(2)]),
            denominators: DenseMultilinearPolynomial::new(vec![FE::from(3), FE::from(4)]),
        };
        let r = FE::from(5);
        let fixed = input.fix_first_variable(&r);
        assert!(matches!(fixed, Layer::LogUpGeneric { .. }));
    }
}
