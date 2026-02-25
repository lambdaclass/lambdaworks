use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::polynomial::DenseMultilinearPolynomial;

use crate::fraction::Fraction;
use crate::layer::Layer;
use crate::univariate::lagrange::UnivariateLagrange;
use crate::univariate::Commitment;

#[derive(Debug, Clone)]
pub enum UnivariateLayer<F: IsFFTField>
where
    F::BaseType: Send + Sync,
{
    GrandProduct {
        values: UnivariateLagrange<F>,
        commitment: Option<Commitment>,
    },
    LogUpGeneric {
        numerators: UnivariateLagrange<F>,
        denominators: UnivariateLagrange<F>,
        numerator_commitment: Option<Commitment>,
        denominator_commitment: Option<Commitment>,
    },
    LogUpMultiplicities {
        numerators: UnivariateLagrange<F>,
        denominators: UnivariateLagrange<F>,
        numerator_commitment: Option<Commitment>,
        denominator_commitment: Option<Commitment>,
    },
    LogUpSingles {
        denominators: UnivariateLagrange<F>,
        denominator_commitment: Option<Commitment>,
    },
}

impl<F: IsFFTField> UnivariateLayer<F>
where
    F::BaseType: Send + Sync,
{
    pub fn n_variables(&self) -> usize {
        match self {
            Self::GrandProduct { values, .. } => values.n_variables(),
            Self::LogUpSingles { denominators, .. } => denominators.n_variables(),
            Self::LogUpGeneric { denominators, .. }
            | Self::LogUpMultiplicities { denominators, .. } => denominators.n_variables(),
        }
    }

    pub fn is_output_layer(&self) -> bool {
        self.n_variables() == 0
    }

    pub fn next_layer(&self) -> Option<Self> {
        if self.is_output_layer() {
            return None;
        }
        match self {
            Self::GrandProduct {
                values,
                commitment: _,
            } => next_grand_product_layer(values).map(|v| Self::GrandProduct {
                values: v,
                commitment: None,
            }),
            Self::LogUpGeneric {
                numerators,
                denominators,
                ..
            } => next_logup_layer(Some(numerators), denominators).map(|(num, den)| {
                Self::LogUpGeneric {
                    numerators: num,
                    denominators: den,
                    numerator_commitment: None,
                    denominator_commitment: None,
                }
            }),
            Self::LogUpMultiplicities {
                numerators,
                denominators,
                ..
            } => next_logup_layer(Some(numerators), denominators).map(|(num, den)| {
                Self::LogUpGeneric {
                    numerators: num,
                    denominators: den,
                    numerator_commitment: None,
                    denominator_commitment: None,
                }
            }),
            Self::LogUpSingles { denominators, .. } => {
                next_logup_layer(None, denominators).map(|(num, den)| Self::LogUpGeneric {
                    numerators: num,
                    denominators: den,
                    numerator_commitment: None,
                    denominator_commitment: None,
                })
            }
        }
    }

    pub fn try_into_output_layer_values(&self) -> Option<Vec<FieldElement<F>>> {
        if !self.is_output_layer() {
            return None;
        }
        Some(match self {
            Self::GrandProduct { values, .. } => vec![values.values[0].clone()],
            Self::LogUpGeneric {
                numerators,
                denominators,
                ..
            }
            | Self::LogUpMultiplicities {
                numerators,
                denominators,
                ..
            } => {
                vec![numerators.values[0].clone(), denominators.values[0].clone()]
            }
            Self::LogUpSingles { denominators, .. } => {
                vec![FieldElement::one(), denominators.values[0].clone()]
            }
        })
    }

    pub fn fix_first_variable(self, x0: &FieldElement<F>) -> Self {
        match self {
            Self::GrandProduct { values, commitment } => Self::GrandProduct {
                values: values.fix_first_variable(x0),
                commitment,
            },
            Self::LogUpGeneric {
                numerators,
                denominators,
                numerator_commitment,
                denominator_commitment,
            } => Self::LogUpGeneric {
                numerators: numerators.fix_first_variable(x0),
                denominators: denominators.fix_first_variable(x0),
                numerator_commitment,
                denominator_commitment,
            },
            Self::LogUpMultiplicities {
                numerators,
                denominators,
                numerator_commitment,
                denominator_commitment,
            } => Self::LogUpGeneric {
                numerators: numerators.fix_first_variable(x0),
                denominators: denominators.fix_first_variable(x0),
                numerator_commitment,
                denominator_commitment,
            },
            Self::LogUpSingles {
                denominators,
                denominator_commitment,
            } => Self::LogUpSingles {
                denominators: denominators.fix_first_variable(x0),
                denominator_commitment,
            },
        }
    }

    /// Converts this univariate layer to a standard multilinear `Layer`.
    ///
    /// The `UnivariateLagrange` values are MLE evaluations on the Boolean hypercube
    /// (indexed such that `values[i] = f(i_0, ..., i_{n-1})` with `i = sum i_k * 2^k`).
    /// This method wraps them in `DenseMultilinearPolynomial` for use with the GKR prover.
    pub fn to_multilinear_layer(&self) -> Layer<F> {
        match self {
            Self::GrandProduct { values, .. } => {
                Layer::GrandProduct(DenseMultilinearPolynomial::new(values.values.clone()))
            }
            Self::LogUpGeneric {
                numerators,
                denominators,
                ..
            } => Layer::LogUpGeneric {
                numerators: DenseMultilinearPolynomial::new(numerators.values.clone()),
                denominators: DenseMultilinearPolynomial::new(denominators.values.clone()),
            },
            Self::LogUpMultiplicities {
                numerators,
                denominators,
                ..
            } => Layer::LogUpMultiplicities {
                numerators: DenseMultilinearPolynomial::new(numerators.values.clone()),
                denominators: DenseMultilinearPolynomial::new(denominators.values.clone()),
            },
            Self::LogUpSingles { denominators, .. } => Layer::LogUpSingles {
                denominators: DenseMultilinearPolynomial::new(denominators.values.clone()),
            },
        }
    }

    /// Extracts all univariate value vectors from this layer.
    ///
    /// Returns the raw evaluation vectors in a consistent order matching the
    /// GKR claim ordering:
    /// - GrandProduct: `[values]`
    /// - LogUpGeneric/LogUpMultiplicities: `[numerators, denominators]`
    /// - LogUpSingles: `[ones, denominators]` (implicit numerator = 1 for each entry)
    pub fn get_univariate_values(&self) -> Vec<Vec<FieldElement<F>>> {
        match self {
            Self::GrandProduct { values, .. } => vec![values.values.clone()],
            Self::LogUpGeneric {
                numerators,
                denominators,
                ..
            }
            | Self::LogUpMultiplicities {
                numerators,
                denominators,
                ..
            } => vec![numerators.values.clone(), denominators.values.clone()],
            Self::LogUpSingles { denominators, .. } => {
                let ones = vec![FieldElement::one(); denominators.len()];
                vec![ones, denominators.values.clone()]
            }
        }
    }

    pub fn set_commitments(&mut self, mut commitments: Vec<Commitment>) {
        match self {
            Self::GrandProduct { commitment, .. } => {
                if let Some(c) = commitments.pop() {
                    *commitment = Some(c);
                }
            }
            Self::LogUpGeneric {
                denominator_commitment,
                numerator_commitment,
                ..
            }
            | Self::LogUpMultiplicities {
                denominator_commitment,
                numerator_commitment,
                ..
            } => {
                if let Some(c) = commitments.pop() {
                    *denominator_commitment = Some(c);
                }
                if let Some(c) = commitments.pop() {
                    *numerator_commitment = Some(c);
                }
            }
            Self::LogUpSingles {
                denominator_commitment,
                ..
            } => {
                if let Some(c) = commitments.pop() {
                    *denominator_commitment = Some(c);
                }
            }
        }
    }

    pub fn get_commitments(&self) -> Vec<Commitment> {
        match self {
            Self::GrandProduct { commitment, .. } => commitment.clone().into_iter().collect(),
            Self::LogUpGeneric {
                numerator_commitment,
                denominator_commitment,
                ..
            }
            | Self::LogUpMultiplicities {
                numerator_commitment,
                denominator_commitment,
                ..
            } => {
                let mut r = numerator_commitment.clone().into_iter().collect::<Vec<_>>();
                r.extend(denominator_commitment.clone());
                r
            }
            Self::LogUpSingles {
                denominator_commitment,
                ..
            } => denominator_commitment.clone().into_iter().collect(),
        }
    }
}

fn next_grand_product_layer<F: IsFFTField>(
    values: &UnivariateLagrange<F>,
) -> Option<UnivariateLagrange<F>>
where
    F::BaseType: Send + Sync,
{
    use crate::univariate::domain::CyclicDomain;
    use crate::univariate::lagrange::UnivariateLagrange;

    let n = values.len();
    if n == 1 {
        return None;
    }

    let log_n = n.ilog2() as usize;
    let new_n = n / 2;
    let mut new_values = Vec::with_capacity(new_n);

    for i in 0..new_n {
        let prod = values.values[2 * i].clone() * values.values[2 * i + 1].clone();
        new_values.push(prod);
    }

    let domain = CyclicDomain::new(log_n - 1).ok()?;
    UnivariateLagrange::new(new_values, domain).ok()
}

fn next_logup_layer<F: IsFFTField>(
    numerators: Option<&UnivariateLagrange<F>>,
    denominators: &UnivariateLagrange<F>,
) -> Option<(UnivariateLagrange<F>, UnivariateLagrange<F>)>
where
    F::BaseType: Send + Sync,
{
    use crate::univariate::domain::CyclicDomain;
    use crate::univariate::lagrange::UnivariateLagrange;

    let n = denominators.len();
    if n == 1 {
        return None;
    }

    let log_n = n.ilog2() as usize;
    let new_n = n / 2;
    let mut new_numerators = Vec::with_capacity(new_n);
    let mut new_denominators = Vec::with_capacity(new_n);

    for i in 0..new_n {
        let num_i = match numerators {
            Some(nums) => (nums.values[2 * i].clone(), nums.values[2 * i + 1].clone()),
            None => (FieldElement::one(), FieldElement::one()),
        };

        let den_i = (
            denominators.values[2 * i].clone(),
            denominators.values[2 * i + 1].clone(),
        );

        let a = Fraction::new(num_i.0, den_i.0);
        let b = Fraction::new(num_i.1, den_i.1);
        let result = a + b;

        new_numerators.push(result.numerator);
        new_denominators.push(result.denominator);
    }

    let domain = CyclicDomain::new(log_n - 1).ok()?;
    let new_numerators = UnivariateLagrange::new(new_numerators, domain.clone()).ok()?;
    let new_denominators = UnivariateLagrange::new(new_denominators, domain).ok()?;

    Some((new_numerators, new_denominators))
}

pub fn gen_layers<F: IsFFTField>(input_layer: UnivariateLayer<F>) -> Vec<UnivariateLayer<F>>
where
    F::BaseType: Send + Sync,
{
    let n_variables = input_layer.n_variables();
    let mut layers = vec![input_layer];
    while let Some(next) = layers.last().unwrap().next_layer() {
        layers.push(next);
    }
    assert_eq!(layers.len(), n_variables + 1);
    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::univariate::domain::CyclicDomain;
    use crate::univariate::lagrange::UnivariateLagrange;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn test_univariate_layer_grand_product() {
        let values: Vec<FE> = (1..=4).map(FE::from).collect();
        let domain = CyclicDomain::new(2).unwrap();
        let uni = UnivariateLagrange::new(values, domain).unwrap();

        let layer = UnivariateLayer::GrandProduct {
            values: uni,
            commitment: None,
        };

        assert_eq!(layer.n_variables(), 2);
        assert!(!layer.is_output_layer());
    }

    #[test]
    fn test_univariate_layer_logup_singles() {
        let values: Vec<FE> = (1..=4).map(|i| FE::from(i * 2)).collect();
        let domain = CyclicDomain::new(2).unwrap();
        let uni = UnivariateLagrange::new(values, domain).unwrap();

        let layer = UnivariateLayer::LogUpSingles {
            denominators: uni,
            denominator_commitment: None,
        };

        let next = layer.next_layer().unwrap();
        assert!(matches!(next, UnivariateLayer::LogUpGeneric { .. }));
    }
}
