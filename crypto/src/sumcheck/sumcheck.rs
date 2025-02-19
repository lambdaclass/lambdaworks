// sumcheck.rs

use alloc::vec::Vec;
use core::ops::Add;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;
use rand::Rng;

/// Para que DenseMultilinearPolynomial funcione correctamente en contextos paralelos,
/// requerimos que la BaseType del campo sea Send y Sync.
pub trait SumCheckPolynomial<F: IsField>: Sized
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Evalúa el polinomio en el punto dado.
    fn evaluate(&self, point: &[FieldElement<F>]) -> Option<FieldElement<F>>;

    /// “Fija” (bind) una variable (en nuestro protocolo, siempre la última) y produce un nuevo
    /// polinomio con una variable menos.
    fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self;

    /// Devuelve las evaluaciones del polinomio sobre el hipercubo booleano \(\{0,1\}^n\).
    fn to_evaluations(&self) -> Vec<FieldElement<F>>;

    /// Devuelve la cantidad de variables.
    fn num_vars(&self) -> usize;

    /// “Colapsa” la última variable pendiente, produciendo un polinomio univariado en esa variable
    /// cuya evaluación es
    ///
    /// \[
    /// U(t) = \sum_{(x_0,\dots,x_{n-2}) \in \{0,1\}^{n-1}} g(x_0,\dots,x_{n-2},t)
    /// \]
    fn to_univariate(&self) -> UnivariatePolynomial<F>;
}

/// Un polinomio univariado simple, representado por sus coeficientes.
pub struct UnivariatePolynomial<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Coeficientes: \( a_0 + a_1 x + \cdots + a_k x^k \).
    pub coeffs: Vec<FieldElement<F>>,
}

impl<F: IsField> UnivariatePolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Evalúa el polinomio en el valor \( x \) usando evaluación naïve.
    pub fn evaluate(&self, x: &FieldElement<F>) -> FieldElement<F> {
        let mut result = FieldElement::zero();
        let mut power = FieldElement::<F>::one();
        for c in &self.coeffs {
            result += c * &power;
            power *= x;
        }
        result
    }
}

/// --- EXTENSIÓN: Trait de extensión para DenseMultilinearPolynomial ---
/// Define el método fix_last_variable (no podemos agregarlo como inherent, pues el tipo es foráneo).
pub trait DenseMultilinearPolynomialExt<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn fix_last_variable(&self, r: &FieldElement<F>) -> DenseMultilinearPolynomial<F>;
}

impl<F: IsField> DenseMultilinearPolynomialExt<F> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    fn fix_last_variable(&self, r: &FieldElement<F>) -> DenseMultilinearPolynomial<F> {
        let n = self.num_vars();
        assert!(n > 0, "Cannot fix variable in a 0-variable poly");
        let half = 1 << (n - 1);
        // Usamos el método evals() para obtener la tabla (ya que el campo evals es privado).
        let new_evals: Vec<FieldElement<F>> = (0..half)
            .map(|j| {
                let a = self.evals()[j].clone();
                let b = self.evals()[j + half].clone();
                &a + r * (b - a.clone())
            })
            .collect();
        DenseMultilinearPolynomial::from_evaluations_vec(n - 1, new_evals)
    }
}

/// --- Implementación de SumCheckPolynomial para DenseMultilinearPolynomial ---
impl<F: IsField> SumCheckPolynomial<F> for DenseMultilinearPolynomial<F>
where
    <F as IsField>::BaseType: Send + Sync,
{
    // Usamos evaluate() existente (convierte el resultado a Option).
    fn evaluate(&self, point: &[FieldElement<F>]) -> Option<FieldElement<F>> {
        self.evaluate(point.to_vec()).ok()
    }

    fn fix_variables(&self, partial_point: &[FieldElement<F>]) -> Self {
        // En Sumcheck se espera fijar una variable a la vez.
        assert!(
            partial_point.len() == 1,
            "Se debe fijar exactamente una variable por ronda"
        );
        // Aquí usamos nuestro método de extensión.
        DenseMultilinearPolynomialExt::fix_last_variable(self, &partial_point[0])
    }

    fn to_evaluations(&self) -> Vec<FieldElement<F>> {
        let mut result = Vec::with_capacity(1 << self.num_vars());
        for i in 0..(1 << self.num_vars()) {
            let mut point = Vec::with_capacity(self.num_vars());
            for bit_idx in 0..self.num_vars() {
                let bit = ((i >> bit_idx) & 1) == 1;
                point.push(if bit {
                    FieldElement::one()
                } else {
                    FieldElement::zero()
                });
            }
            // Llamamos a evaluate y desempaquetamos (suponiendo que el punto es válido).
            result.push(self.evaluate(point).unwrap());
        }
        result
    }

    fn num_vars(&self) -> usize {
        self.num_vars()
    }

    fn to_univariate(&self) -> UnivariatePolynomial<F> {
        // Fijamos la última variable a 0 y a 1.
        let poly0 = DenseMultilinearPolynomialExt::fix_last_variable(self, &FieldElement::zero());
        let poly1 = DenseMultilinearPolynomialExt::fix_last_variable(self, &FieldElement::one());
        let sum0: FieldElement<F> = poly0.to_evaluations().into_iter().sum();
        let sum1: FieldElement<F> = poly1.to_evaluations().into_iter().sum();
        UnivariatePolynomial {
            coeffs: vec![sum0.clone(), sum1 - sum0],
        }
    }
}

/// Prover para Sumcheck.
pub struct Prover<F: IsField, P: SumCheckPolynomial<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub poly: P,
    pub claimed_sum: FieldElement<F>,
    pub current_round: usize,
}

impl<F: IsField, P: SumCheckPolynomial<F>> Prover<F, P>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(poly: P) -> Self {
        let evals = poly.to_evaluations();
        let claimed_sum = evals.into_iter().sum();
        Self {
            poly,
            claimed_sum,
            current_round: 0,
        }
    }

    pub fn c_1(&self) -> FieldElement<F> {
        self.claimed_sum.clone()
    }

    /// Recibe el desafío \( r_j \) del verificador, fija la última variable a ese valor y
    /// convierte el polinomio resultante en un polinomio univariado respecto de la siguiente variable.
    pub fn round(&mut self, r_j: FieldElement<F>) -> UnivariatePolynomial<F> {
        // Aquí usamos fix_variables (que internamente usa fix_last_variable).
        self.poly = self.poly.fix_variables(&[r_j]);
        let univar = self.poly.to_univariate();
        self.current_round += 1;
        univar
    }
}

/// Resultado de una ronda del verificador.
pub enum VerifierRoundResult<F: IsField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    /// Continúa a la siguiente ronda, devolviendo el desafío \( r_j \) para enviar al Prover.
    NextRound(FieldElement<F>),
    /// Ronda final: true si el verificador acepta, false si rechaza.
    Final(bool),
}

/// Verifier para Sumcheck. Se almacena el vector de desafíos para reconstruir el punto completo.
pub struct Verifier<F: IsField, P: SumCheckPolynomial<F>>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub n: usize,
    pub c_1: FieldElement<F>,
    pub round: usize,
    pub poly: Option<P>,                  // acceso oracular (opcional)
    pub last_val: FieldElement<F>, // valor de la ronda anterior, por ejemplo, U_{j-1}(r_{j-1})
    pub challenges: Vec<FieldElement<F>>, // desafíos recibidos en cada ronda.
}

impl<F: IsField, P: SumCheckPolynomial<F>> Verifier<F, P>
where
    <F as IsField>::BaseType: Send + Sync,
{
    pub fn new(n: usize, poly: Option<P>, c_1: FieldElement<F>) -> Self {
        Self {
            n,
            c_1,
            round: 0,
            poly,
            last_val: FieldElement::zero(),
            challenges: Vec::with_capacity(n),
        }
    }

    /// Ejecuta la ronda \( j \) del verificador.
    /// Verifica que la suma de las evaluaciones del polinomio univariado \( U_j \) cumpla la propiedad requerida,
    /// luego escoge un desafío \( r_j \) y actualiza el estado.
    pub fn do_round<R: Rng>(
        &mut self,
        univar: UnivariatePolynomial<F>,
        rng: &mut R,
    ) -> VerifierRoundResult<F> {
        if self.round == 0 {
            let s0 = univar.evaluate(&FieldElement::zero());
            let s1 = univar.evaluate(&FieldElement::one());
            if s0 + s1 != self.c_1 {
                return VerifierRoundResult::Final(false);
            }
        } else {
            let s0 = univar.evaluate(&FieldElement::zero());
            let s1 = univar.evaluate(&FieldElement::one());
            if s0 + s1 != self.last_val {
                return VerifierRoundResult::Final(false);
            }
        }
        // Escoge un desafío aleatorio r_j.
        let r_j = FieldElement::<F>::from(rng.gen::<u64>());
        self.challenges.push(r_j.clone());
        let val = univar.evaluate(&r_j);
        self.last_val = val;
        self.round += 1;
        if self.round == self.n {
            // Ronda final: si tenemos acceso oracular, comparamos con la evaluación real.
            if let Some(ref poly) = self.poly {
                let full_point = self.challenges.clone();
                if let Some(real_val) = poly.evaluate(full_point.as_slice()) {
                    return VerifierRoundResult::Final(real_val == self.last_val);
                }
            }
            VerifierRoundResult::Final(true)
        } else {
            VerifierRoundResult::NextRound(r_j)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;
    use rand::thread_rng;

    const MODULUS: u64 = 101;
    type F = U64PrimeField<MODULUS>;
    type FE = FieldElement<F>;

    #[test]
    fn sumcheck_demo() {
        // Sea el polinomio multilineal con evaluaciones [1, 2, 1, 4] (2 variables).
        let poly = DenseMultilinearPolynomial::new(vec![
            FE::from(1),
            FE::from(2),
            FE::from(1),
            FE::from(4),
        ]);

        // Prover: declara la suma de las evaluaciones.
        let mut prover = Prover::new(poly.clone());
        let c_1 = prover.c_1();

        // Verifier: usa el número de variables y (opcionalmente) acceso oracular.
        let mut verifier = Verifier::new(poly.num_vars(), Some(poly.clone()), c_1);
        let mut rng = thread_rng();

        // Ronda 0: El Prover convierte el polinomio en uno univariado respecto a la última variable.
        let univar0 = poly.to_univariate();
        match verifier.do_round(univar0, &mut rng) {
            VerifierRoundResult::NextRound(r0) => {
                // El Prover recibe el desafío r0 y fija la última variable a r0.
                let univar1 = prover.round(r0);
                // Verifier verifica la ronda 1.
                match verifier.do_round(univar1, &mut rng) {
                    VerifierRoundResult::Final(ok) => assert!(ok, "La verificación final falló"),
                    _ => panic!("Se esperaba una ronda final"),
                }
            }
            VerifierRoundResult::Final(false) => panic!("La verificación de la ronda 0 falló"),
            VerifierRoundResult::Final(true) => {
                panic!("La ronda 0 no debe ser final")
            }
        }
    }
}
