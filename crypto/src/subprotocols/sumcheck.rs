use std::marker::PhantomData;

use crate::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsPrimeField};
use lambdaworks_math::polynomial::{
    dense_multilinear_poly::DenseMultilinearPolynomial, polynomial::Polynomial,
};
use lambdaworks_math::traits::ByteConversion;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn eval_points_quadratic<F: IsField + IsPrimeField, E>(
    poly_a: &DenseMultilinearPolynomial<F>,
    poly_b: &DenseMultilinearPolynomial<F>,
    comb_func: E,
) -> (FieldElement<F>, FieldElement<F>)
where
    <F as IsField>::BaseType: Send + Sync,
    E: Fn(&FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync + Send,
{
    let len = poly_a.len() / 2;
    (0..len)
        .into_par_iter()
        .map(|i| {
            // eval_0: A(low)
            let eval_0 = comb_func(&poly_a[i], &poly_b[i]);

            // eval_2: -A(low) + 2*A(high)
            let poly_a_eval_2 = poly_a[len + i] + poly_a[len + i] - poly_a[i];
            let poly_b_eval_2 = poly_b[len + i] + poly_b[len + i] - poly_b[i];
            let eval_2 = comb_func(&poly_a_eval_2, &poly_b_eval_2);
            (eval_0, eval_2)
        })
        .reduce(
            || (FieldElement::<F>::zero(), FieldElement::<F>::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1),
        )
}

fn eval_points_cubic<F: IsField, E>(
    poly_a: &DenseMultilinearPolynomial<F>,
    poly_b: &DenseMultilinearPolynomial<F>,
    poly_c: &DenseMultilinearPolynomial<F>,
    comb_func: E,
) -> (FieldElement<F>, FieldElement<F>, FieldElement<F>)
where
    <F as IsField>::BaseType: Send + Sync,
    E: Fn(&FieldElement<F>, &FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync,
{
    let len = poly_a.len() / 2;
    (0..len)
        .into_par_iter()
        .map(|i| {
            // eval_0: A(low)
            let eval_0 = comb_func(&poly_a[i], &poly_b[i], &poly_c[i]);

            // eval_2: -A(low) + 2*A(high)
            let poly_a_eval_2 = poly_a[len + i] + poly_a[len + i] - poly_a[i];
            let poly_b_eval_2 = poly_b[len + i] + poly_b[len + i] - poly_b[i];
            let poly_c_eval_2 = poly_c[len + i] + poly_c[len + i] - poly_c[i];
            let eval_2 = comb_func(&poly_a_eval_2, &poly_b_eval_2, &poly_c_eval_2);

            // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
            let poly_a_eval_3 = poly_a_eval_2 + poly_a[len + i] - poly_a[i];
            let poly_b_eval_3 = poly_b_eval_2 + poly_b[len + i] - poly_b[i];
            let poly_c_eval_3 = poly_c_eval_2 + poly_c[len + i] - poly_c[i];
            let eval_3 = comb_func(&poly_a_eval_2, &poly_b_eval_2, &poly_c_eval_2);

            (eval_0, eval_2, eval_3)
        })
        .reduce(
            || {
                (
                    FieldElement::<F>::zero(),
                    FieldElement::<F>::zero(),
                    FieldElement::<F>::zero(),
                )
            },
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        )
}

// Proof attesting to sum over the boolean hypercube
#[derive(Debug)]
pub struct SumcheckProof<F: IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    // Multilinear Polynomial whose sum is claimed to equal `sum` when evaluated over the Boolean Hypercube
    pub poly: DenseMultilinearPolynomial<F>,
    // Sum the proof is attesting to
    pub sum: FieldElement<F>,
    // Univariate polynomial oracles the prover sends to the verifier each round
    pub round_uni_polys: Vec<Polynomial<FieldElement<F>>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Sumcheck<F: IsField + IsPrimeField>
where
    <F as IsField>::BaseType: Send + Sync,
{
    _p: PhantomData<F>,
}

impl<F: IsField + IsPrimeField> Sumcheck<F>
where
    <F as IsField>::BaseType: Send + Sync,
    FieldElement<F>: ByteConversion,
{
    //Used for sum_{(a * b)}
    pub fn prove_quadratic<E>(
        sum: &FieldElement<F>,
        poly_a: &mut DenseMultilinearPolynomial<F>,
        poly_b: &mut DenseMultilinearPolynomial<F>,
        comb_func: E,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>
    where
        E: Fn(&FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync,
    {
        let mut round_uni_polys: Vec<Polynomial<FieldElement<F>>> =
            Vec::with_capacity(poly_a.num_vars());
        let mut challenges = Vec::with_capacity(poly_a.num_vars());
        let mut prev_round_claim = *sum;

        for _ in 0..poly_a.num_vars() {
            let poly = {
                let len = poly_a.len() / 2;
                let (eval_0, eval_2) = eval_points_quadratic(poly_a, poly_b, &comb_func);
                let evals = vec![eval_0, prev_round_claim - eval_0, eval_2];
                Polynomial::new(&evals)
            };

            // append round's Univariate polynomial to transcript

            // Squeeze Verifier Challenge for next round
            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(challenge.clone());

            // add univariate polynomial for this round to the proof
            round_uni_polys.push(poly);

            // compute next claim
            prev_round_claim = poly.evaluate(&challenge);

            // fix next variable of poly
            poly_a.fix_variable(&challenge);
            poly_b.fix_variable(&challenge);
        }

        SumcheckProof {
            poly: poly_a.clone(),
            sum: sum.clone(),
            round_uni_polys,
        }
    }

    pub fn prove_quadratic_batched<E>(
        sum: &FieldElement<F>,
        poly_a: &mut Vec<DenseMultilinearPolynomial<F>>,
        poly_b: &mut Vec<DenseMultilinearPolynomial<F>>,
        powers: Option<&[FieldElement<F>]>,
        comb_func: E,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>
    where
        E: Fn(&FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync,
    {
        let mut round_uni_polys: Vec<Polynomial<FieldElement<F>>> =
            Vec::with_capacity(poly_a[0].num_vars());
        let mut challenges = Vec::with_capacity(poly_a[0].num_vars());
        let mut prev_round_claim = *sum;

        for _ in 0..poly_a[0].num_vars() {
            let mut evals: Vec<(FieldElement<F>, FieldElement<F>)> = Vec::new();

            for (poly_a, poly_b) in poly_a.iter().zip(poly_b.iter()) {
                let (eval_point_0, eval_point_2) =
                    eval_points_quadratic(poly_a, poly_b, &comb_func);
                evals.push((eval_point_0, eval_point_2));
            }

            // TODO: make optional as we want to perform a batched check outside of this
            let mut evals_combined_0;
            let mut evals_combined_2;
            if let Some(powers) = powers {
                evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * powers[i]).sum();
                evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * powers[i]).sum();
            } else {
                evals_combined_0 = (0..evals.len()).map(|i| evals[i].0).sum();
                evals_combined_2 = (0..evals.len()).map(|i| evals[i].1).sum();
            }

            let evals = vec![
                evals_combined_0,
                prev_round_claim - evals_combined_0,
                evals_combined_2,
            ];
            let poly = Polynomial::new(&evals);

            // TODO append the prover's message to the transcript

            // Squeeze Verifier Challenge for next round
            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(challenge.clone());

            // bound all tables to the verifier's challenege
            for (poly_a, poly_b) in poly_a.iter_mut().zip(poly_b.iter_mut()) {
                poly_a.fix_variable(&challenge);
                poly_b.fix_variable(&challenge);
            }

            prev_round_claim = poly.evaluate(&challenge);
            round_uni_polys.push(poly);
        }

        SumcheckProof {
            poly: poly_a[0].clone(),
            sum: sum.clone(),
            round_uni_polys,
        }
    }

    pub fn prove_cubic<E>(
        sum: &FieldElement<F>,
        poly_a: &DenseMultilinearPolynomial<F>,
        poly_b: &DenseMultilinearPolynomial<F>,
        poly_c: &DenseMultilinearPolynomial<F>,
        comb_func: E,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>
    where
        E: Fn(&FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync, {
        let mut round_uni_polys: Vec<Polynomial<FieldElement<F>>> =
            Vec::with_capacity(poly_a.num_vars());
        let mut challenges = Vec::with_capacity(poly_a.num_vars());
        let mut prev_round_claim = *sum;

        for _ in 0..poly_a.num_vars() {
            let mut evals: Vec<(FieldElement<F>, FieldElement<F>, FieldElement<F>)> = Vec::new();

            #[cfg(feature = "rayon")]

            #[cfg(not(feature = "rayon"))]

            for (poly_a, poly_b) in poly_a.iter().zip(poly_b.iter()) {
                let (eval_point_0, eval_point_2, eval_point_3) =
                    eval_points_cubic(poly_a, poly_b, poly_c, &comb_func);
                evals.push((eval_point_0, eval_point_2, eval_point_3));
            }

            // TODO: make optional as we want to perform a batched check outside of this
            let mut evals_combined_0;
            let mut evals_combined_2;

            let evals = vec![
                evals_combined_0,
                prev_round_claim - evals_combined_0,
                evals_combined_2,
            ];
            let poly = Polynomial::new(&evals);

            // TODO append the prover's message to the transcript

            // Squeeze Verifier Challenge for next round
            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(challenge.clone());

            // bound all tables to the verifier's challenege
            for (poly_a, poly_b) in poly_a.iter().zip(poly_b.iter()) {
                poly_a.fix_variable(&challenge);
                poly_b.fix_variable(&challenge);
            }

            prev_round_claim = poly.evaluate(&challenge);
            round_uni_polys.push(poly);
        }

        SumcheckProof {
            poly: poly_a.clone(),
            sum: sum.clone(),
            round_uni_polys,
        }
    }

    pub fn prove_cubic_batched<E>(
        sum: &FieldElement<F>,
        poly_a: &Vec<DenseMultilinearPolynomial<F>>,
        poly_b: &Vec<DenseMultilinearPolynomial<F>>,
        poly_c: &DenseMultilinearPolynomial<F>,
        powers: Option<&[FieldElement<F>]>,
        comb_func: E,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>
    where
        E: Fn(&FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync, {
        let mut round_uni_polys: Vec<Polynomial<FieldElement<F>>> =
            Vec::with_capacity(poly_a[0].num_vars());
        let mut challenges = Vec::with_capacity(poly_a[0].num_vars());
        let mut prev_round_claim = *sum;

        for _ in 0..poly_a[0].num_vars() {
            let mut evals: Vec<(FieldElement<F>, FieldElement<F>)> = Vec::new();

            for (poly_a, poly_b) in poly_a.iter().zip(poly_b.iter()) {
                let (eval_point_0, eval_point_2) =
                    eval_points_quadratic(poly_a, poly_b, &comb_func);
                evals.push((eval_point_0, eval_point_2));
            }

            // TODO: make optional as we want to perform a batched check outside of this
            let mut evals_combined_0;
            let mut evals_combined_2;
            if let Some(powers) = powers {
                evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * powers[i]).sum();
                evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * powers[i]).sum();
            } else {
                evals_combined_0 = (0..evals.len()).map(|i| evals[i].0).sum();
                evals_combined_2 = (0..evals.len()).map(|i| evals[i].1).sum();
            }

            let evals = vec![
                evals_combined_0,
                prev_round_claim - evals_combined_0,
                evals_combined_2,
            ];
            let poly = Polynomial::new(&evals);

            // TODO append the prover's message to the transcript

            // Squeeze Verifier Challenge for next round
            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(challenge.clone());

            // bound all tables to the verifier's challenege
            for (poly_a, poly_b) in poly_a.iter_mut().zip(poly_b.iter_mut()) {
                poly_a.fix_variable(&challenge);
                poly_b.fix_variable(&challenge);
            }

            prev_round_claim = poly.evaluate(&challenge);
            round_uni_polys.push(poly);
        }

        SumcheckProof {
            poly: poly_a[0].clone(),
            sum: sum.clone(),
            round_uni_polys,
        }
    }

    // Special instance of sumcheck for a cubic polynomial with an additional additive term:
    // this is used in Spartan: (a * ((b * c) - d))
    pub fn prove_cubic_additive_term<E>(
        sum: &FieldElement<F>,
        poly_a: &DenseMultilinearPolynomial<F>,
        poly_b: &DenseMultilinearPolynomial<F>,
        poly_c: &DenseMultilinearPolynomial<F>,
        poly_d: &DenseMultilinearPolynomial<F>,
        comb_func: F,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F>
    where
        E: Fn(&FieldElement<F>, &FieldElement<F>) -> FieldElement<F> + Sync, {
        todo!()
    }

    // Create a test for this
    pub fn prove_single(
        poly: &DenseMultilinearPolynomial<F>,
        sum: &FieldElement<F>,
        transcript: &mut impl Transcript,
    ) -> SumcheckProof<F> {
        let mut round_uni_polys: Vec<Polynomial<FieldElement<F>>> =
            Vec::with_capacity(poly.num_vars());
        let mut challenges = Vec::with_capacity(poly.num_vars());

        let mut prev_round_claim = *sum;
        let mut round_poly = *poly;

        // Number round = num vars
        for _ in 0..poly.num_vars() {
            let mut eval_points = vec![FieldElement::zero(); round_poly.num_vars() + 1];
            // TODO: add multicore by flagging
            // Compute evaluation points of the Dense Multilinear Poly
            let round_uni_poly = {
                let mle_half = poly.len() / 2;
                // TODO: push check/error for empty poly to start of proving or into multilinear poly so we eliminate this problem entirely.
                let eval_0 = (0..mle_half)
                    .map(|i| poly[i])
                    .reduce(|a, b| (a + b))
                    .unwrap();
                // We evaluate the poly at each round and each random challenge at 0, 1 we can compute both of these evaluations by summing over the boolearn hypercube for 0, 1 at the fixed point
                // An additional optimization is to sum over eval_0 and compute eval_1 = prev_round_claim - eval_0;
                let evals = vec![eval_0, prev_round_claim - eval_0];
                Polynomial::new(&eval_points)
            };

            // TODO: Append poly to transcript -> Modify Transcript

            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            challenges.push(challenge.clone());

            // add univariate polynomial for this round to the proof
            round_uni_polys.push(round_uni_poly);

            // grab next claim
            prev_round_claim = round_uni_poly.evaluate(&challenge);

            // takes mutable reference and fixes poly at challenge
            // On each round we evaluate over the hypercube to generate the univariate polynomial for this round. Then we fix the challenge for the next variable,
            // reassign and start the next round with the fixed variable. Each round the poly decreases in size
            poly.fix_variable(&challenge);
        }

        SumcheckProof {
            poly: poly.clone(),
            sum: sum.clone(),
            round_uni_polys,
        }
    }

    // Verifies a sumcheck proof returning the claimed evaluation and random points used during sumcheck rounds
    pub fn verify(
        proof: SumcheckProof<F>,
        transcript: &mut impl Transcript,
    ) -> (FieldElement<F>, Vec<FieldElement<F>>) {
        let mut e = proof.sum.clone();
        let mut r: Vec<FieldElement<F>> = Vec::with_capacity(proof.poly.num_vars());

        // verify there is a univariate polynomial for each round
        // TODO: push this if up so that the proof struct enforces this invariant

        for poly in proof.round_uni_polys {
            // Verify degree bound

            // check if G_k(0) + G_k(1) = e
            assert_eq!(
                poly.evaluate(&FieldElement::<F>::zero()) + poly.evaluate(&FieldElement::one()),
                e
            );
            //transcript.append(poly);

            let challenge = FieldElement::from_bytes_be(&transcript.challenge()).unwrap();
            r.push(challenge.clone());

            e = poly.evaluate(&challenge);
        }
        (proof.sum, r)
    }
}

#[cfg(test)]
mod test {
    use core::num;

    use crate::fiat_shamir::default_transcript::DefaultTranscript;
    use crate::subprotocols::sumcheck::{Sumcheck, SumcheckProof};
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;
    use lambdaworks_math::field::traits::IsField;
    use lambdaworks_math::polynomial::dense_multilinear_poly::DenseMultilinearPolynomial;

    type F = Babybear31PrimeField;
    type FE = FieldElement<F>;

    pub fn index_to_field_bitvector<F: IsField>( value: usize, bits: usize) -> Vec<FieldElement<F>> {
        let mut vec: Vec<FieldElement<F>> = Vec::with_capacity(bits);

        for i in (0..bits).rev() {
            if (value >> i) & 1 == 1 {
                vec.push(FieldElement::one());
            } else {
                vec.push(FieldElement::zero());
            }
        }
        vec
    }

    #[test]
    fn prove_cubic() {
    // Create three dense polynomials (all the same)
    let num_vars = 3;
    let num_evals = (2usize).pow(num_vars as u32);
    let mut evals: Vec<FieldElement<F>> = Vec::with_capacity(num_evals);
    for i in 0..num_evals {
     evals.push(FieldElement::from(8 + i as u64));
    }

    let a: DenseMultilinearPolynomial<F> = DenseMultilinearPolynomial::new(evals.clone());
    let b: DenseMultilinearPolynomial<F> = DenseMultilinearPolynomial::new(evals.clone());
    let c: DenseMultilinearPolynomial<F> = DenseMultilinearPolynomial::new(evals.clone());

    let mut claim = FieldElement::<F>::zero();
    for i in 0..num_evals {

      claim += a.evaluate(&index_to_field_bitvector(i, num_vars)).unwrap()
        * b.evaluate(&index_to_field_bitvector(i, num_vars)).unwrap()
        * c.evaluate(&index_to_field_bitvector(i, num_vars)).unwrap();
    }
    let mut polys = [a.clone(), b.clone(), c.clone()];

    let comb_func_prod =
      |polys: &[FieldElement<F>; 3]| -> FieldElement<F> { polys.iter().fold(FieldElement::one(), |acc, poly| acc * *poly) };

    let r = vec![FieldElement::from(3), FieldElement::from(1), FieldElement::from(3)]; // point 0,0,0 within the boolean hypercube

    let mut transcript = DefaultTranscript::new();
    let proof =
      Sumcheck::<F>::prove_cubic(
        &claim,
        &poly_a,
        &poly_b,
        &poly_c,
        comb_func_prod,
        &mut transcript,
      );

    let mut transcript = DefaultTranscript::new();
    let verify_result = Sumcheck::verify(proof, &mut transcript);
    assert!(verify_result.is_ok());

    let (verify_evaluation, verify_randomness) = verify_result.unwrap();
    assert_eq!(prove_randomness, verify_randomness);
    assert_eq!(prove_randomness, r);

    // Consider this the opening proof to a(r) * b(r) * c(r)
    let a = a.evaluate(prove_randomness.as_slice()).unwrap();
    let b = b.evaluate(prove_randomness.as_slice()).unwrap();
    let c = c.evaluate(prove_randomness.as_slice()).unwrap();

    let oracle_query = a * b * c;
    assert_eq!(verify_evaluation, oracle_query);
    }

    #[test]
    fn prove_quad() {
        
    }

    #[test]
    fn prove_single() {

    }
}
