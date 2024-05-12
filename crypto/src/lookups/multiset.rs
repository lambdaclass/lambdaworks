use lambdaworks_math::field::{
        element::FieldElement,
        traits::IsField,
    };
use core::ops::{Add, Mul};

#[derive(Clone, Debug)]
pub struct MultiSet<F: IsField>(pub Vec<FieldElement<F>>);

impl<F: IsField> MultiSet<F> {
    // Creates an empty Multiset
    pub fn new() -> MultiSet<F> {
        MultiSet(vec![])
    }
    /// Pushes a value onto the end of the set
    pub fn push(&mut self, value: FieldElement<F>) {
        self.0.push(value)
    }
    /// Pushes 'n' elements into the multiset
    pub fn extend(&mut self, n: usize, value: FieldElement<F>) {
        let elements = vec![value; n];
        self.0.extend(elements);
    }
    /// Fetches last element in multiset
    /// Panics if there are no elements
    pub fn last(&self) -> FieldElement<F> {
        self.0.last().unwrap().clone()
    }
    fn from_slice(slice: &[FieldElement<F>]) -> MultiSet<F> {
        MultiSet(slice.to_vec())
    }
    /// Returns the cardinality of the multiset
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Concatenates two sets together
    /// Does not sort the concatenated multisets
    pub fn concatenate(&self, other: &MultiSet<F>) -> MultiSet<F> {
        let mut result: Vec<FieldElement<F>> = Vec::with_capacity(self.0.len() + other.0.len());
        result.extend(self.0.clone());
        result.extend(other.0.clone());
        MultiSet(result)
    }

    /// Returns the position of the element in the Multiset
    /// Panics if element is not in the Multiset
    fn position(&self, element: &FieldElement<F>) -> usize {
        let index = self.0.iter().position(|x| *x == *element).unwrap();
        index
    }

    /// Performs a element-wise insertion into the second multiset
    /// Example: f {1,2,3,1} t : {3,1,2,3}
    /// We now take each element from f and find the element in `t` then insert the element from `f` into right next to it's duplicate
    /// We are assuming that `f` is contained in `t`
    pub fn concatenate_and_sort(&self, t: &MultiSet<F>) -> MultiSet<F> {
        assert!(self.is_subset_of(t));
        let mut result = t.clone();

        for element in self.0.iter() {
            let index = result.position(element);
            result.0.insert(index, element.clone());
        }

        result
    }

    /// Checks whether self is a subset of other
    pub fn is_subset_of(&self, other: &MultiSet<F>) -> bool {
        let mut is_subset = true;

        for x in self.0.iter() {
            is_subset = other.contains(x);
            if is_subset == false {
                break;
            }
        }
        is_subset
    }
    /// Checks if an element is in the MultiSet
    pub fn contains(&self, element: &FieldElement<F>) -> bool {
        self.0.contains(element)
    }
    /// Splits a multiset into halves as specified by the paper
    /// If s = [1,2,3,4,5,6,7], we can deduce n using |s| = 2 * n + 1 = 7
    /// n is therefore 3
    /// We split s into two MultiSets of size n+1 each
    /// s_0 = [1,2,3,4] ,|s_0| = n+1 = 4
    /// s_1 = [4,5,6,7] , |s_1| = n+1 = 4
    /// Notice that the last element of the first half equals the first element in the second half
    /// This is specified in the paper
    pub fn halve(&self) -> (MultiSet<F>, MultiSet<F>) {
        let length = self.0.len();

        let first_half = MultiSet::from_slice(&self.0[0..=length / 2]);
        let second_half = MultiSet::from_slice(&self.0[length / 2..]);

        (first_half, second_half)
    }
    /// Aggregates multisets together using a random challenge
    /// Eg. for three sets A,B,C and a random challenge `k`
    /// The aggregate is k^0 *A + k^1 * B + k^2 * C
    pub fn aggregate(sets: Vec<&MultiSet<F>>, challenge: FieldElement<F>) -> MultiSet<F> {
        // First find the set with the most elements
        let mut max = 0usize;
        for set in sets.iter() {
            if set.len() > max {
                max = set.len()
            }
        }

        let mut result = MultiSet(vec![FieldElement::zero(); max]);
        let mut powers = FieldElement::one();

        for set in sets {
            let intermediate_set = set * powers.clone();

            result = result + intermediate_set;

            powers = powers * challenge.clone();
        }

        result
    }
}

impl<F: IsField> Add for MultiSet<F> {
    type Output = MultiSet<F>;
    fn add(self, other: MultiSet<F>) -> Self::Output {
        let result = self
            .0
            .into_iter()
            .zip(other.0.iter())
            .map(|(x, y)| x + y)
            .collect();

        MultiSet(result)
    }
}

impl<F: IsField> Mul<FieldElement<F>> for MultiSet<F> {
    type Output = MultiSet<F>;
    fn mul(self, other: FieldElement<F>) -> Self::Output {
        let result = self.0.into_iter().map(|x| x * other.clone()).collect();
        MultiSet(result)
    }
}

impl<F: IsField> Mul<FieldElement<F>> for &MultiSet<F> {
    type Output = MultiSet<F>;
    fn mul(self, other: FieldElement<F>) -> Self::Output {
        let result = self.0.iter().map(|x| other.clone() * x).collect();
        MultiSet(result)
    }
}

/// Equality operator overloading for field elements
impl<F: IsField> PartialEq<MultiSet<F>> for MultiSet<F>
{
    fn eq(&self, other: &MultiSet<F>) -> bool {
        self.0 == other.0
    }
}

impl<F: IsField> Eq for MultiSet<F>{}


#[cfg(test)]
mod test {

    type FE = FieldElement<Babybear31PrimeField>;
    type F = Babybear31PrimeField;

    use lambdaworks_math::{field::fields::fft_friendly::babybear::Babybear31PrimeField, polynomial::Polynomial};

    use super::*;

    #[test]
    fn test_concatenate() {
        let mut a = MultiSet::new();
        a.push(FE::from(1u64));
        a.push(FE::from(2u64));
        a.push(FE::from(3u64));
        let mut b = MultiSet::new();
        b.push(FE::from(4u64));
        b.push(FE::from(5u64));
        b.push(FE::from(6u64));

        let c = a.concatenate(&b);

        let expected_set = MultiSet(vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
            FE::from(5u64),
            FE::from(6u64),
        ]);
        assert_eq!(expected_set, c);
    }

    #[test]
    fn test_halve() {
        let mut a = MultiSet::new();
        a.push(FE::from(1u64));
        a.push(FE::from(2u64));
        a.push(FE::from(3u64));
        a.push(FE::from(4u64));
        a.push(FE::from(5u64));
        a.push(FE::from(6u64));
        a.push(FE::from(7u64));

        let (h_1, h_2) = a.halve();
        assert_eq!(h_1.len(), 4);
        assert_eq!(h_2.len(), 4);

        assert_eq!(
            MultiSet(vec![
                FE::from(1u64),
                FE::from(2u64),
                FE::from(3u64),
                FE::from(4u64)
            ]),
            h_1
        );

        assert_eq!(
            MultiSet(vec![
                FE::from(4u64),
                FE::from(5u64),
                FE::from(6u64),
                FE::from(7u64)
            ]),
            h_2
        );

        // Last element in the first half should equal first element in the second half
        assert_eq!(h_1.0.last().unwrap(), &h_2.0[0])
    }

    #[test]
    fn test_to_polynomial() {
        let mut a = MultiSet::new();
        a.push(FE::from(0u64));
        a.push(FE::from(1u64));
        a.push(FE::from(2u64));
        a.push(FE::from(3u64));
        a.push(FE::from(4u64));
        a.push(FE::from(5u64));
        a.push(FE::from(6u64));
        a.push(FE::from(7u64));

        let a_poly = Polynomial::interpolate_fft::<F>(&a.0).unwrap();

        assert_eq!(a_poly.degree(), 7)
    }

    #[test]
    fn test_is_subset() {
        let mut a = MultiSet::new();
        a.push(FE::from(1u64));
        a.push(FE::from(2u64));
        a.push(FE::from(3u64));
        a.push(FE::from(4u64));
        a.push(FE::from(5u64));
        a.push(FE::from(6u64));
        a.push(FE::from(7u64));
        let mut b = MultiSet::new();
        b.push(FE::from(1u64));
        b.push(FE::from(2u64));
        let mut c = MultiSet::new();
        c.push(FE::from(100u64));

        assert!(b.is_subset_of(&a));
        assert!(!c.is_subset_of(&a));
    }

    #[test]
    fn test_sort_by() {
        let mut f = MultiSet::new();
        f.push(FE::from(2u64));
        f.push(FE::from(1u64));
        f.push(FE::from(2u64));
        f.push(FE::from(4u64));
        f.push(FE::from(3u64));

        let mut t = MultiSet::new();
        t.push(FE::from(3u64));
        t.push(FE::from(1u64));
        t.push(FE::from(2u64));
        t.push(FE::from(4u64));

        let sorted_s = f.concatenate_and_sort(&t);

        let mut expected_sorted_s = MultiSet::new();
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(1u64));
        expected_sorted_s.push(FE::from(1u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(4u64));
        expected_sorted_s.push(FE::from(4u64));

        assert_eq!(expected_sorted_s, sorted_s);
    }

    #[test]
    fn test_concate_sort() {
        let mut f = MultiSet::new();
        f.push(FE::from(1u64));
        f.push(FE::from(2u64));

        // Table of values
        let mut t = MultiSet::new();
        t.push(FE::from(2u64));
        t.push(FE::from(1u64));
        t.push(FE::from(2u64));

        let mut expected_s = MultiSet::new();
        expected_s.push(FE::from(2u64));
        expected_s.push(FE::from(2u64));
        expected_s.push(FE::from(1u64));
        expected_s.push(FE::from(1u64));
        expected_s.push(FE::from(2u64));

        let sorted_s_1 = f.concatenate_and_sort(&t);
        assert_eq!(sorted_s_1, expected_s)
    }

    #[test]
    fn test_concate_sort2() {
        let mut f = MultiSet::new();
        f.push(FE::from(1u64));
        f.push(FE::from(1u64));
        f.push(FE::from(2u64));
        f.push(FE::from(2u64));
        f.push(FE::from(3u64));
        f.push(FE::from(3u64));
        f.push(FE::from(3u64));

        // Table of values
        let mut t = MultiSet::new();
        t.push(FE::from(1u64));
        t.push(FE::from(1u64));
        t.push(FE::from(2u64));
        t.push(FE::from(2u64));
        t.push(FE::from(3u64));
        t.push(FE::from(3u64));
        t.push(FE::from(4u64));
        t.push(FE::from(5u64));

        let mut expected_sorted_s = MultiSet::new();
        expected_sorted_s.push(FE::from(1u64));
        expected_sorted_s.push(FE::from(1u64));
        expected_sorted_s.push(FE::from(1u64));
        expected_sorted_s.push(FE::from(1u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(2u64));
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(3u64));
        expected_sorted_s.push(FE::from(4u64));
        expected_sorted_s.push(FE::from(5u64));

        let sorted_s = f.concatenate_and_sort(&t);
        assert_eq!(sorted_s.len(), f.len() + t.len());
        assert_eq!(sorted_s.len(), expected_sorted_s.len());
        assert_eq!(sorted_s, expected_sorted_s)
    }
}