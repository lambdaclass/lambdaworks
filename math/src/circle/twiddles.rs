// fn compute_twiddles<F: ComplexExtendable>(domain: CircleDomain<F>) -> Vec<Vec<F>> {
//     assert!(domain.log_n >= 1);
//     let mut pts = domain.coset0().collect_vec();
//     reverse_slice_index_bits(&mut pts);
//     let mut twiddles = vec![pts.iter().map(|p| p.y).collect_vec()];
//     if domain.log_n >= 2 {
//         twiddles.push(pts.iter().step_by(2).map(|p| p.x).collect_vec());
//         for i in 0..(domain.log_n - 2) {
//             let prev = twiddles.last().unwrap();
//             assert_eq!(prev.len(), 1 << (domain.log_n - 2 - i));
//             let cur = prev
//                 .iter()
//                 .step_by(2)
//                 .map(|x| x.square().double() - F::one())
//                 .collect_vec();
//             twiddles.push(cur);
//         }
//     }
//     twiddles
// }


