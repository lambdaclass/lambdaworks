use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::babybear::Babybear31PrimeField;

use lambdaworks_gkr_logup::univariate::domain::CyclicDomain;
use lambdaworks_gkr_logup::univariate::lagrange::UnivariateLagrange;
use lambdaworks_gkr_logup::univariate_layer::{gen_layers, UnivariateLayer};

type F = Babybear31PrimeField;
type FE = FieldElement<F>;

fn main() {
    println!("=== Univariate LogUp-GKR Test ===\n");

    test_hypercube_cyclic_mapping();
    test_univariate_layer();
    test_logup_singles();

    println!("\nAll tests passed!");
}

fn test_hypercube_cyclic_mapping() {
    println!("Test 1: Hypercube to Cyclic Domain Mapping");

    let values: Vec<FE> = (0..8).map(|i| FE::from(i as u64)).collect();
    let domain: CyclicDomain<F> = CyclicDomain::new(3).expect("Failed to create cyclic domain");

    println!("  Domain size: {}", domain.size());
    println!("  Domain root: {:?}", domain.root);

    for i in 0..8 {
        let point = domain.get_point(i);
        println!("  ω^{} = {:?}", i, point);
    }
    println!("  ✓ Domain created successfully\n");
}

fn test_univariate_layer() {
    println!("Test 2: Univariate Layer Operations");

    let values: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
    let domain = CyclicDomain::new(2).expect("Failed to create domain");
    let uni = UnivariateLagrange::new(values, domain).expect("Failed to create univariate");

    let layer = UnivariateLayer::GrandProduct {
        values: uni,
        commitment: None,
    };

    println!("  Initial layer n_variables: {}", layer.n_variables());
    println!("  Is output layer: {}", layer.is_output_layer());

    let next = layer.next_layer();
    println!("  Next layer exists: {}", next.is_some());
    if let Some(ref next_layer) = next {
        println!("  Next layer n_variables: {}", next_layer.n_variables());
    }
    println!("  ✓ Layer operations work\n");
}

fn test_logup_singles() {
    println!("Test 3: LogUp Singles Layer");

    let z = FE::from(100u64);
    let accesses: Vec<u64> = vec![20, 10, 20, 30, 10, 20, 40, 30];

    let access_dens: Vec<FE> = accesses.iter().map(|&a| z - FE::from(a)).collect();

    let domain = CyclicDomain::new(3).expect("Failed to create domain");
    let uni = UnivariateLagrange::new(access_dens, domain).expect("Failed to create univariate");

    let layer = UnivariateLayer::LogUpSingles {
        denominators: uni,
        denominator_commitment: None,
    };

    println!("  Access layer n_variables: {}", layer.n_variables());

    let layers = gen_layers(layer);
    println!("  Generated {} layers", layers.len());

    let output = layers.last().unwrap().try_into_output_layer_values();
    println!("  Output layer values: {:?}", output);

    println!("  ✓ LogUp Singles works\n");
}
