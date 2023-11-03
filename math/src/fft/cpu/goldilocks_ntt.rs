use crate::field::{fields::u64_goldilocks_field::Goldilocks64Field, element::FieldElement, traits::IsField};

pub type F = FieldElement<Goldilocks64Field>;

use rand::{thread_rng,Rng};


pub const fn log2_ceil(value: usize) -> usize {
    let mut result = 0;
    while (1 << result) < value {
        result += 1;
    }
    result
}

pub fn get_twiddle_fwd(n:usize) -> F{
    let rou_array = vec![
        1,
        18446744069414584320,
        281474976710656,
        18446744069397807105,
        17293822564807737345,
        70368744161280,
        549755813888,
        17870292113338400769,
        13797081185216407910,
        1803076106186727246,
        11353340290879379826,
        455906449640507599,
        17492915097719143606,
        1532612707718625687,
        16207902636198568418,
        17776499369601055404,
        6115771955107415310,
        12380578893860276750,
        9306717745644682924,
        18146160046829613826,
        3511170319078647661,
        17654865857378133588,
        5416168637041100469,
        16905767614792059275,
        9713644485405565297,
        5456943929260765144,
        17096174751763063430,
        1213594585890690845,
        6414415596519834757,
        16116352524544190054,
        9123114210336311365,
        4614640910117430873,
        1753635133440165772,
    ];

    F::from(rou_array[n])

}

pub fn get_twiddle_rev(n:usize) -> F{
    let rou_array = vec![
        1,
        18446744069414584320,
        18446462594437873665,
        1099511627520,
        68719476736,
        18446744069414322177,
        18302628881338728449,
        18442240469787213841,
        2117504431143841456,
        4459017075746761332,
        4295002282146690441,
        8548973421900915981,
        11164456749895610016,
        3968367389790187850,
        4654242210262998966,
        1553425662128427817,
        7868944258580147481,
        14744321562856667967,
        2513567076326282710,
        5089696809409609209,
        17260140776825220475,
        11898519751787946856,
        15307271466853436433,
        5456584715443070302,
        1219213613525454263,
        13843946492009319323,
        16884827967813875098,
        10516896061424301529,
        4514835231089717636,
        16488041148801377373,
        16303955383020744715,
        10790884855407511297,
        8554224884056360729,
    ];

    F::from(rou_array[n])

}

pub fn radix_2_ntt(n:usize, values: &mut [F]){

    let length = values.len();

    if n==0 || length<1 {
        return;
    }

    let half = 1 << (n-1);

    radix_2_ntt(n-1, &mut values[..half]);
    radix_2_ntt(n-1,&mut values[half..]);

    // Will change.
    let wn = get_twiddle_fwd(n);
    let mut w = F::one();

    for i in 0..half {
        let a = values[i];
        let b = F::from(Goldilocks64Field::mul(values[i + half].value(), w.value()));

        values[i] = F::from(Goldilocks64Field::add((a).value(),b.value()));

        values[i+half] = F::from(Goldilocks64Field::sub(a.value(),b.value()));

        w = F::from(Goldilocks64Field::mul(w.value(), wn.value()));    }

}

pub fn radix_2_intt(n:usize, values: &mut [F]){
    
    let length = values.len();

    if n==0 || length<1 {
        return;
    }

    let half = 1 << (n-1);

    let wn = get_twiddle_rev(n);
    let mut w = F::one();

    for i in 0..half {
        let a = values[i];
        let b = values[i+half];

        values[i] = F::from(Goldilocks64Field::add(a.value(),b.value()));

        values[i+half] = F::from(Goldilocks64Field::mul(&Goldilocks64Field::sub(a.value(),b.value()),w.value()));

        w = F::from(Goldilocks64Field::mul(w.value(), wn.value()));
    }

    radix_2_intt(n-1, &mut values[..half]);
    radix_2_intt(n-1,&mut values[half..]);

}


pub fn apply_intt(n:usize, values: &mut [F]){

    radix_2_intt(n, values);
    let l = values.len();
    let inv = F::from(l as u64).inv().unwrap();
    
    for e in values.iter_mut().take(l){
        *e = *e * inv;
    }


}


pub fn randomize(n : usize) -> Vec<F>{
    let mut rng = thread_rng();

    let mut random_vector:Vec<F> = Vec::with_capacity(n);

    for _ in 0..n{
        random_vector.push(F::from(rng.gen::<u64>()));
    }       
    random_vector
}

// #[test]
// fn test_on_data(){

//     let a0 = F::from(2);
//     let a1 = F::from(3);
//     let a2 = F::from(1337);
//     let a3 = F::from(65);
//     let a4 = F::from(6);

//     let mut vec_array = vec![a0,a1,a2,a3,a4];
    
//     let n = log2_ceil(vec_array.len());

//     let add = (1<<n) - vec_array.len();

//     let zeros = vec![F::from(0);add];
    
//     vec_array.extend(zeros);

//     let vec_array = vec_array.as_mut_slice();

//     println!("{:?} \n",vec_array);

//     radix_2_ntt(n, vec_array);

//     println!("{:?}\n",vec_array);

//     radix_2_intt(n, vec_array);

//     println!("{:?}\n",vec_array);
// }


#[test]
pub fn test_with_random_data(){

    // Case of # of elements = 1024, no need for padding.
    let mut input = randomize(1024);

    let original = input.clone();

    let n = log2_ceil(input.len());

    radix_2_ntt(n,&mut input);
    
    apply_intt(n, &mut input);

    assert_eq!(original,input);

}