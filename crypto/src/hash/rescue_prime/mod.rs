use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lazy_static::lazy_static;
use std::io::Read;

use sha3::{
    digest::{ExtendableOutput, Update},
    Shake256,
};
// Rescue Prime Optimized implementation based on
// https://github.com/ASDiscreteMathematics/rpo and
// https://github.com/0xPolygonMiden/crypto/tree/main/src/hash/rescue/rpo
pub const ALPHA: u64 = 7;
pub const ALPHA_INV: u64 = 10540996611094048183;

type Fp = FieldElement<Goldilocks64Field>;

#[derive(Clone)]
#[allow(dead_code)]
enum MdsMethod {
    MatrixMultiplication,
    Ntt,
    Karatsuba,
}

pub struct RescuePrimeOptimized<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize> {
    m: usize,
    capacity: usize,
    rate: usize,
    round_constants: Vec<Fp>,
    mds_matrix: Vec<Vec<Fp>>,
    mds_vector: Vec<Fp>,
    alpha: u64,
    alpha_inv: u64,
    mds_method: MdsMethod,
}

impl<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize> Default
    for RescuePrimeOptimized<SECURITY_LEVEL, NUM_FULL_ROUNDS>
{
    fn default() -> Self {
        Self::new(MdsMethod::MatrixMultiplication)
    }
}

impl<const SECURITY_LEVEL: usize, const NUM_FULL_ROUNDS: usize>
    RescuePrimeOptimized<SECURITY_LEVEL, NUM_FULL_ROUNDS>
{
    const P: u64 = 18446744069414584321; // p = 2^64 - 2^32 + 1

    fn new(mds_method: MdsMethod) -> Self {
        assert!(SECURITY_LEVEL == 128 || SECURITY_LEVEL == 160);

        let (m, capacity) = if SECURITY_LEVEL == 128 {
            (12, 4)
        } else {
            (16, 6)
        };
        let rate = m - capacity;

        let mds_vector = Self::get_mds_vector(m);
        let mds_matrix = Self::generate_circulant_matrix(&mds_vector);

        //let round_constants = Self::instantiate_round_constants(
        //     Self::P,
        //     m,
        //     capacity,
        //     SECURITY_LEVEL,
        //     NUM_FULL_ROUNDS,
        // );
        let round_constants = vec![
            Fp::from(5789762306288267392u64),
            Fp::from(6522564764413701783u64),
            Fp::from(17809893479458208203u64),
            Fp::from(107145243989736508u64),
            Fp::from(6388978042437517382u64),
            Fp::from(15844067734406016715u64),
            Fp::from(9975000513555218239u64),
            Fp::from(3344984123768313364u64),
            Fp::from(9959189626657347191u64),
            Fp::from(12960773468763563665u64),
            Fp::from(9602914297752488475u64),
            Fp::from(16657542370200465908u64),
            Fp::from(6077062762357204287u64),
            Fp::from(15277620170502011191u64),
            Fp::from(5358738125714196705u64),
            Fp::from(14233283787297595718u64),
            Fp::from(13792579614346651365u64),
            Fp::from(11614812331536767105u64),
            Fp::from(14871063686742261166u64),
            Fp::from(10148237148793043499u64),
            Fp::from(4457428952329675767u64),
            Fp::from(15590786458219172475u64),
            Fp::from(10063319113072092615u64),
            Fp::from(14200078843431360086u64),
            Fp::from(12987190162843096997u64),
            Fp::from(653957632802705281u64),
            Fp::from(4441654670647621225u64),
            Fp::from(4038207883745915761u64),
            Fp::from(5613464648874830118u64),
            Fp::from(13222989726778338773u64),
            Fp::from(3037761201230264149u64),
            Fp::from(16683759727265180203u64),
            Fp::from(8337364536491240715u64),
            Fp::from(3227397518293416448u64),
            Fp::from(8110510111539674682u64),
            Fp::from(2872078294163232137u64),
            Fp::from(6202948458916099932u64),
            Fp::from(17690140365333231091u64),
            Fp::from(3595001575307484651u64),
            Fp::from(373995945117666487u64),
            Fp::from(1235734395091296013u64),
            Fp::from(14172757457833931602u64),
            Fp::from(707573103686350224u64),
            Fp::from(15453217512188187135u64),
            Fp::from(219777875004506018u64),
            Fp::from(17876696346199469008u64),
            Fp::from(17731621626449383378u64),
            Fp::from(2897136237748376248u64),
            Fp::from(18072785500942327487u64),
            Fp::from(6200974112677013481u64),
            Fp::from(17682092219085884187u64),
            Fp::from(10599526828986756440u64),
            Fp::from(975003873302957338u64),
            Fp::from(8264241093196931281u64),
            Fp::from(10065763900435475170u64),
            Fp::from(2181131744534710197u64),
            Fp::from(6317303992309418647u64),
            Fp::from(1401440938888741532u64),
            Fp::from(8884468225181997494u64),
            Fp::from(13066900325715521532u64),
            Fp::from(8023374565629191455u64),
            Fp::from(15013690343205953430u64),
            Fp::from(4485500052507912973u64),
            Fp::from(12489737547229155153u64),
            Fp::from(9500452585969030576u64),
            Fp::from(2054001340201038870u64),
            Fp::from(12420704059284934186u64),
            Fp::from(355990932618543755u64),
            Fp::from(9071225051243523860u64),
            Fp::from(12766199826003448536u64),
            Fp::from(9045979173463556963u64),
            Fp::from(12934431667190679898u64),
            Fp::from(5674685213610121970u64),
            Fp::from(5759084860419474071u64),
            Fp::from(13943282657648897737u64),
            Fp::from(1352748651966375394u64),
            Fp::from(17110913224029905221u64),
            Fp::from(1003883795902368422u64),
            Fp::from(4141870621881018291u64),
            Fp::from(8121410972417424656u64),
            Fp::from(14300518605864919529u64),
            Fp::from(13712227150607670181u64),
            Fp::from(17021852944633065291u64),
            Fp::from(6252096473787587650u64),
            Fp::from(18389244934624494276u64),
            Fp::from(16731736864863925227u64),
            Fp::from(4440209734760478192u64),
            Fp::from(17208448209698888938u64),
            Fp::from(8739495587021565984u64),
            Fp::from(17000774922218161967u64),
            Fp::from(13533282547195532087u64),
            Fp::from(525402848358706231u64),
            Fp::from(16987541523062161972u64),
            Fp::from(5466806524462797102u64),
            Fp::from(14512769585918244983u64),
            Fp::from(10973956031244051118u64),
            Fp::from(4887609836208846458u64),
            Fp::from(3027115137917284492u64),
            Fp::from(9595098600469470675u64),
            Fp::from(10528569829048484079u64),
            Fp::from(7864689113198939815u64),
            Fp::from(17533723827845969040u64),
            Fp::from(5781638039037710951u64),
            Fp::from(17024078752430719006u64),
            Fp::from(109659393484013511u64),
            Fp::from(7158933660534805869u64),
            Fp::from(2955076958026921730u64),
            Fp::from(7433723648458773977u64),
            Fp::from(6982293561042362913u64),
            Fp::from(14065426295947720331u64),
            Fp::from(16451845770444974180u64),
            Fp::from(7139138592091306727u64),
            Fp::from(9012006439959783127u64),
            Fp::from(14619614108529063361u64),
            Fp::from(1394813199588124371u64),
            Fp::from(4635111139507788575u64),
            Fp::from(16217473952264203365u64),
            Fp::from(10782018226466330683u64),
            Fp::from(6844229992533662050u64),
            Fp::from(7446486531695178711u64),
            Fp::from(16308865189192447297u64),
            Fp::from(11977192855656444890u64),
            Fp::from(12532242556065780287u64),
            Fp::from(14594890931430968898u64),
            Fp::from(7291784239689209784u64),
            Fp::from(5514718540551361949u64),
            Fp::from(10025733853830934803u64),
            Fp::from(7293794580341021693u64),
            Fp::from(6728552937464861756u64),
            Fp::from(6332385040983343262u64),
            Fp::from(13277683694236792804u64),
            Fp::from(2600778905124452676u64),
            Fp::from(3736792340494631448u64),
            Fp::from(577852220195055341u64),
            Fp::from(6689998335515779805u64),
            Fp::from(13886063479078013492u64),
            Fp::from(14358505101923202168u64),
            Fp::from(7744142531772274164u64),
            Fp::from(16135070735728404443u64),
            Fp::from(12290902521256031137u64),
            Fp::from(12059913662657709804u64),
            Fp::from(16456018495793751911u64),
            Fp::from(4571485474751953524u64),
            Fp::from(17200392109565783176u64),
            Fp::from(7123075680859040534u64),
            Fp::from(1034205548717903090u64),
            Fp::from(7717824418247931797u64),
            Fp::from(3019070937878604058u64),
            Fp::from(11403792746066867460u64),
            Fp::from(10280580802233112374u64),
            Fp::from(337153209462421218u64),
            Fp::from(13333398568519923717u64),
            Fp::from(3596153696935337464u64),
            Fp::from(8104208463525993784u64),
            Fp::from(14345062289456085693u64),
            Fp::from(17036731477169661256u64),
            Fp::from(17130398059294018733u64),
            Fp::from(519782857322261988u64),
            Fp::from(9625384390925085478u64),
            Fp::from(1664893052631119222u64),
            Fp::from(7629576092524553570u64),
            Fp::from(3485239601103661425u64),
            Fp::from(9755891797164033838u64),
            Fp::from(15218148195153269027u64),
            Fp::from(16460604813734957368u64),
            Fp::from(9643968136937729763u64),
            Fp::from(3611348709641382851u64),
            Fp::from(18256379591337759196u64),
        ];

        Self {
            m,
            capacity,
            rate,
            round_constants,
            mds_matrix,
            mds_vector,
            alpha: ALPHA,
            alpha_inv: ALPHA_INV,
            mds_method,
        }
    }

    pub fn apply_inverse_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA_INV);
        }

        /*let mut t1 = state.to_vec();
        for x in t1.iter_mut() {
            *x = x.square();
        }

        let mut t2 = t1.clone();
        for x in t2.iter_mut() {
            *x = x.square();
        }

        let t3 = Self::exp_acc(&t2, &t2, 3);
        let t4 = Self::exp_acc(&t3, &t3, 6);
        let t5 = Self::exp_acc(&t4, &t4, 12);
        let t6 = Self::exp_acc(&t5, &t3, 6);
        let t7 = Self::exp_acc(&t6, &t6, 31);

        for i in 0..state.len() {
            let a = (t7[i].square() * t6[i].clone()).square().square();
            let b = t1[i].clone() * t2[i].clone() * state[i].clone();
            state[i] = a * b;
        }*/
    }

    #[inline(always)]
    fn exp_acc(base: &[Fp], tail: &[Fp], num_squarings: usize) -> Vec<Fp> {
        let mut result = base.to_vec();
        for x in result.iter_mut() {
            for _ in 0..num_squarings {
                *x = x.square();
            }
        }
        result
            .iter_mut()
            .zip(tail.iter())
            .for_each(|(r, t)| *r = *r * t.clone());
        result
    }

    fn get_mds_vector(m: usize) -> Vec<Fp> {
        match m {
            12 => vec![7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]
                .into_iter()
                .map(Fp::from)
                .collect(),
            16 => vec![
                256, 2, 1073741824, 2048, 16777216, 128, 8, 16, 524288, 4194304, 1, 268435456, 1,
                1024, 2, 8192,
            ]
            .into_iter()
            .map(Fp::from)
            .collect(),
            _ => panic!("Unsupported state size"),
        }
    }

    fn generate_circulant_matrix(mds_vector: &[Fp]) -> Vec<Vec<Fp>> {
        let m = mds_vector.len();
        let mut mds_matrix = vec![vec![Fp::zero(); m]; m];
        for i in 0..m {
            for j in 0..m {
                mds_matrix[i][j] = mds_vector[(j + m - i) % m].clone();
            }
        }
        mds_matrix
    }

    fn instantiate_round_constants(
        p: u64,
        m: usize,
        capacity: usize,
        security_level: usize,
        num_rounds: usize,
    ) -> Vec<Fp> {
        let seed_string = format!("RPO({},{},{},{})", p, m, capacity, security_level);
        let mut shake = Shake256::default();
        shake.update(seed_string.as_bytes());

        let num_constants = 2 * m * num_rounds;
        let bytes_per_int = 8;
        let num_bytes = bytes_per_int * num_constants;

        let mut shake_output = shake.finalize_xof();
        let mut test_bytes = vec![0u8; num_bytes];
        shake_output.read_exact(&mut test_bytes).unwrap();

        let mut round_constants = Vec::new();

        for i in 0..num_constants {
            let start = i * bytes_per_int;
            let end = start + bytes_per_int;
            let bytes = &test_bytes[start..end];

            if bytes.len() == 8 {
                let integer = u64::from_le_bytes(bytes.try_into().unwrap());
                let constant = Fp::from(integer);

                if constant.value() >= &p {
                    panic!("Generated constant exceeds field size.");
                }

                round_constants.push(constant);
            } else {
                panic!("Invalid number of bytes extracted for u64 conversion.");
            }
        }
        round_constants
    }

    pub fn apply_sbox(state: &mut [Fp]) {
        for x in state.iter_mut() {
            *x = x.pow(ALPHA);
        }
    }

    fn mds_matrix_vector_multiplication(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mut new_state = vec![Fp::zero(); m];
        for i in 0..m {
            for j in 0..m {
                new_state[i] = new_state[i] + self.mds_matrix[i][j] * state[j];
            }
        }
        new_state
    }

    fn mds_ntt(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let omega = match m {
            12 => Fp::from(281474976645120u64),
            16 => Fp::from(17293822564807737345u64),
            _ => panic!("Unsupported state size for NTT"),
        };

        let mds_ntt = ntt(&self.mds_vector, omega);
        let state_rev: Vec<Fp> = std::iter::once(state[0].clone())
            .chain(state[1..].iter().rev().cloned())
            .collect();
        let state_ntt = ntt(&state_rev, omega);

        let mut product_ntt = vec![Fp::zero(); m];
        for i in 0..m {
            product_ntt[i] = mds_ntt[i] * state_ntt[i];
        }

        let omega_inv = omega.inv().unwrap();
        let result = intt(&product_ntt, omega_inv);

        std::iter::once(result[0].clone())
            .chain(result[1..].iter().rev().cloned())
            .collect()
    }

    fn mds_karatsuba(&self, state: &[Fp]) -> Vec<Fp> {
        let m = state.len();
        let mds_rev: Vec<Fp> = std::iter::once(self.mds_vector[0].clone())
            .chain(self.mds_vector[1..].iter().rev().cloned())
            .collect();

        let conv = karatsuba(&mds_rev, state);

        let mut result = vec![Fp::zero(); m];
        for i in 0..m {
            result[i] = conv[i].clone();
        }
        for i in m..conv.len() {
            result[i - m] = result[i - m] + conv[i].clone();
        }

        result
    }

    fn apply_mds(&self, state: &mut [Fp]) {
        let new_state = match self.mds_method {
            MdsMethod::MatrixMultiplication => self.mds_matrix_vector_multiplication(state),
            MdsMethod::Ntt => self.mds_ntt(state),
            MdsMethod::Karatsuba => self.mds_karatsuba(state),
        };
        state.copy_from_slice(&new_state);
    }

    fn add_round_constants(&self, state: &mut [Fp], round: usize) {
        let m = self.m;
        let round_constants = &self.round_constants;

        for j in 0..m {
            state[j] = state[j] + round_constants[round * 2 * m + j];
        }
    }

    fn add_round_constants_second(&self, state: &mut [Fp], round: usize) {
        let m = self.m;
        let round_constants = &self.round_constants;

        for j in 0..m {
            state[j] = state[j] + round_constants[round * 2 * m + m + j];
        }
    }

    pub fn permutation(&self, state: &mut [Fp]) {
        let num_rounds = NUM_FULL_ROUNDS;

        for round in 0..num_rounds {
            self.apply_mds(state);
            self.add_round_constants(state, round);
            Self::apply_sbox(state);
            self.apply_mds(state);
            self.add_round_constants_second(state, round);
            Self::apply_inverse_sbox(state);
        }
    }

    pub fn hash(&self, input_sequence: &[Fp]) -> Vec<Fp> {
        let m = self.m;
        let capacity = self.capacity;
        let rate = self.rate;

        let mut state = vec![Fp::zero(); m];
        let input_len = input_sequence.len();

        if input_len % rate != 0 {
            state[0] = Fp::one();
        }

        let num_full_chunks = input_len / rate;

        for i in 0..num_full_chunks {
            let chunk = &input_sequence[i * rate..(i + 1) * rate];
            for j in 0..rate {
                state[capacity + j] = chunk[j];
            }
            self.permutation(&mut state);
        }

        let last_chunk_size = input_len % rate;

        if last_chunk_size != 0 {
            let mut last_chunk = vec![Fp::zero(); rate];
            for j in 0..last_chunk_size {
                last_chunk[j] = input_sequence[num_full_chunks * rate + j];
            }
            // Apply padding
            last_chunk[last_chunk_size] = Fp::one();

            for j in 0..rate {
                state[capacity + j] = last_chunk[j];
            }

            self.permutation(&mut state);
        }

        state[capacity..capacity + rate / 2].to_vec()
    }

    pub fn hash_bytes(&self, input: &[u8]) -> Vec<Fp> {
        let field_elements = bytes_to_field_elements(input);
        self.hash(&field_elements)
    }
}

fn bytes_to_field_elements(input: &[u8]) -> Vec<Fp> {
    let mut elements = Vec::new();

    let chunk_size = 7;
    let mut buf = [0u8; 8];

    let mut chunks = input.chunks(chunk_size).peekable();

    while let Some(chunk) = chunks.next() {
        buf.fill(0);
        buf[..chunk.len()].copy_from_slice(chunk);
        if chunk.len() < chunk_size {
            buf[chunk.len()] = 1;
        }
        let value = u64::from_le_bytes(buf);
        elements.push(Fp::from(value));
    }

    elements
}

//  NTT and INTT functions
fn ntt(input: &[Fp], omega: Fp) -> Vec<Fp> {
    let n = input.len();
    let mut output = vec![Fp::zero(); n];
    for i in 0..n {
        let mut sum = Fp::zero();
        for (j, val) in input.iter().enumerate() {
            sum = sum + *val * omega.pow((i * j) as u64);
        }
        output[i] = sum;
    }
    output
}

fn intt(input: &[Fp], omega_inv: Fp) -> Vec<Fp> {
    let n = input.len();
    let inv_n = Fp::from(n as u64).inv().unwrap();
    let mut output = ntt(input, omega_inv);
    for val in output.iter_mut() {
        *val = *val * inv_n;
    }
    output
}

// Karatsuba multiplication
fn karatsuba(lhs: &[Fp], rhs: &[Fp]) -> Vec<Fp> {
    let n = lhs.len();
    if n <= 32 {
        let mut result = vec![Fp::zero(); 2 * n - 1];
        for i in 0..n {
            for j in 0..n {
                result[i + j] = result[i + j] + lhs[i] * rhs[j];
            }
        }
        return result;
    }

    let half = n / 2;

    let lhs_low = &lhs[..half];
    let lhs_high = &lhs[half..];
    let rhs_low = &rhs[..half];
    let rhs_high = &rhs[half..];

    let z0 = karatsuba(lhs_low, rhs_low);
    let z2 = karatsuba(lhs_high, rhs_high);

    let lhs_sum: Vec<Fp> = lhs_low
        .iter()
        .zip(lhs_high.iter())
        .map(|(a, b)| *a + *b)
        .collect();

    let rhs_sum: Vec<Fp> = rhs_low
        .iter()
        .zip(rhs_high.iter())
        .map(|(a, b)| *a + *b)
        .collect();

    let z1 = karatsuba(&lhs_sum, &rhs_sum);

    let mut result = vec![Fp::zero(); 2 * n - 1];

    for i in 0..z0.len() {
        result[i] = result[i] + z0[i];
    }

    for i in 0..z1.len() {
        result[i + half] = result[i + half] + z1[i]
            - z0.get(i).cloned().unwrap_or(Fp::zero())
            - z2.get(i).cloned().unwrap_or(Fp::zero());
    }

    for i in 0..z2.len() {
        result[i + 2 * half] = result[i + 2 * half] + z2[i];
    }

    result
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    lazy_static! {
        pub static ref EXPECTED: Vec<Vec<Fp>> = vec![
            vec![
                Fp::from(1502364727743950833u64),
                Fp::from(5880949717274681448u64),
                Fp::from(162790463902224431u64),
                Fp::from(6901340476773664264u64),
            ],
            vec![
                Fp::from(7478710183745780580u64),
                Fp::from(3308077307559720969u64),
                Fp::from(3383561985796182409u64),
                Fp::from(17205078494700259815u64),
            ],
            vec![
                Fp::from(17439912364295172999u64),
                Fp::from(17979156346142712171u64),
                Fp::from(8280795511427637894u64),
                Fp::from(9349844417834368814u64),
            ],
            vec![
                Fp::from(5105868198472766874u64),
                Fp::from(13090564195691924742u64),
                Fp::from(1058904296915798891u64),
                Fp::from(18379501748825152268u64),
            ],
            vec![
                Fp::from(9133662113608941286u64),
                Fp::from(12096627591905525991u64),
                Fp::from(14963426595993304047u64),
                Fp::from(13290205840019973377u64),
            ],
            vec![
                Fp::from(3134262397541159485u64),
                Fp::from(10106105871979362399u64),
                Fp::from(138768814855329459u64),
                Fp::from(15044809212457404677u64),
            ],
            vec![
                Fp::from(162696376578462826u64),
                Fp::from(4991300494838863586u64),
                Fp::from(660346084748120605u64),
                Fp::from(13179389528641752698u64),
            ],
            vec![
                Fp::from(2242391899857912644u64),
                Fp::from(12689382052053305418u64),
                Fp::from(235236990017815546u64),
                Fp::from(5046143039268215739u64),
            ],
            vec![
                Fp::from(9585630502158073976u64),
                Fp::from(1310051013427303477u64),
                Fp::from(7491921222636097758u64),
                Fp::from(9417501558995216762u64),
            ],
            vec![
                Fp::from(1994394001720334744u64),
                Fp::from(10866209900885216467u64),
                Fp::from(13836092831163031683u64),
                Fp::from(10814636682252756697u64),
            ],
            vec![
                Fp::from(17486854790732826405u64),
                Fp::from(17376549265955727562u64),
                Fp::from(2371059831956435003u64),
                Fp::from(17585704935858006533u64),
            ],
            vec![
                Fp::from(11368277489137713825u64),
                Fp::from(3906270146963049287u64),
                Fp::from(10236262408213059745u64),
                Fp::from(78552867005814007u64),
            ],
            vec![
                Fp::from(17899847381280262181u64),
                Fp::from(14717912805498651446u64),
                Fp::from(10769146203951775298u64),
                Fp::from(2774289833490417856u64),
            ],
            vec![
                Fp::from(3794717687462954368u64),
                Fp::from(4386865643074822822u64),
                Fp::from(8854162840275334305u64),
                Fp::from(7129983987107225269u64),
            ],
            vec![
                Fp::from(7244773535611633983u64),
                Fp::from(19359923075859320u64),
                Fp::from(10898655967774994333u64),
                Fp::from(9319339563065736480u64),
            ],
            vec![
                Fp::from(4935426252518736883u64),
                Fp::from(12584230452580950419u64),
                Fp::from(8762518969632303998u64),
                Fp::from(18159875708229758073u64),
            ],
            vec![
                Fp::from(14871230873837295931u64),
                Fp::from(11225255908868362971u64),
                Fp::from(18100987641405432308u64),
                Fp::from(1559244340089644233u64),
            ],
            vec![
                Fp::from(8348203744950016968u64),
                Fp::from(4041411241960726733u64),
                Fp::from(17584743399305468057u64),
                Fp::from(16836952610803537051u64),
            ],
            vec![
                Fp::from(16139797453633030050u64),
                Fp::from(1090233424040889412u64),
                Fp::from(10770255347785669036u64),
                Fp::from(16982398877290254028u64),
            ],
        ];
    }

    // Utility function to generate random field elements
    fn rand_field_element<R: Rng>(rng: &mut R) -> Fp {
        Fp::from(rng.gen::<u64>())
    }

    // Test for S-box operation
    #[test]
    fn test_apply_sbox() {
        let mut rng = StdRng::seed_from_u64(1);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA));

        RescuePrimeOptimized::<128, 7>::apply_sbox(&mut state);
        assert_eq!(expected, state);
    }

    // Test for inverse S-box operation
    #[test]
    fn test_apply_inverse_sbox() {
        let mut rng = StdRng::seed_from_u64(2);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let mut expected = state.clone();
        expected.iter_mut().for_each(|v| *v = v.pow(ALPHA_INV));

        RescuePrimeOptimized::<128, 7>::apply_inverse_sbox(&mut state);
        assert_eq!(expected, state);
    }

    // Test for MDS matrix multiplication
    #[test]
    fn test_mds_matrix_multiplication() {
        let mut rng = StdRng::seed_from_u64(3);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = rescue.mds_matrix_vector_multiplication(&state);
        let mut computed_state = state.clone();
        rescue.apply_mds(&mut computed_state);

        assert_eq!(expected_state, computed_state);
    }

    // Test for NTT-based MDS matrix multiplication
    #[test]
    fn test_mds_ntt() {
        let mut rng = StdRng::seed_from_u64(4);
        let rescue_ntt = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);
        let state: Vec<Fp> = (0..rescue_ntt.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = rescue_ntt.mds_ntt(&state);
        let mut computed_state = state.clone();
        rescue_ntt.apply_mds(&mut computed_state);

        assert_eq!(expected_state, computed_state);
    }

    // Test for Karatsuba-based MDS matrix multiplication
    #[test]
    fn test_mds_karatsuba() {
        let mut rng = StdRng::seed_from_u64(5);
        let rescue_karatsuba = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Karatsuba);
        let state: Vec<Fp> = (0..rescue_karatsuba.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = rescue_karatsuba.mds_karatsuba(&state);
        let mut computed_state = state.clone();
        rescue_karatsuba.apply_mds(&mut computed_state);

        assert_eq!(expected_state, computed_state);
    }

    // Test for round constant addition in permutation function
    #[test]
    fn test_add_round_constants() {
        let mut rng = StdRng::seed_from_u64(6);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let round = 0;
        let expected_state = state
            .iter()
            .enumerate()
            .map(|(i, &x)| x + rescue.round_constants[round * 2 * rescue.m + i])
            .collect::<Vec<_>>();

        rescue.add_round_constants(&mut state, round);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_permutation() {
        let mut rng = StdRng::seed_from_u64(7);
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let mut state: Vec<Fp> = (0..rescue.m)
            .map(|_| rand_field_element(&mut rng))
            .collect();

        let expected_state = {
            let mut temp_state = state.clone();
            for round in 0..7 {
                rescue.apply_mds(&mut temp_state);
                rescue.add_round_constants(&mut temp_state, round);
                RescuePrimeOptimized::<128, 7>::apply_sbox(&mut temp_state);
                rescue.apply_mds(&mut temp_state);
                rescue.add_round_constants_second(&mut temp_state, round);
                RescuePrimeOptimized::<128, 7>::apply_inverse_sbox(&mut temp_state);
            }
            temp_state
        };

        rescue.permutation(&mut state);

        assert_eq!(expected_state, state);
    }

    #[test]
    fn test_hash_single_chunk() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..8).map(Fp::from).collect();
        let hash_output = rescue.hash(&input_sequence);

        // Verify the squeezed output
        assert_eq!(hash_output.len(), 4); // Half the rate (rate = 8)
    }

    #[test]
    fn test_hash_multiple_chunks() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..16).map(Fp::from).collect(); // Two chunks of size 8
        let hash_output = rescue.hash(&input_sequence);

        // Verify the squeezed output
        assert_eq!(hash_output.len(), 4); // Half the rate (rate = 8)
    }

    #[test]
    fn test_hash_with_padding() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_sequence: Vec<Fp> = (0..5).map(Fp::from).collect(); // Input smaller than rate
        let hash_output = rescue.hash(&input_sequence);

        assert_eq!(hash_output.len(), 4);
    }
    #[test]
    fn hash_padding() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        let input1 = vec![1u8, 2, 3];
        let input2 = vec![1u8, 2, 3, 0];

        let hash1 = rescue.hash_bytes(&input1);
        let hash2 = rescue.hash_bytes(&input2);

        assert_ne!(hash1, hash2);

        let input1 = vec![1_u8, 2, 3, 4, 5, 6];
        let input2 = vec![1_u8, 2, 3, 4, 5, 6, 0];

        let hash1 = rescue.hash_bytes(&input1);
        let hash2 = rescue.hash_bytes(&input2);
        assert_ne!(hash1, hash2);

        let input1 = vec![1_u8, 2, 3, 4, 5, 6, 7, 0, 0];
        let input2 = vec![1_u8, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0];

        let hash1 = rescue.hash_bytes(&input1);
        let hash2 = rescue.hash_bytes(&input2);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn sponge_zeroes_collision() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        let mut zeroes = Vec::new();
        let mut hashes = std::collections::HashSet::new();

        for _ in 0..255 {
            let hash = rescue.hash(&zeroes);
            assert!(hashes.insert(hash));
            zeroes.push(Fp::zero());
        }
    }
    // Test for hash function with byte input
    #[test]
    fn test_hash_bytes() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let input_bytes = b"Rescue Prime Optimized";
        let hash_output = rescue.hash_bytes(input_bytes);

        // Verify the squeezed output
        assert_eq!(hash_output.len(), 4); // Half the rate (rate = 8)
    }

    // Test for round constants instantiation
    #[test]
    fn test_instantiate_round_constants() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        let round_constants = &rescue.round_constants;
        assert_eq!(round_constants.len(), 2 * 12 * 7); // 2 * m * NUM_FULL_ROUNDS
    }

    // Test for MDS methods consistency
    #[test]
    fn test_mds_methods_consistency() {
        let rescue_matrix = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let rescue_ntt = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);
        let rescue_karatsuba = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Karatsuba);

        let input = vec![
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
            Fp::from(9u64),
        ];

        let hash_matrix = rescue_matrix.hash(&input);
        let hash_ntt = rescue_ntt.hash(&input);
        let hash_karatsuba = rescue_karatsuba.hash(&input);

        assert_eq!(hash_matrix, hash_ntt);
        assert_eq!(hash_ntt, hash_karatsuba);
    }

    #[test]
    fn generate_test_vectors() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let elements = vec![
            Fp::from(0u64),
            Fp::from(1u64),
            Fp::from(2u64),
            Fp::from(3u64),
            Fp::from(4u64),
            Fp::from(5u64),
            Fp::from(6u64),
            Fp::from(7u64),
            Fp::from(8u64),
            Fp::from(9u64),
            Fp::from(10u64),
            Fp::from(11u64),
            Fp::from(12u64),
            Fp::from(13u64),
            Fp::from(14u64),
            Fp::from(15u64),
            Fp::from(16u64),
            Fp::from(17u64),
            Fp::from(18u64),
        ];

        println!("let expected_hashes = vec![");
        for i in 0..elements.len() {
            let input = &elements[..=i];
            let hash_output = rescue.hash(input);

            print!("    vec![");
            for value in &hash_output {
                print!("Fp::from({}u64), ", value.value());
            }
            println!("],");
        }
        println!("];");
    }

    #[test]
    fn test_print_round_constants() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);

        println!("Round constants:");
        for (i, constant) in rescue.round_constants.iter().enumerate() {
            println!("Constant {}: Fp::from({}u64)", i, constant.value());
        }

        assert_eq!(rescue.round_constants.len(), 2 * rescue.m * 7); // 2 * m * NUM_FULL_ROUNDS
    }

    #[test]
    fn test_hash_vectors() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::MatrixMultiplication);
        let elements: Vec<Fp> = (0..19).map(Fp::from).collect();

        for (i, expected) in EXPECTED.iter().enumerate() {
            let input = &elements[..=i]; // Tomar el prefijo hasta i
            let hash_output = rescue.hash(input);

            assert_eq!(
                hash_output,
                *expected,
                "Hash mismatch for input length {}",
                i + 1
            );
        }
    }

    #[test]
    fn test_hash_example_and_print() {
        let rescue = RescuePrimeOptimized::<128, 7>::new(MdsMethod::Ntt);

        let input = b"Boquita campeon del mundo!";

        let hash_result = rescue.hash_bytes(input);

        println!("Input: {:?}", input);
        println!("Hash result:");
        for (i, value) in hash_result.iter().enumerate() {
            println!("  {}: {}", i, value);
        }

        println!("Hash as u64 values:");
        for value in hash_result.iter() {
            print!("{}, ", value.value());
        }
        println!();
        assert_eq!(hash_result.len(), 4);
    }
    // this gives the same in Polygon Miden
}
