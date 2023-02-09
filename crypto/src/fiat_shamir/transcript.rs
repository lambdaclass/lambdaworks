

pub trait IsHashFunction {
    fn hash(&self, bytes: &Vec<u8>) -> [u8; 1];
}

struct Transcript<T: IsHashFunction> {
    // Función de hash
    // Tiene que saber "bind" / "append"
    // Tiene que saber "compute" / correr hash
    hash_function: T,
    data: Vec<u8>
}

impl<T: IsHashFunction> Transcript<T> {
    fn new(hash_function: T) -> Self {
        Self {
            hash_function: hash_function,
            data: Vec::new()
        }
    }

    fn append(&mut self, new_data: &Vec<u8>) {
        self.data.append(&mut new_data.clone());
    }

    fn challenge(&mut self) {
        self.data = self.hash_function.hash(&self.data).to_vec();
    }
}


#[cfg(test)]
mod tests {
    use crate::fiat_shamir::transcript;

    use super::*;

    struct DummyHash2;
    impl IsHashFunction for DummyHash2 {
        fn hash(&self, bytes: &Vec<u8>) -> [u8; 1] {
            [bytes.iter().fold(0, |acc, &x| (acc ^ x))]
        }
    }

    #[test]
    fn basic_challenge() {
        let mut transcript = Transcript::new(DummyHash2);
        
        let point_a: Vec<u8> = vec![0xFF, 0xAB];
        let point_b: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

        transcript.append(&point_a); // point_a
        transcript.append(&point_b); // point_a + point_b

        transcript.challenge(); // Hash(point_a  + point_b)

        assert_eq!(transcript.data, [0x98]);

        let point_c: Vec<u8> = vec![0xFF, 0xAB]; 
        let point_d: Vec<u8> = vec![0xDD, 0x8C, 0x9D]; 

        transcript.append(&point_c); // Hash(point_a  + point_b) + point_c
        transcript.append(&point_d); // Hash(point_a  + point_b) + point_c + point_d

        transcript.challenge(); // Hash(Hash(point_a  + point_b) + point_c + point_d)
        assert_eq!(transcript.data, [0x0]);
    }
}
