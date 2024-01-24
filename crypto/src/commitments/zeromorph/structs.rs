use core::mem;

use lambdaworks_math::{elliptic_curve::traits::IsPairing, errors::DeserializationError, traits::{AsBytes, Deserializable}};

use super::zeromorph::ZeromorphError;

//TODO: gate with alloc
#[derive(Debug, Clone, Default)]
pub struct ZeromorphSRS<P: IsPairing> {
    pub g1_powers: Vec<P::G1Point>,
    /// [g2: (P::G2Point), tau_2: (P::G2Point), tau_n_max_sub_2_n: (P::G2Point)]
    pub g2_powers: [P::G2Point; 3],
}

impl<P: IsPairing> ZeromorphSRS<P>
{
    pub fn new(g1_powers: &[P::G1Point], g2_powers: &[P::G2Point; 3]) -> Self {
        Self {
            g1_powers: g1_powers.into(),
            g2_powers: g2_powers.clone(),
        }
    }

    //TODO: Delete?
    /*
    pub fn trim(
        &self,
        max_degree: usize,
    ) -> Result<(ZeromorphProverKey<P>, ZeromorphVerifierKey<P>), ZeromorphError> {
        if max_degree > self.g1_powers.len() {
            return Err(ZeromorphError::KeyLengthError(
                max_degree,
                self.g1_powers.len(),
            ));
        }
        let offset = self.g1_powers.len() - max_degree;
        let offset_g1_powers = self.g1_powers[offset..].to_vec();
        Ok((
            ZeromorphProverKey {
                g1_powers: self.g1_powers.clone(),
                offset_g1_powers: offset_g1_powers,
            },
            ZeromorphVerifierKey {
                g1: self.g1_powers[0],
                g2: self.g2_powers[0],
                tau_2: self.g2_powers[1],
                tau_n_max_sub_2_n: self.g2_powers[offset],
            },
        ))
    }
    */
}

#[cfg(feature = "std")]
impl<P: IsPairing> ZeromorphSRS<P>
where
    <P as IsPairing>::G1Point: Deserializable,
    <P as IsPairing>::G2Point: Deserializable,
{
    pub fn from_file(file_path: &str) -> Result<Self, crate::errors::SrsFromFileError> {
        let bytes = std::fs::read(file_path)?;
        Ok(Self::deserialize(&bytes)?)
    }
}

impl<P: IsPairing> AsBytes for ZeromorphSRS<P>
where
    <P as IsPairing>::G1Point: AsBytes,
    <P as IsPairing>::G2Point: AsBytes,
{
    fn as_bytes(&self) -> Vec<u8> {
        let mut serialized_data: Vec<u8> = Vec::new();
        // First 4 bytes encodes protocol version
        let protocol_version: [u8; 4] = [0; 4];

        serialized_data.extend(&protocol_version);

        // Second 8 bytes store the amount of G1 elements to be stored, this is more than can be indexed with a 64-bit architecture, and some millions of terabytes of data if the points were compressed
        let mut main_group_len_bytes: Vec<u8> = self.g1_powers.len().to_le_bytes().to_vec();

        // For architectures with less than 64 bits for pointers
        // We add extra zeros at the end
        while main_group_len_bytes.len() < 8 {
            main_group_len_bytes.push(0)
        }

        serialized_data.extend(&main_group_len_bytes);

        // G1 elements
        for point in &self.g1_powers {
            serialized_data.extend(point.as_bytes());
        }

        // G2 elements
        for point in &self.g2_powers {
            serialized_data.extend(point.as_bytes());
        }

        serialized_data
    }
}

impl<P: IsPairing> Deserializable for ZeromorphSRS<P>
where
    <P as IsPairing>::G1Point: Deserializable,
    <P as IsPairing>::G2Point: Deserializable,
{
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError> {
        const MAIN_GROUP_LEN_OFFSET: usize = 4;
        const MAIN_GROUP_OFFSET: usize = 12;

        let main_group_len_u64 = u64::from_le_bytes(
            // This unwrap can't fail since we are fixing the size of the slice
            bytes[MAIN_GROUP_LEN_OFFSET..MAIN_GROUP_OFFSET]
                .try_into()
                .unwrap(),
        );

        let main_group_len = usize::try_from(main_group_len_u64)
            .map_err(|_| DeserializationError::PointerSizeError)?;

        let mut main_group: Vec<P::G1Point> = Vec::new();
        let mut secondary_group: Vec<P::G2Point> = Vec::new();

        let size_g1_point = mem::size_of::<P::G1Point>();
        let size_g2_point = mem::size_of::<P::G2Point>();

        for i in 0..main_group_len {
            // The second unwrap shouldn't fail since the amount of bytes is fixed
            let point = P::G1Point::deserialize(
                bytes[i * size_g1_point + MAIN_GROUP_OFFSET
                    ..i * size_g1_point + size_g1_point + MAIN_GROUP_OFFSET]
                    .try_into()
                    .unwrap(),
            )?;
            main_group.push(point);
        }

        let g2s_offset = size_g1_point * main_group_len + MAIN_GROUP_OFFSET;
        for i in 0..3 {
            // The second unwrap shouldn't fail since the amount of bytes is fixed
            let point = P::G2Point::deserialize(
                bytes[i * size_g2_point + g2s_offset
                    ..i * size_g2_point + g2s_offset + size_g2_point]
                    .try_into()
                    .unwrap(),
            )?;
            secondary_group.push(point);
        }

        let secondary_group_slice = [secondary_group[0].clone(), secondary_group[1].clone(), secondary_group[2].clone()];

        let srs = ZeromorphSRS::new(&main_group, &secondary_group_slice);
        Ok(srs)
    }
}


#[derive(Clone, Debug)]
pub struct ZeromorphProverKey<P: IsPairing> {
    pub g1_powers: Vec<P::G1Point>,
    pub offset_g1_powers: Vec<P::G1Point>,
}

/*
#[derive(Copy, Clone, Debug)]
pub struct ZeromorphVerifierKey<P: IsPairing> {
    pub g1: P::G1Point,
    pub g2: P::G2Point,
    pub tau_2: P::G2Point,
    pub tau_n_max_sub_2_n: P::G2Point,
}

#[derive(Clone, Debug)]
pub struct ZeromorphProof<P: IsPairing> {
    pub pi: P::G1Point,
    pub q_hat_com: P::G1Point,
    pub q_k_com: Vec<P::G1Point>,
}
*/