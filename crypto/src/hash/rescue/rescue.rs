use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;


pub struct Rescue {
    pub width: usize,
    pub rate: usize,
    pub capacity: usize,
    pub n: usize  
}

impl Rescue {
    pub fn new(width: usize, rate: usize, capacity: usize, n: usize) -> Self {
        Self {width, rate, capacity, n}
    }

    pub fn hash(&self, x: &FieldElement<Stark252PrimeField>, y: &FieldElement<Stark252PrimeField>) -> FieldElement<Stark252PrimeField> {
        let mut state = vec![*x, *y];
        let _ = self.permutation(&mut state); 
        let output = &state[..self.rate];
        output[0].clone()
    }


    pub fn permutation(&self, state: &mut Vec<FieldElement<Stark252PrimeField>>)  {
    
        let alpha: u64 = 3;
        let alpha_inv:u128 = 180331931428153586757283157844700080811;

        let m = self.width;
        let n = self.n;
        let r = self.rate;

        // round constants
        let round_constants: Vec<FieldElement<Stark252PrimeField>> = vec![
            FieldElement::<Stark252PrimeField>::from_hex("0x22F1430A79B8B5C17A0A1FD6170D9AB40").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x18D2F47E58F413DDC2A9A92A91E6D4EF").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x34F5F3F4D7BEB95B8769CAB9E38053D").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x3BEA7FB0666A37D1530F8F07525B35AD").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x39C8E41F24785DE8485D8E8E38F5F50").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x2357A61E27AD075BBC20A3A618CEC51").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x30CEA0A45DDCF59A17E5B79279EF97F").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xE189B1486C68B1ACBCEB28A4F4E59E4").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x17B8A41A8CA39BBD6D9F21D1A231C3A5").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x1E9BB1E0A40289CF5D").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xBD8E59FAA656544A4889BC4486D5127").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x952A1C8008D693DF8B49B8BB196CD8E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x59AA3A48F985A5D95B01DDCE78527F").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x993F346593FB35EC0D5498C6AA9B6E4").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x3A172ED80755A764581B19BB2ECD3E8").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x2AF7646BCF20A19848B07DD56A940D5").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x181F5021A4F1844810D8D2341E475E3").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xA0FEF2B5B19D780DD1E4CB1A").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x8A828F39587B8165AD56DF4CA833016").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x5A1A4C93FA860FE3E4AC8E2A61B9D8C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x8A80D760BD2C0F8DB8AE40940BFC9E4").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x99269F0BB6A0B46FA6AE3CB8D6A25CA").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xE34D00F943D53C4A").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x68F3E2A6FE35E62FA4D60547B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x62D4F2D0B5C3613925F27A4373").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x53A506ABD").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x7D8B587AA2197D").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x3E9B536943EA5C5E0D0C4939B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x1BD4384F6A3EB2DD889D097EFB3A15B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x16B3F53C2F54B5DFF603BC96E09328E4").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x13CE27CCCA6A884F8C54E0F1AA7C57E8").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xA870B56F5B49A75F3F8029E3").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x6571F190700F18ADFB24E6103C11CE8").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xA862E04758F7D2FA2E1CF69A14AD0BC").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x1AF0957A8E9A7E726B2A96").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x1880B30B218C7D16A1E99D6663611903").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x3B6CC9CC76226252E8EE37C83").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x26E35A6F9F214AADBF3702325B049").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x3CB8F83820580D264AEC8AC0D56AD0").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0xE6D5FFC951D157F812951B191DFE92").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x1A4CC8226F17B80B5F4F2E349A9D45C4616").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x1119B73C3661B0E4B8CDE276D9CA870E1FB").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0FD50E0EB32B55D6A0401C6B76D215ACBE6").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0BACDE7F1481C11AA50D2A34AD06C10C11A").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0CAA733F41E8BC84C7E1579F6A8DE8FF6CB").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0DA9657A0A292C8B9596E10384086011B7E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0AEBF3781E2F45ABD75D81E3B08B2E5A6B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0400B54D6E192F60CE69E7C9B2620593BC").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x04A39C47B30F00CE59E2F61D399A1505AC").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03A90E0B43A6EBFA6A4821E06D19E8B76A").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0288A219C0FD39A3B1149962E84A2E0776").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x05AC5D036CC126F15AC77A63BD63A30C6B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x04A90F53B3DCD632EC60B6044C2E15841B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0109FFABD9E9B6C3BE5016B00000292036").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0191059404A7C69F3E72EFA1449D7A4C29").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0259A5749A1583724D055B8E307CAACEE4").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03367CDD3A3BB6EDB04F3AEE9CEAEEB976").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0689B005C606A475D8124D35E67A235C5E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0D0701E0A8EAC4A457106A9BB4F816C1AE").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x02998CDE7CC0752F3A24065F6B5E54576D").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x010E31AAFDDEA4C6B8CD0F496F5B4714E6").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0DA7951F8B8663A87F9821B26053A2E68CB").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0ABEB53C47DDA17F6C91E7D1F0B935F76BF").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0D5F098B1A6D77D21C259D4EA49A1A5460").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0FB358EF2FF40E33CB0F9D99C11238398A").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x026541B06F8DEE1C78D4CC7B79A9C4F3B7").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x00252D21492E1573A3B1D636A97D20F6CD").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0C7E0058955BBE6A170739B0DCE6704155").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x056A2010039F4825A0E5034DFE0F4EE6D1").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x028E2EFE200622C6471A0A244477A4F75C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0096AB4C631001A0C5605A58EBB0160B78").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E7830C4BD3B4F4E200200F0E640E0E080").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0413E7A5004F1E9B8000000000000000004").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03BBEE49C60E399B20000040E00C0C0000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x02A5B660200C01A0A0020057C0A0C4C010").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0D98D48E4BC4A96C9C3A0288242A8A2A82").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0A45F322B1D33AC6A80CDEE3D25C2D9DB6").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0F238EE7D1E7A8D24A4EBA0450A0E73B35").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0B2E305FEB883B7C01CB25F78BAF9DEE7C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0219B553D7A8A64E0475BB032932F3B24F").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0DE680FD17084E1C73C393B3F172FAFA36").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x085400000E000C8C0E0000010000E0000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x013A17B3D51F3B102801880E1E80E0E80E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x019BA8AA1AA1E80E802E80E8080E808080").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x022E12B52A8101020E8080800002002008").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E43D8B0D303D51C0E43D51EDC0E43C0E0").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0AA151161E808080E4E1E808080808E4E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0C0CE0604000000000000000000000000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E6C7C422D008000000000000000000000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0C7B2E001000D4E8000008008008008000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E1AA040C0000000200E0E0E0E0E0E0E0E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0A7D8E2002040000200800020020020020").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0A6E0A802E0E2E2E2E2E2E2E2E2E2E2E2").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E58C09C0E60E80E20E02E02E02E0A02E02").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0EA54AC7A5C4E02C05A2500250A250A02E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E65650260A055C40C0E0204A4C4C0A4C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0E2269E87C80E24A86002E2E2080208A2E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0A2D090A38080008A08082080000002020").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0AEC2E00E200E80000E800000000000000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x020A8F7DA4DABEDD0FF6F81C1CE777704A5").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x000302EC2FEC2B9E70B362271E5717E3319").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x02014DD5A5B77C4B0F6E9A8ED39E7C2C7F4").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03090E21D6F260010EB5E0115E016C5E15E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x049A3F6504D8B88B6B0ECA5E2EC6ACADAE6").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x00DFD38E6CA1C0CCBF8ECF2B7771D6712B3").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x036E72195A1F008F0E80880E1E809E818E0").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x04015B85A77F4603A4CBE2B2BCC3A9C21E0").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x01B09E71DDBB5127CE3E73035B1E08B32E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x036C8EDEE18E0E01880A71A1A1A8EA8E0E").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03268E3AAE016D6E1B47BB7122AEEE282D").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x036D4386DE4B0BEC0CBA41E0C42E0B0BE2").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x01477A013E26B4467BA66BAA1E667671C6").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0366009B117740E6100465B1BA18E09B3").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x00730C1CE8830E2B48A869010F036C32D0").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x00E0A4A4160C95C8BB41BC1C15D1D0E54B").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x011D67D0E887C5C816EBEE36080B37B37C3").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x02AFAE3C88C28C80BC0C0A3A3A3C03A3A03").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x0D814B86B0AC02D8DD141400D00DADA08B2").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x01A9C812BD30D2180A1B288390A10A132F").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03C548460E6E5D0A440740AC1A7A22A822").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x036D0A8A2C1A8F1C1CA2E00A2E00E0E2C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x036C9B9A9D0090E10E96E0E10E0E0E0E0").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x02520E12A1C920E0E0E00800E01E00008").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03B2C7C7A4C824E8E3E2B4F7F4FAB2C72C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03A6E0D2928B0CA32A480A1F540A101FE8").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03F920FB9AB09A2922A1F0E80E2B04E80F").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03A2E09BE90302C4E6A0682C20C0E0C20C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x03432E2C80E0A02C6A30E460C0E6A4C68C").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x001E000E68000E000000E00000E0E0E000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x037E4E6C0A008E000A0000C0000000000").unwrap(),
            FieldElement::<Stark252PrimeField>::from_hex("0x01A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1A1").unwrap(),

        ];

        // mds matrix

        let mds: Vec<Vec<FieldElement<Stark252PrimeField>>> = vec![
            vec![
                FieldElement::<Stark252PrimeField>::from_hex("0x0F2B68A89C131E280DD33EEBCB10F49E").unwrap(),
                FieldElement::<Stark252PrimeField>::from(4),
            ],
            vec![
                FieldElement::<Stark252PrimeField>::from_hex("0x0F2B68A89C131E280DD33EEBCB10F49E").unwrap(),
                FieldElement::<Stark252PrimeField>::from(13),
            ],
        ];

        // zero

        let zero = FieldElement::<Stark252PrimeField>::from(0);

        for _ in 0..n {
          for i in 0..m {
            state[i] = state[i].pow(alpha);  
          }
          
        let mut temp = vec![zero; m];
            for i in 0..m {
                for j in 0..m {          
                temp[i] = temp[i] + mds[i][j] * state[j]; 
            }
        }
          
        for i in 0..m {
            state[i] = temp[i] + round_constants[2*r*m + i];  
        }
           
        for i in 0..m {
            state[i] = state[i].pow(alpha_inv);
        }

        let mut temp = vec![zero; m];
            for i in 0..m {
                for j in 0..m {          
                temp[i] = temp[i] + mds[i][j] * state[j]; 
            }
        }

        for i in 0..m {
            state[i] = temp[i] + round_constants[2*r*m + i]; 
        }

        }

      }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_with_dec_inputs() {
        let rescue = Rescue::new(2, 1, 1, 27);
        let x = FieldElement::<Stark252PrimeField>::from(1);
        let y = FieldElement::<Stark252PrimeField>::from(1);
        let expected = FieldElement::<Stark252PrimeField>::from_hex("0x4c86df57efef446181b4d8100b2fdcf31dff0d9bcdf2f5f6a13a23074de804f").unwrap();
        let result = rescue.hash(&x, &y);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_hash_with_hex_inputs() {
        let rescue = Rescue::new(2, 1, 1, 27);
        let x = FieldElement::<Stark252PrimeField>::from_hex("0x1234328495738291039").unwrap();
        let y = FieldElement::<Stark252PrimeField>::from_hex("0x456").unwrap();
        let expected = FieldElement::<Stark252PrimeField>::from_hex("0x589ac0f0fa3e90bd535733bac64227065947f5703b94be031a6a1c4f723dcdd").unwrap();
        let result = rescue.hash(&x, &y);
        assert_eq!(result, expected);
    }
}