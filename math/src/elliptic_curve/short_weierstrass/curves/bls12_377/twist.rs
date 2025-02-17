use super::field_extension::Degree2ExtensionField;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

const GENERATOR_X_0: U384 = U384::from_hex_unchecked("0x018480be71c785fec89630a2a3841d01c565f071203e50317ea501f557db6b9b71889f52bb53540274e3e48f7c005196");
const GENERATOR_X_1: U384 = U384::from_hex_unchecked("0x00ea6040e700403170dc5a51b1b140d5532777ee6651cecbe7223ece0799c9de5cf89984bff76fe6b26bfefa6ea16afe");
const GENERATOR_Y_0: U384 = U384::from_hex_unchecked("0x00690d665d446f7bd960736bcbb2efb4de03ed7274b49a58e458c282f832d204f2cf88886d8c7c2ef094094409fd4ddf");
const GENERATOR_Y_1: U384 = U384::from_hex_unchecked("0x00f8169fd28355189e549da3151a70aa61ef11ac3d591bf12463b01acee304c24279b83f5e52270bd9a1cdd185eb8f93");

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12377TwistCurve;

impl IsEllipticCurve for BLS12377TwistCurve {
    type BaseField = Degree2ExtensionField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point is mathematically verified to be a valid point on the curve.
        // - `unwrap_unchecked()` is safe because the provided coordinates satisfy the curve equation.
        unsafe {
            Self::PointRepresentation::new([
                FieldElement::new([
                    FieldElement::new(GENERATOR_X_0),
                    FieldElement::new(GENERATOR_X_1),
                ]),
                FieldElement::new([
                    FieldElement::new(GENERATOR_Y_0),
                    FieldElement::new(GENERATOR_Y_1),
                ]),
                FieldElement::one(),
            ])
            .unwrap_unchecked()
        }
    }
}

impl IsShortWeierstrass for BLS12377TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([FieldElement::zero(),
        FieldElement::from_hex_unchecked("0x10222f6db0fd6f343bd03737460c589dc7b4f91cd5fd889129207b63c6bf8000dd39e5c1ccccccd1c9ed9999999999a")])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_377::field_extension::{BLS12377PrimeField, Degree2ExtensionField},
                traits::IsShortWeierstrass,
            },
            traits::IsEllipticCurve,
        },
        field::element::FieldElement,
        unsigned_integer::element::U384,
    };

    use super::BLS12377TwistCurve;
    type Level0FE = FieldElement<BLS12377PrimeField>;
    type Level1FE = FieldElement<Degree2ExtensionField>;

    #[test]
    fn create_generator() {
        let g = BLS12377TwistCurve::generator();
        let [x, y, _] = g.coordinates();
        assert_eq!(
            BLS12377TwistCurve::defining_equation(x, y),
            Level1FE::zero()
        );
    }
    #[test]
    // Values checked in sage
    fn add_point_three_times_equals_operate_with_self_3() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x11a87eda97b96e733c2eb833ae35531b87878b416d57b370c7cd13b5f3c413387633b0ca6dfead19305318501376087")),
            Level0FE::new(U384::from_hex_unchecked("0xa4a6d842722f2636937acf0e889ab343e121a599b8a3a9bd3be766da7d84f8e060be00f06bb2d29df963bf2d847598"))
        ]);
        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x75589c0925d6cf45e715460ea7cb3388d4e21d9b79aa8411567d8de85ba5561bcab80a5c0b363e31817d458e5b2a2a")),
            Level0FE::new(U384::from_hex_unchecked("0xcb4e1e4b160cc6c92e7b3dd0b2f4053bc7178d201e7788796a6035c59ccd586635796e97003b1f76eca273576f01ac"))
        ]);

        let expectedx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x1aba32e70b88834d0cd5fb3a27eda8211eb94b6a191cd287f798145c09dc64dabda377836c31d31cc728635425ccc61")),
            Level0FE::new(U384::from_hex_unchecked("0x146b578a9c3d92f64baafa082d27c3446fb197659bbd4ac45e290f5f49ae308599448dc288140a4049bac3c6e3e1aac"))
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x1417486a114a5a9074151a3a2c710dc105cce81de69a91067758355cda7049a2ad8077a2343679bba473484ad0cddd6")),
            Level0FE::new(U384::from_hex_unchecked("0x195c939995782a07b26e3b44b49d58eb0951d452d0e8928f218a8e63f74f74860cb56265e437da80df67b6254b27cd"))
        ]);
        let p = BLS12377TwistCurve::create_point_from_affine(px, py).unwrap();
        let expected = BLS12377TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();
        assert_eq!(p.operate_with_self(3_u16), expected);
    }

    #[test]
    // Values checked in sage
    fn operate_with_self_test() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x2233db786c8cb6a3b6846aebad4ce3f5346961c8bade4c129d920170d1ceeb02d84fd4e12b592f0cba64e083d75167")),
            Level0FE::new(U384::from_hex_unchecked("0x12d1c7ac03cb1991993cdb9d38c50588b67c18ed9b9db5f84ac5b59c201f6493a42f690169b96b7a9f12fae4718b6cb"))
        ]);

        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x160cc59bb3929b1073996aa370c880e002b81f3e4f7275636caf55754bbfcfe1d43c5d91ee7f3cb49254be2366d5d0")),
            Level0FE::new(U384::from_hex_unchecked("0xe66460970bf2fc79831983744d7c83fad910266fd56f08b4a8f737e7609d88930503091e06228a184627b57c02352"))
        ]);

        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x1124128cf7166bb67207327079f8a3e75d876e3b6dd54cc05c766951c5d832aa202c57ed2308d78283e4c859be8fee3")),
            Level0FE::new(U384::from_hex_unchecked("0x1889e19d4f67d120d367c15f7bc23529fe4e335627e0eb16ec2bafe6199f0e7d393c5413fc7157ec03d5d56e1eb333"))
        ]);

        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x16db05c371bf6f28e92faa77d3abf5c66c4feea84d334807f46ee69227a2144a827b00f8bcf4179b97922a85ebd5776")),
            Level0FE::new(U384::from_hex_unchecked("0x105def68752ac5cb73f3b692b3c877f2febe6cd8a679584f4b1c64f9886f12b15dd7909251bc1e90d559fadd6b1f8f5"))
        ]);

        let scalar = U384::from_hex_unchecked(
            "0x3061aa3679b1865fa09dac7339a87511147f015aa8997fae59b475d751bc16f",
        );

        let p = BLS12377TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12377TwistCurve::create_point_from_affine(qx, qy).unwrap();

        assert_eq!(p.operate_with_self(scalar), q);
    }

    #[test]
    // Values checked in sage
    fn add_two_points() {
        let px = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x11a87eda97b96e733c2eb833ae35531b87878b416d57b370c7cd13b5f3c413387633b0ca6dfead19305318501376087")),
            Level0FE::new(U384::from_hex_unchecked("0xa4a6d842722f2636937acf0e889ab343e121a599b8a3a9bd3be766da7d84f8e060be00f06bb2d29df963bf2d847598"))
        ]);

        let py = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x75589c0925d6cf45e715460ea7cb3388d4e21d9b79aa8411567d8de85ba5561bcab80a5c0b363e31817d458e5b2a2a")),
            Level0FE::new(U384::from_hex_unchecked("0xcb4e1e4b160cc6c92e7b3dd0b2f4053bc7178d201e7788796a6035c59ccd586635796e97003b1f76eca273576f01ac"))
        ]);

        let qx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0xa27279ea48d8e812223956c84ae89c5de09e67c7052c056d936da4578dcf0b30799f2212af0017fe50e146c5c2ce05")),
            Level0FE::new(U384::from_hex_unchecked("0xaed1579f3de386644e0ffac471e98f8f1d97a6be4d88a0c516bc9fd4d7780649424c3d51bb85d074d66560e2c2bb07"))
        ]);

        let qy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x6e7e249e0d7c81d0fec62a3f96d6fefd7d6c87019559b03664a015a2ed536d98e6432b93ec219426f9729f119c7982")),
            Level0FE::new(U384::from_hex_unchecked("0x104c4e4adb48273511a0473d742724f48042b967591ab9b86731bb902debfd44d89571af704340614f6618863fc5841"))
        ]);

        let expectedx = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x19d74f8769a27aae05c5b50b2d051849f6feab554d0fd4f51c7fe43d918fac1d966c97dd6e61b2f7fc13694495dc259")),
            Level0FE::new(U384::from_hex_unchecked("0x11f4194886ad75e3baff9853734e9ef4fe5f20d333951ba126bca7dd54ebc4998b17d4d315d29147dc22124c5464a64"))
        ]);
        let expectedy = Level1FE::new([
            Level0FE::new(U384::from_hex_unchecked("0x14faa941c0cdf5567c294bc9e73e83c4602f331a7400222c8475fba4c6a688b12ce8f7cc20e54eaac1af6046aec58bd")),
            Level0FE::new(U384::from_hex_unchecked("0x7cee0c4760e2bf76047cda31141e6242c5893ba5c2fba5f23c0048545912e309dc647f4cfe2db17b23aa2f57c3d8c3"))
        ]);

        let p = BLS12377TwistCurve::create_point_from_affine(px, py).unwrap();
        let q = BLS12377TwistCurve::create_point_from_affine(qx, qy).unwrap();
        let expected = BLS12377TwistCurve::create_point_from_affine(expectedx, expectedy).unwrap();

        assert_eq!(p.operate_with(&q), expected);
    }
}
