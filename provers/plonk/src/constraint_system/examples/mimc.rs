use lambdaworks_math::field::{element::FieldElement as FE, traits::IsField};

use crate::constraint_system::{ConstraintSystem, Variable};

/// The MIMC hash function.
pub fn mimc<F: IsField>(
    system: &mut ConstraintSystem<F>,
    coefficients: &[FE<F>],
    data: &[Variable],
) -> Variable {
    let mut h = system.new_constant(FE::zero());

    for item in data.iter() {
        let mut x = *item;
        for c in coefficients.iter() {
            // x = (x + h + c) ** 5
            x = system.linear_combination(&x, FE::one(), &h, FE::one(), c.clone(), None);
            let x_pow_2 = system.mul(&x, &x);
            let x_pow_4 = system.mul(&x_pow_2, &x_pow_2);
            x = system.mul(&x_pow_4, &x);
        }
        // h = x + 2h + item
        h = system.linear_combination(&x, FE::one(), &h, FE::from(2), FE::zero(), None);
        h = system.add(&h, item);
    }
    h
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashMap;

    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrField,
        field::element::FieldElement as FE,
    };

    use crate::constraint_system::{examples::mimc::mimc, ConstraintSystem};

    #[test]
    fn test_mimc() {
        let coefficients = vec![
            "1dbfc7763d69ca7d15701422f37bc6692bd01ebc4da42360f81f9adb4a91b01a",
            "4fd2cddd334dab1c4005161c290f25a0e18d4175ecfa898b17095d8ec2dd344a",
            "9cc76e9b37ba649b0accb508950d834af091f3d687c208d9013e1685075f092",
            "16472c2e925fbba0fad047c428a4e8e4801414975e9841d5518b57fbcf26dde1",
            "1c2e148c40ea201b748bee72845b349bfa4a4497837af0d569ae47afc6e4243",
            "705ea7625cbcb5daf4d3dc5d0322e7b3adbe32227dc52234035881407825dbaf",
            "1272efa088fdd0c941712554929ee2bf4e298fce57337dda8f4d704a8bdec1ea",
            "4f966f7b066b2d838afab7b99705b1fbecff809f083be8a03ea1a656be14d72a",
            "283392b9145c98fc9680ee035816761cb79155557f0b302511a928c221b04c03",
            "430a47a5110d6ee4da087ee3291a217f7afba21d696eb74de6ce41cf50aeeff4",
            "1c9fbf2d5b15f5b4b9aaa8dfb452a310b6fa3de7b2b7c68260f8e4aff63840ef",
            "49c756d15bbf811f532811dba19f5fda9df678bcdd4017024ef4daded412af7d",
            "3d6d63a3302df941979292e4be9a85f9a960698ce9a2e5d430423f4adf7a9bb8",
            "5f6c2da1c738096eaac7763afc219965955b33e619ce5679c3f5d3aef1792b0a",
            "32d630538e47bf4f8968170577a08cb1b26864879c86dafb652cce5068bdb5aa",
            "2eb8b2a5593fdef777738374339441e112704f378f7cca12d4146d30a005b96",
            "123313cced613293c40586b110f8e4244cd67cc4380c8f5df4ec60f42216ce28",
            "42d1c99dccb35f9afa170ee24eb146903819160985f2460d7785ac4381ca037c",
            "35375cf9debbeba36a0ed9286c67a18bd2112dc028387b905b36c23dce8c4926",
            "67e693adf50e0e16fa03d5f9481d71ed0f63ed4527e080941d1ba0473c18bcc3",
            "1d5f6a82f699df8c7fff5b5f90047128ead7923635c92a4849ff28689b6c7258",
            "372a3d44e73aae9443ca680956bcd23dbdd5f790e0c5cfa45a0fcfb9ee920144",
            "630b2c9009da6417963e8d45ae92e59322746e545e04f026004a2f76c12422f5",
            "33269ebd4d0f0a2874a217899b11a13361d262c1be48f2a46e6b132f897a5ac4",
            "394d93f60615db568325c284dd916d735072cb57b6cd2a0072d976154d8a3eea",
            "14e83ce42e31effc8be6e0119ecc4157c1c44206e159aff0761e92a945aa0591",
            "3495919dabe2a35059ef2e1802ae59992fd7b3a14786378ac9f622b907c6da55",
            "5e88df9396c526cd97c00d7e1865a2175cf44f5271c85bad098d14013238df41",
            "925666e8a081d7b9f6b74ed57bd8e41533c20d7715b0bb47fabca5a465c4019",
            "332e4e5ff2e5d1afaaf9ef551934b1006ba305f26b5b35940e71605a5ebb5f56",
            "3462e730e81f90ddf9bb1046abaca984656932c13d1f00c387181c3c9aa43576",
            "65bad101fa269d55e51bbf694f5541225e26986350b4165ede5a7e1232355a69",
            "6e66ec021919cde6932d3b0d4c2c63076f0da7e33b3af529548304096d127502",
            "4c609941ec5da50d43b8d6d7d45fdd4faa8bb69929fc3337ddfc1bee29f7b94",
            "127f12060eb1a416ee0d304c538e094a13eb18310a2ecfd0fa81cce82a59e43d",
            "e247806a33437f19022c6958e51a172f6fd58853ef95d2ea3f8123ce9c2a399",
            "361564af11cea08fdc3afd9bd53471561356ad5b62e762c7d6023fbb12d5b7b",
            "6e12938c2d2d52577956a23d5df8a8e56d8e7a5bdfcc9cd3330835c9a865608b",
            "2c65d8fca4105323322504d653328c6692137481e686f256c3acf98b8888edd",
            "217451f2b930057065d940024678dabf1525e8375522a23da9186255df7514fc",
            "b2dddc8994767c7d3632cc7bc089becf8ef3b65540fb4709b8cc78ba12b044b",
            "5331126da252555886cd62e62bf8fb25c1040470bc827734f516c2f0c90fff3f",
            "317a581c6091951f08e8580adc43c1b02729900d2acb2de27e0f6b034b7d8c56",
            "6c6741993eb1d5bc90edd8ac667037865ea3e9c0788d3c319739a8bcc0893ba9",
            "54420b8489fa8145c279c03817315b31bc39445306be7f48fcba9a46c9f4f3b2",
            "2059da76bbb25f44687caae9f61e4afb811bf899a3aae060751049f7d21dd606",
            "6a30ab452ec8fe1e76c742ab33ec57448512ff27385a0a2bac70a1686d573570",
            "5416c3b67ec815aa7481f04a9b8dbde374786caea7d3f0ba4a98f326efec02a1",
            "4b32a6c01df4fdf6f2cc3bacec09b008ccc5c3644b69139c4346a3e75fab3f4b",
            "2e91572a13a6baf97560b43b5b862aebd8b7d95c0fda9c097d823cc9ef0599e",
            "18d4e26bbbdfa70d96ed89322834c8b1d36a3b3d373e9be7cfa588a8b5c0287b",
            "3d3f818cd10fcf2d1fe9ef125dfbe112e8298eca96f58b0b86f42608b976d165",
            "31e7074392297131067adb72832e19c1e271d312c551d7e4f40b441f942da24f",
            "245ddf2de52031410d4171e67579a57f7866bb3ae20a4525f82505d050a86281",
            "289673cb2bef13266ce3f7179e624b2e383d24a45ef6c375ba998e3fe9286a36",
            "3c59fba0c6311941376d9b0280c32e726d0711a734999892707246ba7b2bd32",
            "144c00621ab41c0c0f354ef520654c0150d61f502b7d923b9822a1a33294cab1",
            "68c6a95568f6ed64ff72c30387d7606d072448ecc708997317c6db6cf515cbaf",
            "662fe152dc7461f350a0a8c9286fa3d635ff00931d3a296e358345595f72ed3c",
            "217b043aadd7058a7e9270dc0a2f571a8d1ccd116297b85823de86d173e54321",
            "68d4303e7691e3a4795db36ea36432f8f1075438e9b0f8d1fb5999dfd4974d38",
            "1e26115ea5e4c4f066b84107cd8f6fce5792d77dda305a01790a54f3e234d210",
            "6e1bd58127f00ccc79c3609843b1ad75de0527f21df9eacfe29ab4c563a67753",
            "5a4aad3966fd75111f70775b9fa62a6d1c18702b26fe5851c83efa2b10954c92",
            "71c059faec1533beacad247017b29980008f0d937370301ee2401018fc2aa7da",
            "2b625e82f540d4603233baec3d48d81d9d855962b50771c6d5df82012044e896",
            "47e22b67d921cd1626e262ff3c739c646696e6336f1aedeb95881c62b511268b",
            "286509b96b3aa4a9d101a53e83b1b25fdd76dc3c00052c9126b8200d1449834e",
            "476bb19f615e3a38389ead4b38e8a61665c089682e7f7f46830d211db2616135",
            "67a0f1036c1628ed81e80f2493dbc5100b736843bb3a0d36f67d2b2dce99a192",
            "956eedcce3f1bb98cc45a3ad88ab894ddb3f7e775a11b961698f73c8381e07c",
            "63b48515137ba347cfff4389958351d07be7f13ee7187d4d5902d085637ee7bf",
            "3ff35869606dfa185b81adc1465fc268a4f481f75562aca9e4b46c00a77ec6c2",
            "729558bf05bd766305ddebb83018c6a52916acf31fd71085bd2165515bccaa86",
            "557fce386beeea241a2b8bb4fbb47cec057e235ca733fca67935761e140c61b3",
            "7f5af6d793912f4649026b8e7c55bafb8c14c003b296afdd2924c4540df0f45",
            "34ae79f5d988f866842080049ee7af47c48a7f2c1466638f8800259b4f2af2fb",
            "27fffd50aeb4aeac31469860bb68f2673d176f334f084440b8d806534f1d4698",
            "124f077ee1466fd7d3dc1e15b460663820dfb1cf542988480333d260f1ead81f",
            "12ad6f35913b3a56083aee7ee7e4a489ad73c400c030b2befa1cbc20313e359f",
            "3befad3a0b8f4debf05a376cd38129e0c87d7b446443611252269bd6f0206da0",
            "60974721e0b87c5dc35408f791d7843feed7f63cd5975a661ed67be0bffd343d",
            "4db00887ffb9981dba5da03142d103d18096731637be4bdf1831a261eb4abc1c",
            "3449ca4e443a46c14719d5771d05701bbbc4db571a3d7770240c3bc91c020dcb",
            "10e2e709f73f334e5f1373567b95d5e3edcf807f613826ea7120044c8444556f",
            "5d767c67116c8a0b388f24ac74212190be52a295c46cde008fdb8539ee58a49",
            "24616115f5f6421892eccc479da1b684c6b525bc1d3ff3cc95727863a2bc035d",
            "1f3973d80f425cf3e02e44930a273219df15dc4cb04c32ee086bccdfa6dc312",
            "338900eb90ef72de7560c97f7d8e64a68114e9ee696c0141fb6a922db16353e0",
            "8a621008ece8b2cad60e9cf048cb4cb8eb95e7a7c9d517ebaf165fec3387fc1",
            "3b3472a80d728abd9758e42fd22a478ebcd08b59ffa3c5e628e9e789a71a82a9",
            "167848c58dddcac256afdf24a93e12829f611534cc437bba34c774241cfc1812",
            "18c263472f9e8f2f262f6a572e33723761bb322cdc021b2cb4b136b0b74db77d",
            "1a54a00df68e3d7ec52e62b61c624b0b6951031d982315a46c444eb55347b669",
            "323e73ff092080d3d3c326c037bc64bf3c5553af4817447155cac913ee16232",
            "4b16f8918636214e2483bd6c0cac7ce1755891c8044b6a5a5848f8044382c9b4",
            "55da1b7e81416386c36dd95b752c15142b1225c39f88d269cca7cae381acabd8",
            "dbfbbf19841c1792826f69ad92b862e3800af884a9cc2166c43a8b02b64eb16",
            "37af211973056b0ee14d5776101e03e20360924d488bdee58b840a9cc65b530a",
            "33a2cf480bbaabc0529bd29b5ead59325ff6eb4eb99b83f8e4e52b8bdd8d8ceb",
            "10fe5117d64559e99e3bc90440f1d0c87ee1cbc7d14cbf524cc6e25c54291fcb",
            "71408145cdaa0a727a889eff3586f0755d76abf8e157ad07fb199cf1444cee49",
            "305f5892133b16e865cd1bcd3ca96f39a552e9e24ce724a6679d53fb4d421de1",
            "3792c249ab22a410bd9765026d09c4975767a364ed4ce8cda5c739d413538f4d",
            "54051fb18e4577eef62592a030adedcc11b22ade24a32e76f8bf68ef96039c22",
            "3562918322d14865722d461ee61e323c3988de5496d311e5e3b752a173d0f524",
            "4966ed088e26f77208302acb1977596cfed466aa7021ee9fa455a1568b9cc8ee",
            "217e57ad60015c9a4c3525239f1226f1a12b00dc220c2a3476edb9d6e33718b",
            "65f67c734e6dd080b1490d748c4ca54c3be080f68ff0983449d5de28dadad1c",
            "3db4cc8fd2f2f8e1478ad41b7c1e5c5ef19301bb87f44b49b378fe3e7e3a2264",
            "3261a8cb17034b0c32bc98cc77513ad895233f70e86d8ff6df57485ad194afc6",
        ];
        let coefficients: Vec<_> = coefficients
            .iter()
            .map(|hex_str| FE::from_hex(hex_str).unwrap())
            .collect();

        let system = &mut ConstraintSystem::<FrField>::new();
        let data = vec![system.new_variable()];
        let output = mimc(system, &coefficients, &data);

        let input_value =
            FE::from_hex("23a950068dd3d1e21cee48e7919be7ae32cdef70311fc486336ea9d4b5042535")
                .unwrap();
        let expected_output_value =
            FE::from_hex("136ff6a4e5fc9a2103cc54252d93c3be07f781dc4405acd9447bee65cfdc7c14")
                .unwrap();

        let inputs = HashMap::from([(data[0], input_value)]);
        let assignments = system.solve(inputs).unwrap();

        assert_eq!(assignments.get(&output).unwrap(), &expected_output_value);
    }
}
