// These constants can be found both in Jonathan's implementation
// https://github.com/xJonathanLEI/starknet-rs/blob/35c287e1a06e6ab68447f5f0b9df53f910960f57/starknet-crypto-codegen/src/poseidon/params.rs
// And the round 0 ones matches the one used
// in Cairo Lang
// https://github.com/starkware-libs/cairo-lang/blob/c98fc0b50529185b7018208cb3460191eeb53e0d/src/starkware/cairo/stark_verifier/air/layouts/starknet/autogenerated.cairo#L1574-L1596
    
pub const ADD_ROUND_CONSTANTS_HEXSTRINGS: [[&str; 3]; 91] = [["0x6861759ea556a2339dd92f9562a30b9e58e2ad98109ae4780b7fd8eac77fe6f", "0x3827681995d5af9ffc8397a3d00425a3da43f76abf28a64e4ab1a22f27508c4", "0x3a3956d2fad44d0e7f760a2277dc7cb2cac75dc279b2d687a0dbe17704a8309"], ["0x626c47a7d421fe1f13c4282214aa759291c78f926a2d1c6882031afe67ef4cd", "0x78985f8e16505035bd6df5518cfd41f2d327fcc948d772cadfe17baca05d6a6", "0x5427f10867514a3204c659875341243c6e26a68b456dc1d142dcf34341696ff"], ["0x5af083f36e4c729454361733f0883c5847cd2c5d9d4cb8b0465e60edce699d7", "0x7d71701bde3d06d54fa3f74f7b352a52d3975f92ff84b1ac77e709bfd388882", "0x603da06882019009c26f8a6320a1c5eac1b64f699ffea44e39584467a6b1d3e"], ["0x4332a6f6bde2f288e79ce13f47ad1cdeebd8870fd13a36b613b9721f6453a5d", "0x53d0ebf61664c685310a04c4dec2e7e4b9a813aaeff60d6c9e8caeb5cba78e7", "0x5346a68894845835ae5ebcb88028d2a6c82f99f928494ee1bfc2d15eaabfebc"], ["0x550a9e24176509ea7631ccaecb7a4ab8694ab61f238797098147e69dd91e5a3", "0x219dcccb783b1cbaa62773fedd3570e0f48ad3ed77c8b262b5794daa2687000", "0x4b085eb1df4258c3453cc97445954bf3433b6ab9dd5a99592864c00f54a3f9a"], ["0x53e8a8e8a404c503af2bf3c03e420ea5a465939d04b6c72e2da084e5aabb78d", "0x5ca045c1312c09d1bd14d2537fe5c19fb4049cb137faf5df4f9ada962be8ca8", "0x7c74922a456802c44997e959f27a5b06820b1ed97596a969939c46c162517f4"], ["0xc0bba6880d2e686bf5088614b9684ff2526a20f91670435dc6f519bb7ab83f", "0x4526bcaec43e8ebd708dd07234c1b2dc1a6203741decd72843849cd0f87934a", "0x1cc9a17b00d3607d81efaea5a75a434bef44d92edc6d5b0bfe1ec7f01d613ed"], ["0x28b1e269b84c4012aa8cdbead0bc1ce1eb7284e2b28ed90bc7b4a4fde8f01f", "0x62af2f41d76c4ad1d9a2482fbdaf6590c19656bcb945b58bb724dc7a994498d", "0x5cfd7e44946daa6b2618213b0d1bf4a2269bed2dc0d4dbf59e285eee627df1a"], ["0x7ff2afb40f3300856fdd1b94da8d3bbcf0312ab9f16ac9bc31955dc8386a747", "0x5cd236bdc15b54183e90bab8ae37f8aab40efae6fa9cd919b3248ee326e929c", "0x5463841390e22d60c946418bf0e5822bd999084e30688e741a90bbd53a698a"], ["0x24c940fff3fe8c8b2021f13eb4d71747efd44a4e51890ae8226e7406144f805", "0x4e50cb07b3873268dc88f05393d9d03153ca4c02172dd1d7fc77d45e1b04555", "0x62ca053e4da0fc87b430e53238d2bab1d9b499c35f375d7d0b32e1189b6dcb5"], ["0x719f20ac59d1ebcaaf37fe0b851bc2419cd89100adff965951bff3d3d7e1191", "0x7645ca5e87a9f916a82fe5bb90807f44050ac92ca52f5c798935cf47d55a8fd", "0x15b8aeaca96ab53200eed38d248ecda23d4b71d17133438015391ca63663767"], ["0x53d94dbbca7cb2aa8252f106292ac3b98799e908f928c196c1b658bf10b2e2", "0x28f90b403e240f1c6f4c0a3b70edbb3942b447c615c0f033913831c34de2d1e", "0x2485167dc233ba6e1161c4d0bf025159699dd2feb36e3e5b70ae6e770e22081"], ["0x1c8b08a90d6ee46ff7de548541dd26988f7fdaacdd58698e938607a5feca6e8", "0x105c3bf5cba256466b75e79d146f9880c7c4df5ecdad643ce05b16901c4881e", "0x238019787f4cc0b627a65a21bef2106d5015b85dfbd77b2965418b02dbc6bd7"], ["0x15e624d7698fdf9b73dce29a5f24c465c15b52dec8172923a6ebc99a6ddc5e1", "0x5d3688ba56f34fdf56bc056ad8bf740ca0c2efef23b04a479f612fde5800a0a", "0x229abdef3fef7ae9e67ed336e82dc6c2e26d872d98b3cce811c69ae363b444d"], ["0x3e8096ecfcbcde2ee400801a56f236db2c43d1e33c92b57ac58daf2d3fc44db", "0x3ad5fec670d7039108d605aae834c7ce6a7cd4e1b47bf6a02265352c57db9bd", "0x7cf4598c0cf143875877afdbb4df6794ef597fff1f98557adca32046aeaef0a"], ["0x58aecc0081b55134a4d1c4c8f27932e4170c37841fef49aca0ec7a123c00ad6", "0x757b4b7ee98e0a15460b71995790396e4ef3c859db5b714ec09308d65d2ca61", "0x6b82800937f8981f3cd974f43322169963d2b54fd2b7ed348dc6cc226718b5d"], ["0x3a915b1814707273427e34ab8fbb7ca044f14088fedae9606b34a60b1e9c64", "0x54afbf1bd990043f9bc01028ff44195c0bb609d367b76269a627689547bfbef", "0x5e1ceb846fe1422b9524c7d014931072c3852df2d991470b08375edf6e762bb"], ["0x7f751f98968212ebe5dff3ce06e8cb916709e0c48e3020c6b2b01c1bec0814b", "0x36f6b64463f7c29fc3180616e340536bea7f01d226b68b6d45cd6dfbff811e4", "0x61135c9846faf39b4511d74fe8de8b48dd4d0e469d6703d7ed4fe4fe8e0dbac"], ["0xb58921a3fbdbb559b78f6acfca9a21a4ba83cc6e0ae3527fbaad907fc912b8", "0x22a4f8a5cdc7474b9d16b61c2973847211d84eb2fb27b816e52821c2e2b1b1e", "0x41cf6db5d6145edfeccbbc9a50b2ceedeb1765c61516ffcb112f810ad67036f"], ["0xbe44689973db2b1cfc05fa8f4aec6fac6a0ff2fdfab744ade9de11416b6831", "0x39bf209c4e117e16489cda45128096d6d148a237142dc4951df0b8239be148b", "0x209cf541e5f74fc2b93310b8ce37b092a58282643860b5707c7eb980ea03a06"], ["0x6b562e6005f34ee0bdc218ba681b6ba7232e122287036d18c22dd5afa95326d", "0xe8103a23902be5dc6d5f59253a627a2a39c8aca11a914670e7a35dea38c8f", "0x6a3725548c664fd06bdc1b4d5f9bed83ef8ca7468d68f4fbbf345de2d552f72"], ["0x67fcd6997472e8e605d0f01a8eccc5f11a45c0aa21eb4ebb447b4af006a4a37", "0x26144c95c8de3634075784d28c06c162a44366f77792d4064c95db6ecb5cff0", "0x5b173c8b0eb7e9c4b3a874eb6307cda6fd875e3725061df895dc1466f350239"], ["0x7e1c2d6fde8ac9f87bae06ad491d391c448f877e53298b6370f2165c3d54ddb", "0x4db779f3e5b7424996f451b156fe4e28f74d61e7771f9e3fa433b57ca6627a9", "0xbb930d8a6c6583713435ec06b6fed7825c3f71114acb93e240eed6970993dd"], ["0x4472d73b2830565d708467e9296fb5599d3a08814c31c4189e9579c046e878f", "0x7ba9c303dfee2d89e10e3c883ca5ce5614d23739b7cb2052cc23612b11170e2", "0x21c0e3319ede47f0425dc9b2c1ed30e6356cb133e97579b822548eb9c4dc4b7"], ["0x2cfd61139e50ddd37b09933816e2a0932e53b7dc4f4947565c1d41e877eb191", "0x5abea18941a4976844544d92ee0eca65bdd10b3f170b0dc2f30acd37e26d8e7", "0x77088fdb015c7947a6265e44fef6f724ea28ae28b26e6eee5a751b7ce6bcc21"], ["0x3abdc9d677231325b3e3c43cfd443076b4ce33cddbc8446120dce84e6122b73", "0x2250f430b7fe7d12e5d00b6b83e52a52ca94879ccfab81a7a602662c2d62c4d", "0x5c92ef479c11bb51fb24ef76d57912b12660e7bd156d6cabbb1efb79a25861b"], ["0x235ec597391648b510f616fa8b87900fd08fd4208a785cffcf784a63a0fd5c6", "0x4ed4e872eb7e736207be77e9d11e38f396b5c0ba3376e855523c00b372cc668", "0x5f9406febca3879b756ef3f6331890b3d46afa705908f68fb7d861c4f275a1b"], ["0x1d9c501d9ff1fba621a9f61b68873c05f17b0384661f06d97edf441abdaa49d", "0x4b0de22bbd0a58534982c8e28d2f6e169e37ba694774c4dfa530f41c535952e", "0x1b4d48bd38a3f8602186aabb291eca0d319f0e3648b2574c49d6fd1b033d903"], ["0x7558bbea55584bf1725d8aa67ddba626b6596bbd2f4e65719702cefcead4bab", "0x1108f1a9500a52f561ea174600e266a70b157d56ece95b60a44cf7a3eef17be", "0x8913d96a4f36b12becb92b4b6ae3f8c209fb90caab6668567289b67087bf60"], ["0x6502262c51ad8f616926346857dec8cca2e99f5742b6bf223f4d8a6f32867a6", "0x7cb5fcdc00892812889280505c915bde962ea034378b343cd3a5931d2ec0e52", "0x2eb919524a89a26f90be9781a1515145baea3bc96b8cd1f01b221c4d2a1ce87"], ["0x58efb6272921bc5eada46635e3567dced0662c0161223e3c1c63e8de3ec3d73", "0x62fcd49ca9c7587b436d205ffc2a39594254a1ac34acd46d6955e7844d4f88e", "0x635895330838846e62d9acce0b625f885e5941e54bd3a2106fcf837aef5313b"], ["0x7da445b81e9b3d36d47a5f4d23b92a378a17f119d5e6e70629f8b41fefb12e3", "0x2b22dab62f0817e9fc5737e189d5096a9027882bef1738943b7016256118343", "0x1af01472348f395bacdfed1d27664d0d5bdea769be8fcb8fbef432b790e50d5"], ["0x76b172dbbeec5a31de313b9390f79ec9284163c8e4986bc5b682e5ac6360309", "0x70efaeae36f6af0f362f6cb423d2009b30ddb4178d46def0bdb2905b3e0862", "0x6cb99b36e521ac0a39872686b84ee1d28c4942b8036a1c25a0e4117ccaeedf"], ["0x29fd44305a5a9a70bbf9674e544bda0fb3d0fe5bb3aa743fd1b8a4fc1dc6055", "0x6b447ded1046e83629b184d8c36db3a11a6778d8848142aa6363d6619f9764", "0x642a8b4be4ba812cbfcf55a77339b5d357cceb6946fdc51c14b58f5b8989b59"], ["0x489e0a26f65a1eecc6cc6aa5b6e775cbc51a73700bd794a7acd79ae1d95882a", "0x3b19d4ef195975bbf78ab5dc2fd1d24816428f45a06293c1b9d57b9a02e9200", "0x7d2dd994756eacba576b74790b2194971596f9cd59e55ad2884c52039013df5"], ["0x1922810cc08f50bf300df869823b9f18b3327e29e9e765002970ef0f2e8c5f3", "0x52f3afaf7c9102f1d46e1d79a70745b39c04376aafff05771cbd4a88ed418ac", "0x7ccfc88e44a0507a95260f44203086e89552bbe53dcc46b376c5bcab6ea788e"], ["0x2949125939e6ad94100228beff83823f5157dd8e067bc8819e40a1ab008dd9c", "0x6cb64e3a0d37a6a4273ce4ee6929ba372d6811dde135af4078ba6e1912e1014", "0xd63b53707acf8962f05f688129bf30ad43714257949cd9ded4bf5953837fae"], ["0xbcb1549c9cabb5d13bb968b4ea22d0bb7d7460a6965702942092b32ef152d4", "0x3d1c5233657ce31f5ead698fe76f6492792a7205ba0531a0ca25b8d8fe798c1", "0x2240b9755182ee9066c2808b1e16ea448e26a83074558d9279f450b79f97516"], ["0xcc203d8b0f90e30fe8e54f343cef59fe8d70882137de70c9b43ab6615a646c", "0x310c6cc475d9346e061bacdc175ea9e119e937dea9d2100fa68e03c1f77910b", "0x7f84b639f52e57420bc947defced0d8cbdbe033f578699397b83667049106c7"], ["0x584ca7f01262c5bd89c4562f57139f47e9f038cb32ec35abe4e1da8de3e164a", "0x1135eefaf69b6e4af7d02f562868be3e02fdc72e01e9510531f9afa78abbbde", "0x372082b8a6c07100a50a3d33805827ad350c88b56f62c6d36a0d876856a99e8"], ["0x7c3c12b819a8aad87499bac1a143fc59674f132e33898f0c119e3d12462dfe6", "0x4f1354c51e8f6905b84157cfeff6822c056ce9e29d602eb46bd9b75a23836cf", "0x2da9f26a8271659075739ba206507a08ac360150e849950ef3973548fbd2fca"], ["0x287173956a2beb111b5ec29195e38cc3f6a65ff50801aa75fd78dd550702843", "0x7273101c190ff64212420095a51c8411c7f3227f6a7a4a64ae6ba7f9201e126", "0x2dbf2a6b56b26d23ebeb61e500687de749b03d3d349169699258ee4c98005fc"], ["0x85b6cbb29739a6808e67f00ab89b52ab89ef8d92530394e4b910efd706c7fb", "0x3d55b5f1171efda1dacbcbadfd5b910b493fa9589fd937e3e06ce26b08925a3", "0xaaedaa6ef2fa707d16b3b295410c0e44f7a2f8135c207824f6ae2a9b16e90c"], ["0x6aca6ebf70b1cb46c6331e9f1a5c4cc89b80f8adc5d18915c1cd0d496ccf5e1", "0x1678602af36c28abb010f831d403d94d5e90003e6d37c677e9dd157fb27761", "0x2022036bdf687f041b547fefdf36d4c2cd3f4b0526a88aafe60a0a8f508bad2"], ["0x7bfc350957c968ca664397414bdfb8f9b8dfe49fb63e32353d4e2e8d1d4af6", "0x2d639cbd418cb9fc24ea29ccd1d15ab81f43a499b27a06d3c5e2176f7ad79af", "0xecdea7f959a4d488403d5b39687a1fe0dee3369e5fbc0f4779569f64506e0c"], ["0x3f656bdc4fefd92b70658e2f1992ef9f22e5f2d28c490e21d4e34357154b558", "0xd1b8cb1561eed32319638ccab9033dfec47596f8a6f4ce6594e19fddd59254", "0x758ffc77c62e3e0f86ef6ea01545ad76f281ec2941da7222d1e8b4e2ec1f192"], ["0x20315ca079570df995386e96aeaa1b4596aacd28f83c32f29a591c95e6fcac5", "0x3e55cf341e7c280cb05f3d6ff9c8d9f2cfe76b84a9d1b0f54884b316b740d8d", "0x4d56feb32cde74feede9749739be452e92c029007a06f6e67c81203bf650c68"], ["0x4ee807aa678a9a433b6171eaa6a2544497f7599fb8145d7e8089f465403c89b", "0x25d2bacc8f1ee7548cb5f394de2cb6e1f365e56a1bc579d0f9a8ad2ef2b3821", "0x5f573de597ce1709fc20051f6501268cd4b278811924af1f237d15feb17bd49"], ["0x30297c3c54a505f5826a280e053cf7a3c1e84a1dcf8b33c682cf85ddac86deb", "0x2f5e9c47c9a86e043c7526a59783f03c6bc79b69b8709fe6a052b93a8339ae8", "0x1bf75c7a739da8d29f9c23065ff8ccb1da7deec83e130bcd4a27a416c72b84b"], ["0x60563d5f852ae875989017bd5c4cfdc29cd27fc4e91eeabdb8e864df3c3c675", "0x7a4b1d70885aa820969635468daec94f8156c20e3131bd71005be1cd16ccf9e", "0x347bb025695e497f1e201cd62aa4600b8b85cf718cd1d400f39c10e59cc5852"], ["0x6783ab1e1ef97bb9e7f9381eb6ab0de2c4c9c2de413691ba8aa666292e9e217", "0x133e0280c6de90e7b3870a07823c081fd9c4cb99d534debd6a7bfb4e5b0dd46", "0x865d450ce29dc42fb5db72460b3560a2f093695573dff94fd0216eb925beec"], ["0x1de023f840e054a35526dabacf0dee948efba06bcbb414ecd81a6b301664e57", "0x55fc1e341bfdf7805015a96f724c5ac7cc7b892a292d38190631ab1a5388c4", "0x2df6557bfd4a4e7e7b27bf51552d2b5162706a3e624faca01a307ef8d532858"], ["0x113a8a66962ce08d92a6bd3e9c1d55ef8f226da95e4d629046d73d0507f6271", "0x271577d6ee9fa377f2c889874ba5b44ca1076033db5c2de4f3367b08c008e53", "0x3396b33911219b6b0365c09348a561ef1ccb956fc673bc5291d311866538574"], ["0x1e1392f2da08549c8a7d89e899189306170baa3c3436e6a5398f69c8f321636", "0x661545081032013df118e1d6e7c61a333e313b1a9a5b6d69c876bd2e7d694ca", "0x6b14294e71cd7fb776edbd432d20eb8f66d00533574e46573516f0cacdeec88"], ["0x7252fbbb06c2848338b1c41df31e4e51fe2a18e2406c671915cab6eb1a1d4f2", "0x3ccf71be7cc2a9abcf5a09807c69679430c03645747621b7f5327cb00ff99da", "0x29778dc707504fa6a9f7c97b4ceef0a9b39001d034441617757cd816dac919a"], ["0x39473f6f06bb99e33590d34e3bae36e491f7bbf86a26aa55a8f5b27bb98d4c5", "0x7ba7c32f875b71b895caa0215f996fd4ad92bab187e81417063dde91c08c027", "0x37c1367e49cbfc403b22aac82abf83b0ed083148a5f4c92839e5d769bdab6b6"], ["0x5c9eb899931d2f4b53ffcf833cdfa05c2068375ff933eb37ae34157c0b2d951", "0x5f6054a4d48698ec27772fb50a7d2e5c1557ffdc1ffd07331f2ca26c6e3b661", "0x20e6d62a2fe0fe9b0fab83e8c7d1e8bfd0fec827960e40a91df64664dcd7774"], ["0x6290a56a489ad52120c426fe0e409c2ff17adf51f528cafb0d026d14ffd6aac", "0x3703f16f990342c2267a6f7ece342705a32ca4c101417286279f6fc315edc7c", "0x5194962daf6679b9a0c32b5a9a307ba92e2c630f70e439195b680dd296df3fd"], ["0xe8eae20a79a7c1242c34617b01340fb5fd4bea2aa58b98d2400d9b515ee5e2", "0x369058169d63091ae28bfb28def7cd8d00dd7c2894fae4ffec65242afa5cd45", "0x418c963bc97195a74077503ee472f22cfdff0973190ab189c7b93103fd78167"], ["0x68d07a3eefc78dc5b28b3f4dc93167fb8c97112d14a25b4d4db559720156386", "0x517e892228df2d4f15a3c4241c98ba25ba0b5557375003f8748583a61836372", "0x5cc0f0f6cf9be94a150116e7932f8fe74ac20ad8100c41dc9c99538792e279b"], ["0x53d5d7863434c6629bdb1f8a648e4820883543e821f0f5c1668884c0be41ec8", "0xa158126b89e6b0a600bf53f8101707b072218912dd0d9df2528f67de24fdf5", "0x6b53b807265387ee582069a698323d44c204bed60672b8d8d073bed2fede503"], ["0x1097fb448406b7a6de0877efd58c01be53be83bde9601a9acc9e0ca2091fda0", "0xcbc0ff7239d3763902396389d67b3049ce1fefde66333ce37ca441f5a31bec", "0x79a3d91dd8a309c632eb43d57b5c5d838ceebd64603f68a8141ebef84280e72"], ["0x23fb472fe575135300f74e8f6de8fe1185078218eceb938900e7598a368db9", "0x7ac73134016d2a8a4c63a6b9494c0bd7a6ba87cc33e8a8e23ebda18bfb67c2a", "0x19a16068c3eac9c03f1b5c5ee2485ccc163d9ab17bb035d5df6e31c3dcf8f14"], ["0x1f24b4356a6bbfd4d4ef9fd1634752820ee86a925725ac392134d90def073ea", "0x3e44e7f7aeea6add59b6b4d11c60a528fb70727f35d817305971592333d36", "0x5f93b02f826741414535a511ed3eb4fe85987ae57bc9807cbd94cd7513d394e"], ["0xf0a0a88db99247d71c3d51d4197fa3fd1cc76e670607e35ca2d3bada29523a", "0x3432226916d31f3acac1e211431fd4cd2b6f2e80626af6564bdde3e77608db0", "0x55625941bfea6f48175192845a7ad74b0b82940ef5f393ca3830528d59cf919"], ["0xddf48695b204477dfe4f8cb3ef1b39783e9b92f9276b858e2e585e318e20a4", "0x260730a657ff8f38851a679ab2a1490434ee50d4953e7c5d3194578b08ae8e3", "0x4cfd231373aa46d96283840bdb79ba6d7132775b398d324bcd206842b961aa9"], ["0x3203843c41cd453f14fa0bc0b2191a27ebc659e74fd48f981e963de57eff25d", "0x2c2f6ae5624d1fb8435d1c86bf76c260f5e77a54b006293705872e647cc46", "0x780225456e63903b3e561384ef2e73a85b0e142b69752381535022014765f06"], ["0x7f602ec1a80a051fd21b07f8e2960613082fc954b9a9ff641cc432a75c81887", "0x62561b0a0a72239b60f6aaf7022b7d323fe77cd7c1ab432f0c8c118ca7e6bca", "0x604fe5a6a22344aa69b05dea16b1cf22450c186d093754cb9b84a8a03b70bc8"], ["0x1cf9987a4044716d3dc140bf5f9b76f6eada5995905189f8682eaf88aef2b7b", "0x6bc0b2487c1eece3db47a4bdd60cf69debee233e91b50e9ee42ce22cbfbacbf", "0x2f5dbb5055eb749a11403b93e90338b7620c51356d2c6adcbf87ab7ea0792e6"], ["0x446328f4dddae6529743c43883d59c45f63b8a623a9cf318489e5fc4a550f61", "0x4ba30c5240cde5bca6c4010fb4b481a25817b43d358399958584d2c48f5af25", "0x5f5275f76425b15c89209117734ae85708351d2cf19af5fe39a32f89c2c8a89"], ["0x576f3b5156f4763e18c7f98df3b2f7b993cdda4eb8cb92415e1be8e6af2fc17", "0x11dc3f15cba928aed5a44b55a5b026df84a61719ed5adbb93c0e8e12d35ef3d", "0x44c40e6bd52e91ad9896403ae4f543ae1c1d9ea047d75f8a6442b8feda04dca"], ["0x1836d733a54013ebd0ccbf4974e80ac1954bf90fe9ea4e2c914ad01166026d8", "0x3c553be9776b628a8159d306ef084727611df8037761f00f84ca02ce731b3ac", "0x6ce94781c1a23fda1c7b87e0436b1b401ae11a6d757843e342f5017076a059"], ["0x381ec71fbdef3160253be9f00f4e6b9e107f457812effb7371cc2daa0acd0ed", "0x1844da9cc0eeadc6490d847320d9f3cd4fb574aa687bafdfe0ffa7bf2a8f1a1", "0x7a8bf471f902d5abb27fea5b401483dedf97101047459682acfd7f9b65a812f"], ["0x633b6fb004de62441915fb51ac174456f5a9cdff7aecb6e6b0d063839e56327", "0x179ee5cec496194771200382bfc6d17bbe546ba88fed8b17535fd70fbc50ab6", "0x2806c0786185986ea9891b42d565256b0312446f07435ac2cae194330bf8c42"], ["0x438703d948708ae90c7a6b8af194b8b603bb2cdfd26bfa356ac9bb6ee041393", "0x24446628f56029d7153bd3a482b7f6e1c56f4e02225c628a585d58a920035af", "0x4c2a76e5ce832e8b0685cdeeea3a253ae48f6606790d817bd96025e5435e259"], ["0x78a23323520994592933c079b148aed57d5e4ce1ab122d370983b8caa0e0300", "0x79ca6c5e1025b2151144ea5937dd07cadce1aa691b19e6db87070ba51ec22c0", "0x6b2e4a46e37af3cf952d9d34f8d6bd84a442ebfd1ac5d17314e48922af79c5d"], ["0x305d6cd95cc2eab6805d93d3d8d74e1ca7d443f11e34a18e3529e0d03435c2", "0x6097b4b8b90db14b39743ed23f8956cabb7aea70cc624a415c7c17b37fbf9a9", "0x64e1b3f16c26c8845bdb98373e77dad3bdcc90865b0f0af96288707c18893f"], ["0x649fafe673f21e623384d841221b73421c56014af2ffdf57f1579ae911fd335", "0x7d806dccbf1a2696b294404e849722f2baa2f4d19005a49d1ba288a77fefe30", "0x5951a37da53e3bbc0b3e2db1a9a235d7a03f48f443be6d659119c44aafc7522"], ["0x6d87fa479fb59524d1912c3554ae3d010496a31bdacb542c816a1607a907731", "0x1451cccd4200fa9d473ad73466b4e8c0a712a0b12bb6fc9462a3ac892acc9b2", "0x3ca1b6400b3e51007642535f1ca9b03832ca0faa15e1c4ed82dd1efdc0763da"], ["0x52c55735b2f0a6560ad1516a8f13592b0dd024ff4162539f993a99c7a1a4d95", "0x7e04de60aa80132f0149d1dee29617de750bd5ce3e9fa5e62951d65f6b924cd", "0x271784e6920a68e47c4c8fab71c8f8303ef29e26f289223edf63291c0a5495"], ["0x5c7c19061a84d5960a04b8f0adaa603c8afe93f17b7f0e56b49514af43d0c69", "0x172db5affe783af419da337cb79061e090943c2959dea1b38e4436f5482eafe", "0x518b7975a6d8d310eac9fe4082916f021a7ecbadf18809746a9e061a2cb9456"], ["0x20c5539dc45dd56d4bbc2440a9f5061d74b8ae5e37b34e8755a0315f1e196db", "0x1ea6f5fb309fa4a08bc7d516e80efc3a977b47208283cf35a9d8bc213b90b14", "0x50ce323c5128dc7fdd8ddd8ba9cfe2efd424b5de167c7257d1f766541e29ded"], ["0x401e37d0e276547695538b41d3c28215b865f5b7d1b497a8919284c613cb7d8", "0x645a0de30acc3117f2893056fc5880255daa12cc61261cc0fab9cf57c57397b", "0x69bc3841eb0a310d9e988d75f09f698d4fdc9d0d69219f676b66ae7fa3d495b"], ["0x2684bbe315ad2c4bdd47c38fe72db47cf0ae0c455cda5484baf523f136bdc6", "0x11e0f83c547ca5c68202e8d34e5595a88858c2afa664365e4acb821fd8a13ee", "0x4af4a7635f8c7515966567ceec34315d0f86ac66c1e5a5ecac945f1097b82ef"], ["0x4fba58cf8aaf4893cb7158908ccc18b1dc48894d2bb46225c72b11f4c74b271", "0x397c4c169115b468cc90da2e664f8c29a7f89be0ead679a38b0f44c8a2a0e20", "0x6563b9ebb6450dbad397fa5dd13c501f326dd7f32be22e20998f59ec7bacff"], ["0x376edb238f7b630ea81d307f4c79f9afec48562076dd09c36cd79e9cb817165", "0x60d4208bb50eb15f29ed22addcd50a1b337504039690eb858584cda96e2e061", "0x6a37d569d2fbc73dbff1019dc3465ec0f30da46918ab020344a52f1df9a9210"], ["0xd3b174c7290c6bf412083ff35d23821dc512f1df073c1b429130371ac63b1a", "0x226ed3d763477454b46eb2a5c3b814634d974919689fb489fe55e525b980373", "0x5f3997e7dafcb2de0e7a23d33d2fd9ef06f4d79bd7ffa1930e8b0080d218513"], ["0x7c5eec716d94634434df335a10bbac504f886f7f9d3c1648348c3fae8fdf14d", "0x53cc30d7fe0f84e7e24fd22c0f9ad68a89da85553f871ef63d2f55f57e1a7c", "0x368821ee335d71819b95769f47418569474a24f6e83b268fefa4cd58c4ec8fa"], ["0x5334f75b052c0235119816883040da72c6d0a61538bdfff46d6a242bfeb7a1", "0x5d0af4fcbd9e056c1020cca9d871ae68f80ee4af2ec6547cd49d6dca50aa431", "0x30131bce2fba5694114a19c46d24e00b4699dc00f1d53ba5ab99537901b1e65"], ["0x5646a95a7c1ae86b34c0750ed2e641c538f93f13161be3c4957660f2e788965", "0x4b9f291d7b430c79fac36230a11f43e78581f5259692b52c90df47b7d4ec01a", "0x5006d393d3480f41a98f19127072dc83e00becf6ceb4d73d890e74abae01a13"], ["0x62c9d42199f3b260e7cb8a115143106acf4f702e6b346fd202dc3b26a679d80", "0x51274d092db5099f180b1a8a13b7f2c7606836eabd8af54bf1d9ac2dc5717a5", "0x61fc552b8eb75e17ad0fb7aaa4ca528f415e14f0d9cdbed861a8db0bfff0c5b"]];
