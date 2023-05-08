window.BENCHMARK_DATA = {
  "lastUpdate": 1683574715735,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "estefano.bargas@fing.edu.uy",
            "name": "Estéfano Bargas",
            "username": "xqft"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "657f6b6cfe7e67f25483dc63d6345908e8d978c2",
          "message": "Added support to all FFT implementations for inputs of all sizes via zero padding (#275)\n\n* Changed param name, coeffs to input\n\n* Added non-pow-2/edge cases support for CPU FFT\n\n* Removed next_pow_of_two helper, added tests\n\n* Removed null input edge case handling\n\n* Added zero-padding to Metal FFT API\n\n* Fix clippy\n\n* Use prop_assert in proptests\n\n* Added FFT special considerations doc\n\n* Changed get_twiddles to accept u64\n\n* Update doc",
          "timestamp": "2023-05-02T16:32:38Z",
          "tree_id": "c8f6ee9879a8fe57dacf95cb68d507277d1565e8",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/657f6b6cfe7e67f25483dc63d6345908e8d978c2"
        },
        "date": 1683045506030,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 136815744,
            "range": "± 3620787",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 268674208,
            "range": "± 1220051",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 527235917,
            "range": "± 2278131",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1037766812,
            "range": "± 11439154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33780581,
            "range": "± 198004",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69367016,
            "range": "± 740200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132752914,
            "range": "± 1011997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275326114,
            "range": "± 3253874",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31082369,
            "range": "± 197805",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59602708,
            "range": "± 985765",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120897781,
            "range": "± 3509269",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 253219635,
            "range": "± 22944187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 169710917,
            "range": "± 773257",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329420343,
            "range": "± 1570501",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 647299854,
            "range": "± 4170879",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1310941417,
            "range": "± 10305694",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 467693489,
            "range": "± 2256462",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 931061583,
            "range": "± 5023126",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1832943812,
            "range": "± 4048020",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3578632666,
            "range": "± 71867245",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ajgarassino@gmail.com",
            "name": "ajgara",
            "username": "ajgara"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d3e0eb9d72b3ee5979f3da9cf71a119bd3492add",
          "message": "Add rounds to stark prover (#287)\n\n* Add first round\n\n* Add round 1.2\n\n* Add round 2\n\n* Add round 3\n\n* Add round 4\n\n* Cargo clippy and fmt\n\n* minor refactor\n\n* move round structs to the top\n\n* move z squared\n\n* Add Domain struct\n\n* clippy and change argument order\n\n* move domain to lib\n\n---------\n\nCo-authored-by: Sergio Chouhy <sergio.chouhy@gmail.com>",
          "timestamp": "2023-05-02T16:40:18Z",
          "tree_id": "a22a34d4220920eba13ee7adb6ecee1083600a1c",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d3e0eb9d72b3ee5979f3da9cf71a119bd3492add"
        },
        "date": 1683045968999,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 136477882,
            "range": "± 2341265",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 268745833,
            "range": "± 1241266",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 525875104,
            "range": "± 1753365",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1039331771,
            "range": "± 14126556",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34015606,
            "range": "± 265055",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69597009,
            "range": "± 331461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132645075,
            "range": "± 1117401",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275123198,
            "range": "± 2416564",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30667477,
            "range": "± 115609",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59242220,
            "range": "± 741023",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120283208,
            "range": "± 3246735",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 252202777,
            "range": "± 15481054",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 168712640,
            "range": "± 448904",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 331392697,
            "range": "± 1682752",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 655695500,
            "range": "± 10193753",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1334686104,
            "range": "± 7841618",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 468450843,
            "range": "± 2813544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 926224145,
            "range": "± 6868910",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1819014604,
            "range": "± 12849323",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3584753229,
            "range": "± 26403714",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c611e65d9717b7e5c47f5b6f662e65ede76e606d",
          "message": "Sqrt (#290)\n\n* Initial version of sqrt\n\n* Remove graph\n\n* Formatting, fix non_qr\n\n* Fixed sqrt()\n\n* Clippy and nits\n\n* Changed test name\n\n* Fix\n\n* Remove unused file\n\n* Remove unused random fn\n\n* Reverted change that should be in another pr\n\n* Cargo fmt\n\n* Add test for more curves\n\n* Remove comments\n\n* Remove unused trait requirement\n\n* Add test\n\n* Renamed tests\n\n* Improve documentation\n\n* Add test for 0 in sqrt\n\n* fix linter\n\n* Add Legendre Symbol enum\n\n* Cargo fmt\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>",
          "timestamp": "2023-05-02T19:09:05Z",
          "tree_id": "47bd83fe09c2f3a703125da74a3949348f3902f7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c611e65d9717b7e5c47f5b6f662e65ede76e606d"
        },
        "date": 1683054929354,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 136108899,
            "range": "± 2207487",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 270473614,
            "range": "± 1645609",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 527938375,
            "range": "± 3892215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1045352791,
            "range": "± 8280853",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34116432,
            "range": "± 333197",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69780829,
            "range": "± 1059926",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132438281,
            "range": "± 567454",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 274870906,
            "range": "± 3869869",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31306986,
            "range": "± 161299",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59268259,
            "range": "± 574775",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 115542591,
            "range": "± 4496446",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 235994909,
            "range": "± 12077695",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 179129065,
            "range": "± 5647910",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 336136083,
            "range": "± 42603773",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 844292874,
            "range": "± 36227479",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1704104812,
            "range": "± 31984091",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 557882979,
            "range": "± 70808761",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1243271833,
            "range": "± 162719371",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2904258750,
            "range": "± 70062483",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3746534375,
            "range": "± 65748056",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43053772+diegokingston@users.noreply.github.com",
            "name": "Diego K",
            "username": "diegokingston"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f707f7d1107d8f9b7772a29a71ee37070731f7bb",
          "message": "Update References.md (#285)",
          "timestamp": "2023-05-02T19:11:51Z",
          "tree_id": "8a35db14604d98c8ff00d4448795cdade07c386a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f707f7d1107d8f9b7772a29a71ee37070731f7bb"
        },
        "date": 1683055079905,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 203582437,
            "range": "± 20162356",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 398108916,
            "range": "± 44516106",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 768473604,
            "range": "± 122482740",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1704502833,
            "range": "± 148302912",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 48324329,
            "range": "± 11866699",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 102098870,
            "range": "± 29059720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 202927354,
            "range": "± 50386806",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 363375343,
            "range": "± 104147028",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31004112,
            "range": "± 1276422",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59921343,
            "range": "± 1258479",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 114751337,
            "range": "± 4475076",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 225428618,
            "range": "± 6442925",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 168276185,
            "range": "± 299564",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 331030167,
            "range": "± 1738274",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650847062,
            "range": "± 3104896",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1310437875,
            "range": "± 9645097",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 465463749,
            "range": "± 2894563",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 916943021,
            "range": "± 4490279",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1834633187,
            "range": "± 7459109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3547254583,
            "range": "± 20982172",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41742639+schouhy@users.noreply.github.com",
            "name": "Sergio Chouhy",
            "username": "schouhy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "89aeeb43b8454e64f6aa41f14d0a0c8bde1e5cdb",
          "message": "Add steps to stark verifier (#289)\n\n* move z challenge to step 0\n\n* incorporate domain\n\n* add boundary and transitions coefficientes to challenges\n\n* move code to step 2\n\n* move fri challenges to step 1\n\n* move fri check to step 3\n\n* move deep composition polynomial check to own step 4 function\n\n* sort function parameters\n\n* fix comments\n\n* fix signature\n\n---------\n\nCo-authored-by: MauroFab <maurotoscano2@gmail.com>",
          "timestamp": "2023-05-02T19:33:32Z",
          "tree_id": "4d5ed36a4c4bdbb60b2cb5ebacde1918a730b291",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/89aeeb43b8454e64f6aa41f14d0a0c8bde1e5cdb"
        },
        "date": 1683056363935,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 137173174,
            "range": "± 1570424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 269598812,
            "range": "± 962496",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 527134729,
            "range": "± 3970333",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1043547333,
            "range": "± 29334253",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33537482,
            "range": "± 243812",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69552093,
            "range": "± 590848",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132594219,
            "range": "± 430856",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 281093718,
            "range": "± 672535",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31218172,
            "range": "± 111858",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59697544,
            "range": "± 598228",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 124296876,
            "range": "± 6543566",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 260563551,
            "range": "± 15245932",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 170297686,
            "range": "± 361950",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 335214958,
            "range": "± 1405242",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 666613416,
            "range": "± 13704042",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1332295750,
            "range": "± 14490322",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 464748437,
            "range": "± 2065137",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 926197916,
            "range": "± 8593390",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1823934687,
            "range": "± 11782066",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3566940458,
            "range": "± 22145571",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "estefano.bargas@fing.edu.uy",
            "name": "Estéfano Bargas",
            "username": "xqft"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "657f6b6cfe7e67f25483dc63d6345908e8d978c2",
          "message": "Added support to all FFT implementations for inputs of all sizes via zero padding (#275)\n\n* Changed param name, coeffs to input\n\n* Added non-pow-2/edge cases support for CPU FFT\n\n* Removed next_pow_of_two helper, added tests\n\n* Removed null input edge case handling\n\n* Added zero-padding to Metal FFT API\n\n* Fix clippy\n\n* Use prop_assert in proptests\n\n* Added FFT special considerations doc\n\n* Changed get_twiddles to accept u64\n\n* Update doc",
          "timestamp": "2023-05-02T16:32:38Z",
          "tree_id": "c8f6ee9879a8fe57dacf95cb68d507277d1565e8",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/657f6b6cfe7e67f25483dc63d6345908e8d978c2"
        },
        "date": 1683057621625,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2368857143,
            "range": "± 1079730",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4957378226,
            "range": "± 14954388",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4952746073,
            "range": "± 3691253",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10568520257,
            "range": "± 188184103",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 10340958200,
            "range": "± 4671558",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 22999101454,
            "range": "± 150391873",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 21553852249,
            "range": "± 8614355",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 48912030869,
            "range": "± 256195635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1152936630,
            "range": "± 222619",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2419020299,
            "range": "± 515743",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5065948490,
            "range": "± 1276136",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10575959186,
            "range": "± 1383681",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 83092680,
            "range": "± 1330411",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 166256788,
            "range": "± 947117",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 331774959,
            "range": "± 2030967",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 673023198,
            "range": "± 1236985",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3570202016,
            "range": "± 2132000",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7467005280,
            "range": "± 3120258",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 15593340055,
            "range": "± 11388430",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 32496230827,
            "range": "± 20507135",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 58848785834,
            "range": "± 6772303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 118080757391,
            "range": "± 20627609",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 236457882085,
            "range": "± 39438961",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 474687522455,
            "range": "± 4753183078",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 3092,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 395702,
            "range": "± 172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1906,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 2505,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1899,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1855,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 34031425,
            "range": "± 10659",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 47706382,
            "range": "± 8589",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 315915,
            "range": "± 91",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 34201976,
            "range": "± 9064",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ajgarassino@gmail.com",
            "name": "ajgara",
            "username": "ajgara"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d3e0eb9d72b3ee5979f3da9cf71a119bd3492add",
          "message": "Add rounds to stark prover (#287)\n\n* Add first round\n\n* Add round 1.2\n\n* Add round 2\n\n* Add round 3\n\n* Add round 4\n\n* Cargo clippy and fmt\n\n* minor refactor\n\n* move round structs to the top\n\n* move z squared\n\n* Add Domain struct\n\n* clippy and change argument order\n\n* move domain to lib\n\n---------\n\nCo-authored-by: Sergio Chouhy <sergio.chouhy@gmail.com>",
          "timestamp": "2023-05-02T16:40:18Z",
          "tree_id": "a22a34d4220920eba13ee7adb6ecee1083600a1c",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d3e0eb9d72b3ee5979f3da9cf71a119bd3492add"
        },
        "date": 1683058072621,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2553993427,
            "range": "± 3524162",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4080084274,
            "range": "± 41478114",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5350467191,
            "range": "± 1823177",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9082886792,
            "range": "± 33447006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 11167963737,
            "range": "± 1618705",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19639343480,
            "range": "± 40545278",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 23293200290,
            "range": "± 4384223",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 41516456352,
            "range": "± 115557414",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1297065184,
            "range": "± 522078",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2721718621,
            "range": "± 854135",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5692903756,
            "range": "± 1770065",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11892651748,
            "range": "± 3761486",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 59590129,
            "range": "± 1128359",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 140755719,
            "range": "± 714903",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 283412467,
            "range": "± 1875549",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 583478827,
            "range": "± 4298678",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3891005740,
            "range": "± 2852601",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 8161933319,
            "range": "± 6507574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 17030118834,
            "range": "± 4836336",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 35520756883,
            "range": "± 10544893",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 59031542048,
            "range": "± 196087339",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 118238503831,
            "range": "± 22814560",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 236999240511,
            "range": "± 448154430",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 477045559461,
            "range": "± 1582201124",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 512,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 15647,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 416,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 686,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 397,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 399,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 34015335,
            "range": "± 9504",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 47548824,
            "range": "± 12899",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 275891,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 34036959,
            "range": "± 10763",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43053772+diegokingston@users.noreply.github.com",
            "name": "Diego K",
            "username": "diegokingston"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f707f7d1107d8f9b7772a29a71ee37070731f7bb",
          "message": "Update References.md (#285)",
          "timestamp": "2023-05-02T19:11:51Z",
          "tree_id": "8a35db14604d98c8ff00d4448795cdade07c386a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f707f7d1107d8f9b7772a29a71ee37070731f7bb"
        },
        "date": 1683067097934,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2361191574,
            "range": "± 5782430",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4682376616,
            "range": "± 13840984",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4936955046,
            "range": "± 2686135",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10124519648,
            "range": "± 34890804",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 10310839500,
            "range": "± 5437455",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21731336008,
            "range": "± 22927646",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 21499438746,
            "range": "± 8289660",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 46201598133,
            "range": "± 82489037",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1148570036,
            "range": "± 551892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2409474725,
            "range": "± 234512",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5045863226,
            "range": "± 1015029",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10543475402,
            "range": "± 6322522",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 74777853,
            "range": "± 977923",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 158209516,
            "range": "± 1000007",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 317223623,
            "range": "± 380851",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 646992162,
            "range": "± 4204014",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3562902188,
            "range": "± 1579506",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7448601691,
            "range": "± 2124529",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 15558779867,
            "range": "± 6558281",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 32455313681,
            "range": "± 9120515",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 58850385396,
            "range": "± 9672151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 118070588103,
            "range": "± 29044391",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 236388195857,
            "range": "± 12911789",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 474420724870,
            "range": "± 117294740",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 84,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 347,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 204,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 33979525,
            "range": "± 9758",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 47593995,
            "range": "± 23890",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 317824,
            "range": "± 140",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 34159977,
            "range": "± 9061",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c611e65d9717b7e5c47f5b6f662e65ede76e606d",
          "message": "Sqrt (#290)\n\n* Initial version of sqrt\n\n* Remove graph\n\n* Formatting, fix non_qr\n\n* Fixed sqrt()\n\n* Clippy and nits\n\n* Changed test name\n\n* Fix\n\n* Remove unused file\n\n* Remove unused random fn\n\n* Reverted change that should be in another pr\n\n* Cargo fmt\n\n* Add test for more curves\n\n* Remove comments\n\n* Remove unused trait requirement\n\n* Add test\n\n* Renamed tests\n\n* Improve documentation\n\n* Add test for 0 in sqrt\n\n* fix linter\n\n* Add Legendre Symbol enum\n\n* Cargo fmt\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>",
          "timestamp": "2023-05-02T19:09:05Z",
          "tree_id": "47bd83fe09c2f3a703125da74a3949348f3902f7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c611e65d9717b7e5c47f5b6f662e65ede76e606d"
        },
        "date": 1683068169847,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2593379617,
            "range": "± 41445901",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 5201203978,
            "range": "± 68610955",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5578376187,
            "range": "± 123060574",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 11349558910,
            "range": "± 55503722",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 11172690169,
            "range": "± 93027080",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 24480043189,
            "range": "± 100698191",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 23253622548,
            "range": "± 189743160",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 51660082955,
            "range": "± 246525217",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1266692216,
            "range": "± 12005694",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2640027645,
            "range": "± 56614049",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5628871169,
            "range": "± 35619561",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11600819846,
            "range": "± 78621347",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 80379510,
            "range": "± 1170101",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 172055857,
            "range": "± 1353695",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 342924212,
            "range": "± 1477619",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 689675547,
            "range": "± 3543856",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3900546780,
            "range": "± 42753210",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 8422460138,
            "range": "± 218147675",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 17077250494,
            "range": "± 212719109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 35750975589,
            "range": "± 429852368",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 64906284307,
            "range": "± 563875531",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 129401637602,
            "range": "± 1313796981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 260593842991,
            "range": "± 2891539884",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 520550772508,
            "range": "± 3843900755",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 197,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1589,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 261,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 466,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 302,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 295,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 37827260,
            "range": "± 963442",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 52593567,
            "range": "± 1325391",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 367271,
            "range": "± 5205",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 39637970,
            "range": "± 516893",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "41742639+schouhy@users.noreply.github.com",
            "name": "Sergio Chouhy",
            "username": "schouhy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "89aeeb43b8454e64f6aa41f14d0a0c8bde1e5cdb",
          "message": "Add steps to stark verifier (#289)\n\n* move z challenge to step 0\n\n* incorporate domain\n\n* add boundary and transitions coefficientes to challenges\n\n* move code to step 2\n\n* move fri challenges to step 1\n\n* move fri check to step 3\n\n* move deep composition polynomial check to own step 4 function\n\n* sort function parameters\n\n* fix comments\n\n* fix signature\n\n---------\n\nCo-authored-by: MauroFab <maurotoscano2@gmail.com>",
          "timestamp": "2023-05-02T19:33:32Z",
          "tree_id": "4d5ed36a4c4bdbb60b2cb5ebacde1918a730b291",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/89aeeb43b8454e64f6aa41f14d0a0c8bde1e5cdb"
        },
        "date": 1683069575624,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2623156721,
            "range": "± 47948691",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 5504521217,
            "range": "± 103031879",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5832857800,
            "range": "± 96038738",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 11896636878,
            "range": "± 178866458",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 11390326059,
            "range": "± 357465984",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 25427801544,
            "range": "± 453392692",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 24521608831,
            "range": "± 298130501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 51303483379,
            "range": "± 496997833",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1277125665,
            "range": "± 20790205",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2667652311,
            "range": "± 19758889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5939965787,
            "range": "± 202168828",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11898482569,
            "range": "± 422791234",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 91647704,
            "range": "± 2134191",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 196720217,
            "range": "± 2841167",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 395611961,
            "range": "± 7181238",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 801337797,
            "range": "± 23288241",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3937404645,
            "range": "± 78564031",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 8477014607,
            "range": "± 245842713",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 17363374744,
            "range": "± 483270381",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 36487175081,
            "range": "± 1031247407",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 64958258516,
            "range": "± 1426528921",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 131332160615,
            "range": "± 2255058864",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 254705669307,
            "range": "± 4183644284",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 509884816703,
            "range": "± 1419441327",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 817,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 25646,
            "range": "± 1512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 724,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 29,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1028,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 834,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 748,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 36827236,
            "range": "± 608266",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 51414516,
            "range": "± 854084",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 320430,
            "range": "± 3411",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 37282090,
            "range": "± 681161",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "martin.paulucci@lambdaclass.com",
            "name": "Martin Paulucci",
            "username": "mpaulucci"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d1f48eb2a44f3683b62242101825f0187c595cb7",
          "message": "Add support to dealing with multiple fields in metal (#284)\n\n* Add support for more than one field.\n\n* Remove hardcoded stark256.\n\n* Remove `field_supports_metal` hack.\n\n* Move fields around.\n\n* Add comment on stark fields.\n\n* Update math/src/field/traits.rs\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>",
          "timestamp": "2023-05-02T20:32:58Z",
          "tree_id": "e72fb9607a109bffbd3fdc784efc77f102fb1327",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d1f48eb2a44f3683b62242101825f0187c595cb7"
        },
        "date": 1683072024372,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2373652996,
            "range": "± 1872551",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4909982131,
            "range": "± 18657212",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4963300284,
            "range": "± 4651857",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10579017022,
            "range": "± 30640536",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 10357641137,
            "range": "± 8656353",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 22716657582,
            "range": "± 77561198",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 21595695092,
            "range": "± 19062749",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 48339123347,
            "range": "± 51228765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1153319145,
            "range": "± 233931",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2419054136,
            "range": "± 375606",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5066370713,
            "range": "± 519215",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10583560107,
            "range": "± 3405800",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 82668874,
            "range": "± 295873",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 163938808,
            "range": "± 980504",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 330495768,
            "range": "± 1991998",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 670766004,
            "range": "± 6797642",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3575612024,
            "range": "± 2302655",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7477007191,
            "range": "± 6000456",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 15612895481,
            "range": "± 5630259",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 32540807809,
            "range": "± 20222021",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 58890834749,
            "range": "± 8345261",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 118146641196,
            "range": "± 16918357",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 236418703825,
            "range": "± 128283235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 474238018434,
            "range": "± 43317633",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 34,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 33949041,
            "range": "± 22799",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 47618775,
            "range": "± 14786",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 317252,
            "range": "± 191",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 34123357,
            "range": "± 3193",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "estefano.bargas@fing.edu.uy",
            "name": "Estéfano Bargas",
            "username": "xqft"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "09cd5886aeb1e3c4a0f900f25a800970beffdff7",
          "message": "Redesigned the FFT polynomial API for better integration with the STARK prover (#299)\n\n* Added parameter for choosing amount of evals\n\n* Changed order to blowup factor\n\n* Added interpol, fixed tests, nits\n\n* Added InputError\n\n* Improve tests\n\n* Fix tests and metal interpol\n\n* Fix tests\n\n* Fix benchs\n\n* Added CPU fallback warnings\n\n* Fix linting",
          "timestamp": "2023-05-05T19:59:19Z",
          "tree_id": "c7e735843414b27b1c630e6924f15d8c5f5839b7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/09cd5886aeb1e3c4a0f900f25a800970beffdff7"
        },
        "date": 1683317116711,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 129007189,
            "range": "± 1545186",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 254350757,
            "range": "± 3342038",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 494913677,
            "range": "± 511054",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 973709208,
            "range": "± 5560490",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34752679,
            "range": "± 606769",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69539944,
            "range": "± 488297",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132358810,
            "range": "± 1005235",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276281051,
            "range": "± 3625778",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31426381,
            "range": "± 177323",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59894365,
            "range": "± 411288",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120830990,
            "range": "± 3902152",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 258799573,
            "range": "± 21730788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 165975489,
            "range": "± 559647",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330426416,
            "range": "± 2084103",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 649373604,
            "range": "± 3998820",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1308726625,
            "range": "± 16518379",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 457407739,
            "range": "± 1826107",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 916193771,
            "range": "± 3919678",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1791414229,
            "range": "± 15950380",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3542710770,
            "range": "± 69309208",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ajgarassino@gmail.com",
            "name": "ajgara",
            "username": "ajgara"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "183d9e4fddc45b3c5ee8073d46ba9dd24bbca218",
          "message": "When one element is zero, the pairing returns one (#316)",
          "timestamp": "2023-05-08T15:17:40Z",
          "tree_id": "79fbbb827711e1a78df0e294a7f5137f43f1f330",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/183d9e4fddc45b3c5ee8073d46ba9dd24bbca218"
        },
        "date": 1683559417351,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 129190500,
            "range": "± 1034330",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255920069,
            "range": "± 3503253",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 492799114,
            "range": "± 755866",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 977396583,
            "range": "± 11397986",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33846328,
            "range": "± 410032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69839996,
            "range": "± 403401",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136253203,
            "range": "± 4233848",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 292208979,
            "range": "± 8210736",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29582719,
            "range": "± 903115",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58301732,
            "range": "± 1874281",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 123567619,
            "range": "± 5583816",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 255795958,
            "range": "± 19004742",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 167409647,
            "range": "± 636239",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330376291,
            "range": "± 1455008",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 652416166,
            "range": "± 5814340",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1317082479,
            "range": "± 17387488",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 458170864,
            "range": "± 671644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 905250125,
            "range": "± 6699319",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1803334333,
            "range": "± 4409215",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3540600812,
            "range": "± 10167467",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1f143cd2aca56202d3f9b49efa6bdff1cb5482e9",
          "message": "Add default Fr type for bls 12 381 (#318)\n\n* Add default types\n\n* Update mod\n\n* Remove hardcoded values\n\n* Fmt\n\n* Fix remove harcoded values",
          "timestamp": "2023-05-08T19:32:50Z",
          "tree_id": "d998de6ea3ed2eaa90847d85c8b2246e8376b6eb",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1f143cd2aca56202d3f9b49efa6bdff1cb5482e9"
        },
        "date": 1683574714254,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 129351613,
            "range": "± 1796426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248985889,
            "range": "± 3440444",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 524255437,
            "range": "± 14875134",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 966965395,
            "range": "± 15211951",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33855403,
            "range": "± 430242",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 75892364,
            "range": "± 2156057",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 134873855,
            "range": "± 2401968",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276842666,
            "range": "± 4028533",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 28106990,
            "range": "± 774934",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59107005,
            "range": "± 2640721",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 123143697,
            "range": "± 4269484",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 266968916,
            "range": "± 12843255",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 167768822,
            "range": "± 545225",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329742812,
            "range": "± 1322110",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 652199979,
            "range": "± 6056441",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1304653458,
            "range": "± 12032203",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 452218093,
            "range": "± 5289816",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 909460625,
            "range": "± 16674574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1798307687,
            "range": "± 38383889",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3501626166,
            "range": "± 19832610",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}