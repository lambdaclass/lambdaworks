window.BENCHMARK_DATA = {
  "lastUpdate": 1683058074658,
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
      }
    ]
  }
}