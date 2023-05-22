window.BENCHMARK_DATA = {
  "lastUpdate": 1684784871382,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
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
          "id": "e52aecd69780a6ceb9d9dd8df6d48715c1a782c6",
          "message": "Synchronize code and protocol (#295)\n\n* add trace commitments to transcript\n\n* move sampling of z to round 3\n\n* add batch_commit function\n\n* add commitments of the composition polynomial to the transcript\n\n* refactor sampling of boundary and transition coefficients\n\n* add ood evaluations to transcript\n\n* minor refactor\n\n* move sample batch to lib\n\n* extract deep composition randomness to round 4\n\n* refactor fri commitment phase\n\n* refactor next_fri_layer\n\n* remove last iteration of fri commit phase\n\n* refactor fri_commit_phase\n\n* move sampling of q_0 to query phase. Rename\n\n* refactor fri decommitment\n\n* add fri last value to proof and remove it from decommitments\n\n* remove layers commitments from decommitment\n\n* remove unused FriQuery struct\n\n* leave only symmetric points in the proof\n\n* remove unnecesary last fri layer\n\n* reuse composition poly evaluations from round_1 in the consistency check\n\n* minor refactor\n\n* fix trace ood commitments and small refactor\n\n* move fri_query_phase to fri mod\n\n* minor refactor in build deeep composition poly function\n\n* refactor deep composition poly related code\n\n* minor modifications to comments\n\n* clippy\n\n* add comments\n\n* move iota sampling to step 0 and rename challenges\n\n* minor refactor and add missing opening checks\n\n* minor refactor\n\n* move transition divisor computation outside of the main loop of the computation of the composition polynomial\n\n* add protocol to docs\n\n* clippy and fmt\n\n* remove old test\n\n* fix typo\n\n* fmt\n\n* remove unnecessary public attribute\n\n* move build_execution trace to round 1\n\n* add rap support to air\n\n* remove commit_original_trace function\n\n* Add auxiliary trace polynomials and commitments in the first round\n\n* Test simple fibonacci\n\n* fix format in starks protocol docs\n\n* fmt, clippy\n\n* remove comments, add sampling of cairo rap\n\n* add commented test\n\n* remove old test\n\n* Fix transition degrees and transition contraint in fibonacci rap test\n\n* Add debug assertion directive in test that uses debug module so that compilation in release mode does not error\n\n* Add inline never lint to bench functions\n\n* Remove Fibonacci17 iai benchmark\n\n* Add unused code lint\n\n* Remove QuadraticAIR iai benchmark\n\n* Remove newlines\n\n---------\n\nCo-authored-by: ajgarassino <ajgarassino@gmail.com>\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>",
          "timestamp": "2023-05-12T15:03:19Z",
          "tree_id": "c3e1417d8aac326f635ad54916dfe71a3cbf2e47",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e52aecd69780a6ceb9d9dd8df6d48715c1a782c6"
        },
        "date": 1683904149315,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131679736,
            "range": "± 3042039",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248445208,
            "range": "± 1239039",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 493451292,
            "range": "± 1276070",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 995264229,
            "range": "± 18123326",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34183126,
            "range": "± 379909",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69672603,
            "range": "± 444288",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133138050,
            "range": "± 1241119",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275558698,
            "range": "± 2168052",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31173579,
            "range": "± 188591",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58768561,
            "range": "± 754510",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120711756,
            "range": "± 5083731",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 253777583,
            "range": "± 20307224",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 167203222,
            "range": "± 392214",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 332167906,
            "range": "± 1551310",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 654800458,
            "range": "± 4356596",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1333594416,
            "range": "± 19034302",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 459042229,
            "range": "± 960082",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 904817333,
            "range": "± 6330288",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1802371958,
            "range": "± 6122711",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3544186541,
            "range": "± 26277379",
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
          "id": "e52aecd69780a6ceb9d9dd8df6d48715c1a782c6",
          "message": "Synchronize code and protocol (#295)\n\n* add trace commitments to transcript\n\n* move sampling of z to round 3\n\n* add batch_commit function\n\n* add commitments of the composition polynomial to the transcript\n\n* refactor sampling of boundary and transition coefficients\n\n* add ood evaluations to transcript\n\n* minor refactor\n\n* move sample batch to lib\n\n* extract deep composition randomness to round 4\n\n* refactor fri commitment phase\n\n* refactor next_fri_layer\n\n* remove last iteration of fri commit phase\n\n* refactor fri_commit_phase\n\n* move sampling of q_0 to query phase. Rename\n\n* refactor fri decommitment\n\n* add fri last value to proof and remove it from decommitments\n\n* remove layers commitments from decommitment\n\n* remove unused FriQuery struct\n\n* leave only symmetric points in the proof\n\n* remove unnecesary last fri layer\n\n* reuse composition poly evaluations from round_1 in the consistency check\n\n* minor refactor\n\n* fix trace ood commitments and small refactor\n\n* move fri_query_phase to fri mod\n\n* minor refactor in build deeep composition poly function\n\n* refactor deep composition poly related code\n\n* minor modifications to comments\n\n* clippy\n\n* add comments\n\n* move iota sampling to step 0 and rename challenges\n\n* minor refactor and add missing opening checks\n\n* minor refactor\n\n* move transition divisor computation outside of the main loop of the computation of the composition polynomial\n\n* add protocol to docs\n\n* clippy and fmt\n\n* remove old test\n\n* fix typo\n\n* fmt\n\n* remove unnecessary public attribute\n\n* move build_execution trace to round 1\n\n* add rap support to air\n\n* remove commit_original_trace function\n\n* Add auxiliary trace polynomials and commitments in the first round\n\n* Test simple fibonacci\n\n* fix format in starks protocol docs\n\n* fmt, clippy\n\n* remove comments, add sampling of cairo rap\n\n* add commented test\n\n* remove old test\n\n* Fix transition degrees and transition contraint in fibonacci rap test\n\n* Add debug assertion directive in test that uses debug module so that compilation in release mode does not error\n\n* Add inline never lint to bench functions\n\n* Remove Fibonacci17 iai benchmark\n\n* Add unused code lint\n\n* Remove QuadraticAIR iai benchmark\n\n* Remove newlines\n\n---------\n\nCo-authored-by: ajgarassino <ajgarassino@gmail.com>\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>",
          "timestamp": "2023-05-12T15:03:19Z",
          "tree_id": "c3e1417d8aac326f635ad54916dfe71a3cbf2e47",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e52aecd69780a6ceb9d9dd8df6d48715c1a782c6"
        },
        "date": 1683916285316,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2258796046,
            "range": "± 1712073",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3991302724,
            "range": "± 96795409",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4722351568,
            "range": "± 6176709",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8892125592,
            "range": "± 38888056",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 9857117646,
            "range": "± 6402851",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19456750079,
            "range": "± 136738534",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 20525834933,
            "range": "± 8873146",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 41082611680,
            "range": "± 248675365",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1299610343,
            "range": "± 878016",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2727854241,
            "range": "± 995948",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5711825562,
            "range": "± 1765491",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11938801729,
            "range": "± 6237739",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 75447967,
            "range": "± 2657354",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 153219524,
            "range": "± 3160211",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 299801619,
            "range": "± 2323103",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 599881863,
            "range": "± 6681819",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3577524050,
            "range": "± 1969339",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7485917464,
            "range": "± 3428550",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 15632502642,
            "range": "± 21302381",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 32608461151,
            "range": "± 18835678",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 21816496779,
            "range": "± 20801293",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 43955314092,
            "range": "± 43586607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 88812751062,
            "range": "± 28173165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 178904341330,
            "range": "± 1335983411",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 511,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 16318,
            "range": "± 114",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 388,
            "range": "± 19",
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
            "value": 678,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 538,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 397,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 6762536,
            "range": "± 1513",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 2023249556,
            "range": "± 2380731",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 11192398668,
            "range": "± 44153833",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 70445937868,
            "range": "± 182322907",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 488053484238,
            "range": "± 2117803953",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 2645468,
            "range": "± 892",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47506558+MegaRedHand@users.noreply.github.com",
            "name": "Tomás",
            "username": "MegaRedHand"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d4c9a8d0f411ac76de45a44ab404d7c02d5a9bb5",
          "message": "Implement MSM on CPU using Pippenger's algorithm (#326)\n\n* Add pippenger CPU implementation\n\n* Fix pippenger\n\nWas indexing the `UnsignedInteger` limbs as little endian, but they are in big endian order.\n\n* Add constants for proptests\n\n* Clean up the generic params of `pippenger::msm`\n\n* Add wrapper around msm that fixes the window size\n\n* Add benchmarks\n\n* Change benchmark sizes to powers of two\n\n* Approximate optimal window size with adhoc formula\n\n* Fix heuristic\n\nWas using trailing zeros so it worked correctly only on powers of two.\nChanged it to use leading zeros and it should work now.\n\n* Add comments in CPU pippenger impl\n\n* Change `debug_assert_eq` -> `if { return Err }`",
          "timestamp": "2023-05-17T14:01:43Z",
          "tree_id": "655c702fbbfba0c9afa8bcec465f77e408cf22fd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d4c9a8d0f411ac76de45a44ab404d7c02d5a9bb5"
        },
        "date": 1684332453164,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128613158,
            "range": "± 5316181",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 247639965,
            "range": "± 4933255",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 493385749,
            "range": "± 6433639",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 975147708,
            "range": "± 6689792",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34262006,
            "range": "± 298622",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67707952,
            "range": "± 1161106",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132615763,
            "range": "± 984105",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276657239,
            "range": "± 2296360",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31265388,
            "range": "± 229256",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59589831,
            "range": "± 671456",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 118656599,
            "range": "± 1794247",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 259840635,
            "range": "± 23602903",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 167326067,
            "range": "± 858062",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330861552,
            "range": "± 1044185",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 652660687,
            "range": "± 5459267",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1318444708,
            "range": "± 12006501",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 457369916,
            "range": "± 2869088",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 913759249,
            "range": "± 7141833",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1800668854,
            "range": "± 6713068",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3539439333,
            "range": "± 16783338",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47506558+MegaRedHand@users.noreply.github.com",
            "name": "Tomás",
            "username": "MegaRedHand"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d4c9a8d0f411ac76de45a44ab404d7c02d5a9bb5",
          "message": "Implement MSM on CPU using Pippenger's algorithm (#326)\n\n* Add pippenger CPU implementation\n\n* Fix pippenger\n\nWas indexing the `UnsignedInteger` limbs as little endian, but they are in big endian order.\n\n* Add constants for proptests\n\n* Clean up the generic params of `pippenger::msm`\n\n* Add wrapper around msm that fixes the window size\n\n* Add benchmarks\n\n* Change benchmark sizes to powers of two\n\n* Approximate optimal window size with adhoc formula\n\n* Fix heuristic\n\nWas using trailing zeros so it worked correctly only on powers of two.\nChanged it to use leading zeros and it should work now.\n\n* Add comments in CPU pippenger impl\n\n* Change `debug_assert_eq` -> `if { return Err }`",
          "timestamp": "2023-05-17T14:01:43Z",
          "tree_id": "655c702fbbfba0c9afa8bcec465f77e408cf22fd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d4c9a8d0f411ac76de45a44ab404d7c02d5a9bb5"
        },
        "date": 1684343945869,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2024908615,
            "range": "± 2265296",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4734605438,
            "range": "± 11120446",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4233374360,
            "range": "± 2732918",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10278370337,
            "range": "± 26884555",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 8838779166,
            "range": "± 6897942",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 22013317353,
            "range": "± 43942340",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 18422145484,
            "range": "± 13390297",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 46681065641,
            "range": "± 106857146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1152041230,
            "range": "± 204032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2416299769,
            "range": "± 200737",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5059936546,
            "range": "± 500484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10571201735,
            "range": "± 6049633",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 82536020,
            "range": "± 861029",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 181158922,
            "range": "± 1058389",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 363376936,
            "range": "± 1574358",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 739346148,
            "range": "± 2598892",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3200713796,
            "range": "± 1580987",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 6704193032,
            "range": "± 2330920",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 13992516788,
            "range": "± 8268693",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 29192715836,
            "range": "± 9862765",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 21667990375,
            "range": "± 2395408",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 43634966394,
            "range": "± 4420370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 87848141818,
            "range": "± 11369846",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 176875439508,
            "range": "± 15091012",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 83,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 347,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 120,
            "range": "± 16",
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
            "value": 201,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 157,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 156,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 6817599,
            "range": "± 769",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 1945311443,
            "range": "± 2416573",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 10577078746,
            "range": "± 29993360",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 65066906594,
            "range": "± 115239414",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 441901508742,
            "range": "± 1794808204",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 2663078,
            "range": "± 671",
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
          "id": "be84d1e9abeebfc1a42b8a4d7a2744c290a1e344",
          "message": "Update References.md (#345)",
          "timestamp": "2023-05-19T18:18:43Z",
          "tree_id": "835c959656cc06c0f1dca20b1393a1ded100668e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/be84d1e9abeebfc1a42b8a4d7a2744c290a1e344"
        },
        "date": 1684520688075,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 129689337,
            "range": "± 2179016",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248550375,
            "range": "± 969200",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 482256312,
            "range": "± 4659663",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 973545792,
            "range": "± 6476567",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34073974,
            "range": "± 276824",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67795947,
            "range": "± 1512678",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132834938,
            "range": "± 993431",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276103781,
            "range": "± 3875416",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30915956,
            "range": "± 289198",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58778688,
            "range": "± 380818",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122393914,
            "range": "± 5966225",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 264913416,
            "range": "± 15623757",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 166844340,
            "range": "± 474665",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 331510635,
            "range": "± 1113029",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 652896437,
            "range": "± 5219211",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1318604333,
            "range": "± 14045968",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 458843708,
            "range": "± 2065529",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 911959875,
            "range": "± 6304440",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1802150979,
            "range": "± 4545730",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3516487083,
            "range": "± 15513689",
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
          "id": "be84d1e9abeebfc1a42b8a4d7a2744c290a1e344",
          "message": "Update References.md (#345)",
          "timestamp": "2023-05-19T18:18:43Z",
          "tree_id": "835c959656cc06c0f1dca20b1393a1ded100668e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/be84d1e9abeebfc1a42b8a4d7a2744c290a1e344"
        },
        "date": 1684534648146,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2418464860,
            "range": "± 2092346",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 5235851042,
            "range": "± 17608776",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5048346945,
            "range": "± 9226589",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 11319019667,
            "range": "± 57972365",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 10532573462,
            "range": "± 11408792",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 24063479534,
            "range": "± 81052425",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 21930053527,
            "range": "± 16052839",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 51319686968,
            "range": "± 139633606",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1382365923,
            "range": "± 1073268",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2889822829,
            "range": "± 9875704",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 6063873562,
            "range": "± 5052800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 12679110654,
            "range": "± 6859551",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 84794178,
            "range": "± 1333046",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 176618339,
            "range": "± 4301404",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 355961947,
            "range": "± 5376221",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 718351218,
            "range": "± 3579082",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3822681099,
            "range": "± 3089048",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7993040909,
            "range": "± 8651869",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 16708322504,
            "range": "± 3547379",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 34796855874,
            "range": "± 18995158",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 25969511658,
            "range": "± 105435527",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 52298705340,
            "range": "± 203590364",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 105385694180,
            "range": "± 236887845",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 212119412470,
            "range": "± 248091771",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 47,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 115,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 118,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 210,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 162,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 162,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 8176066,
            "range": "± 21503",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 2364973167,
            "range": "± 5044025",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 13027366964,
            "range": "± 15439494",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 80521892816,
            "range": "± 131046593",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 550095807928,
            "range": "± 1271183490",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 3201886,
            "range": "± 1048",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47506558+MegaRedHand@users.noreply.github.com",
            "name": "Tomás",
            "username": "MegaRedHand"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4a6d295beafcaa66e1212b28e601c57ae8fc1c96",
          "message": "Add parallel Pippenger's on CPU via rayon (#336)\n\n* Add pippenger CPU implementation\n\n* Fix pippenger\n\nWas indexing the `UnsignedInteger` limbs as little endian, but they are in big endian order.\n\n* Add constants for proptests\n\n* Clean up the generic params of `pippenger::msm`\n\n* Add wrapper around msm that fixes the window size\n\n* Add benchmarks\n\n* Change benchmark sizes to powers of two\n\n* Approximate optimal window size with adhoc formula\n\n* Fix heuristic\n\nWas using trailing zeros so it worked correctly only on powers of two.\nChanged it to use leading zeros and it should work now.\n\n* Add comments in CPU pippenger impl\n\n* Add CPU parallel pippenger via rayon (+benchmarks)\n\n* Fix parallel pippenger\n\n* Add window size cap\n\n* Extract heuristic to helper function\n\n* Appease clippy\n\n* Rename `hidings` -> `points`\n\n* Add comment regarding an optimization",
          "timestamp": "2023-05-19T22:27:10Z",
          "tree_id": "fbf895d46d2a55ec6adfe14df678bada04938454",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4a6d295beafcaa66e1212b28e601c57ae8fc1c96"
        },
        "date": 1684535586812,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131141055,
            "range": "± 2630663",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 246372968,
            "range": "± 2483128",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 482569031,
            "range": "± 3744618",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 978989208,
            "range": "± 8331834",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34197170,
            "range": "± 326748",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67823985,
            "range": "± 1486646",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132513586,
            "range": "± 301164",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276784583,
            "range": "± 3197880",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30840647,
            "range": "± 318896",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58541963,
            "range": "± 928676",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122135678,
            "range": "± 3277087",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 271585125,
            "range": "± 12545116",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 165297777,
            "range": "± 431701",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330474156,
            "range": "± 1247542",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 649627833,
            "range": "± 2045633",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1318900020,
            "range": "± 11110204",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 462029270,
            "range": "± 2308183",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 919071812,
            "range": "± 2626648",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1803747562,
            "range": "± 5266875",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3541137125,
            "range": "± 12326001",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47506558+MegaRedHand@users.noreply.github.com",
            "name": "Tomás",
            "username": "MegaRedHand"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4a6d295beafcaa66e1212b28e601c57ae8fc1c96",
          "message": "Add parallel Pippenger's on CPU via rayon (#336)\n\n* Add pippenger CPU implementation\n\n* Fix pippenger\n\nWas indexing the `UnsignedInteger` limbs as little endian, but they are in big endian order.\n\n* Add constants for proptests\n\n* Clean up the generic params of `pippenger::msm`\n\n* Add wrapper around msm that fixes the window size\n\n* Add benchmarks\n\n* Change benchmark sizes to powers of two\n\n* Approximate optimal window size with adhoc formula\n\n* Fix heuristic\n\nWas using trailing zeros so it worked correctly only on powers of two.\nChanged it to use leading zeros and it should work now.\n\n* Add comments in CPU pippenger impl\n\n* Add CPU parallel pippenger via rayon (+benchmarks)\n\n* Fix parallel pippenger\n\n* Add window size cap\n\n* Extract heuristic to helper function\n\n* Appease clippy\n\n* Rename `hidings` -> `points`\n\n* Add comment regarding an optimization",
          "timestamp": "2023-05-19T22:27:10Z",
          "tree_id": "fbf895d46d2a55ec6adfe14df678bada04938454",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4a6d295beafcaa66e1212b28e601c57ae8fc1c96"
        },
        "date": 1684549306649,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2403038280,
            "range": "± 8629017",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 5172220692,
            "range": "± 23501246",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4943606646,
            "range": "± 23674996",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 11220853488,
            "range": "± 22226232",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 10288816953,
            "range": "± 64150034",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 24052998394,
            "range": "± 30185049",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 21586650678,
            "range": "± 186977427",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 51422753626,
            "range": "± 108906556",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1349164700,
            "range": "± 4667718",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2842477724,
            "range": "± 11751582",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5936623788,
            "range": "± 7603321",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 12380243636,
            "range": "± 51422601",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 81543910,
            "range": "± 839325",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 173038132,
            "range": "± 1005020",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 352401879,
            "range": "± 806111",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 721925294,
            "range": "± 2221410",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3750897124,
            "range": "± 25113979",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7815757119,
            "range": "± 68633467",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 16421940198,
            "range": "± 159042180",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 34752928527,
            "range": "± 12300657",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 25944603260,
            "range": "± 8413739",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 52235062420,
            "range": "± 23984884",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 105036303023,
            "range": "± 442599933",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 209530667257,
            "range": "± 1265305375",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 114,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 118,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 213,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 162,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 162,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 8129004,
            "range": "± 36244",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 2348135041,
            "range": "± 6857547",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 12871499555,
            "range": "± 30159710",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 78779917036,
            "range": "± 436461436",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 537419409709,
            "range": "± 5036575418",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 3082006,
            "range": "± 24174",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8aa74416432c663495b2269180992fc6a3d0a311",
          "message": "perf: optimize polynomial division (#357)\n\nComputing a the inverse in `Fp` is the most expensive step in `Fp`\ndivision, which in turn takes most of the time in the polynomial\ndivision itself.\nExtracting this out of the hot loop and replacing the call site with a\nproduct gives a huge boost (70-90% measured in polynomial benchmarks and\nalmost 10% in STARK proving benchmarks).",
          "timestamp": "2023-05-22T12:06:43Z",
          "tree_id": "8ea0780d008ea84646dd464cbb951dd6c168ae1b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8aa74416432c663495b2269180992fc6a3d0a311"
        },
        "date": 1684757556273,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130848772,
            "range": "± 3569107",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 245623347,
            "range": "± 1964950",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 487049239,
            "range": "± 5692263",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 982930125,
            "range": "± 5460311",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34097800,
            "range": "± 226211",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67698892,
            "range": "± 1072715",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132903760,
            "range": "± 645909",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275539343,
            "range": "± 2383417",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31107713,
            "range": "± 187068",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59358460,
            "range": "± 640768",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 123772501,
            "range": "± 5885087",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 272452999,
            "range": "± 19686977",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 165377231,
            "range": "± 414661",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 335540604,
            "range": "± 1634135",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 658908979,
            "range": "± 3063061",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1321042916,
            "range": "± 19550412",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 463300364,
            "range": "± 3769769",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 911698521,
            "range": "± 3276563",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1802445354,
            "range": "± 7087381",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3543734104,
            "range": "± 10392037",
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
          "id": "509167c53e54529348beaba144e69a1f02e0d1a8",
          "message": "Add kzg tests (#328)\n\n* Tests\n\n* Fix tests\n\n* Add tests\n\n* Add tests\n\n* Add tests\n\n* Clippy\n\n* Remove unnecesary type definition\n\n* Delete constraint_system.rs",
          "timestamp": "2023-05-22T12:43:20Z",
          "tree_id": "119b1b38b2faad14e078ceb78022c178fe809db6",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/509167c53e54529348beaba144e69a1f02e0d1a8"
        },
        "date": 1684759753189,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131576382,
            "range": "± 3361659",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248993749,
            "range": "± 2691132",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 490154354,
            "range": "± 6724315",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 982628645,
            "range": "± 10546600",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34119499,
            "range": "± 228005",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67986201,
            "range": "± 850163",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133053491,
            "range": "± 845911",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276508229,
            "range": "± 3744068",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 28541451,
            "range": "± 2004283",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59529454,
            "range": "± 641680",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 117788791,
            "range": "± 2045195",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 244792513,
            "range": "± 26669071",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 174007345,
            "range": "± 5485753",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330873094,
            "range": "± 9412498",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 636934083,
            "range": "± 8576956",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1309030208,
            "range": "± 24746665",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 452008020,
            "range": "± 2118109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 905353208,
            "range": "± 5676972",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1795655500,
            "range": "± 12061366",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3537671895,
            "range": "± 36261274",
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
          "id": "cadd83c23e273ee40a13887e1c63b3716e75a834",
          "message": "Implemented parallel FFT on CUDA (#268)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Remove commented code\n\n* Remove allow(dead_code)\n\n* Update comment about limb size\n\n* Update link and change to permalink\n\n* Add Oppen's operator>> bithack\n\n* Fix comment\n\n* Revert \"Fix comment\"\n\nThis reverts commit 7e6ce71571a1f6e0dbd0dff91213db91a0be36b1.\n\n* Remove empty tests module\n\nThis module is tested via `Polynomial::*_fft` in `fft/src/polynomial.rs`\n\n* Use longlong instead of long, and fix u128 mul.\n\n* Address \"reduce branch conditions\" TODOs\n\n---------\n\nCo-authored-by: Tomás <tomas.gruner@lambdaclass.com>\nCo-authored-by: Tomás <47506558+MegaRedHand@users.noreply.github.com>\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>",
          "timestamp": "2023-05-22T14:33:54Z",
          "tree_id": "16128b3842acefefa9858db3638c94203966eadd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/cadd83c23e273ee40a13887e1c63b3716e75a834"
        },
        "date": 1684766382662,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130559831,
            "range": "± 3803685",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 249579562,
            "range": "± 6527772",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 489426895,
            "range": "± 6916995",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 981324354,
            "range": "± 7321120",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33766223,
            "range": "± 276932",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69507910,
            "range": "± 806242",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132610797,
            "range": "± 387520",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275884906,
            "range": "± 2632541",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31064510,
            "range": "± 280034",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58777022,
            "range": "± 1086930",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120599259,
            "range": "± 3599679",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 256904291,
            "range": "± 17248191",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 165259391,
            "range": "± 505192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 334188593,
            "range": "± 1381187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 659084187,
            "range": "± 2717065",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1319905208,
            "range": "± 13904786",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 457222896,
            "range": "± 987372",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 920615771,
            "range": "± 4460929",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1804814459,
            "range": "± 3551454",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3499840208,
            "range": "± 26999446",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8aa74416432c663495b2269180992fc6a3d0a311",
          "message": "perf: optimize polynomial division (#357)\n\nComputing a the inverse in `Fp` is the most expensive step in `Fp`\ndivision, which in turn takes most of the time in the polynomial\ndivision itself.\nExtracting this out of the hot loop and replacing the call site with a\nproduct gives a huge boost (70-90% measured in polynomial benchmarks and\nalmost 10% in STARK proving benchmarks).",
          "timestamp": "2023-05-22T12:06:43Z",
          "tree_id": "8ea0780d008ea84646dd464cbb951dd6c168ae1b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8aa74416432c663495b2269180992fc6a3d0a311"
        },
        "date": 1684769327679,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2039117811,
            "range": "± 1988303",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 5023642213,
            "range": "± 14716199",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4263087614,
            "range": "± 3118058",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 11001769050,
            "range": "± 28597649",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 8922764928,
            "range": "± 5676523",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 23570985566,
            "range": "± 45748736",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 18505133005,
            "range": "± 26939815",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 49367090467,
            "range": "± 662681961",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1154925226,
            "range": "± 209372",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2422687397,
            "range": "± 437019",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5073182102,
            "range": "± 964628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10601009431,
            "range": "± 1087252",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 86927114,
            "range": "± 797503",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 182285680,
            "range": "± 664812",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 368741381,
            "range": "± 909387",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 754827405,
            "range": "± 1821779",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3215653239,
            "range": "± 1359618",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 6719708244,
            "range": "± 3169016",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 14036343386,
            "range": "± 15708942",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 29271343583,
            "range": "± 41578784",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 21683637372,
            "range": "± 2851695",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 43660022635,
            "range": "± 10181156",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 87928737056,
            "range": "± 25207697",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 177062100184,
            "range": "± 52272366",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 3091,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 395825,
            "range": "± 302",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1884,
            "range": "± 1",
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
            "value": 2484,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1879,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1843,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 5989396,
            "range": "± 1618",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 1954532625,
            "range": "± 2643935",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 10778790012,
            "range": "± 22727999",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 66697826029,
            "range": "± 103996137",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 454941544675,
            "range": "± 1691187991",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 1540150,
            "range": "± 355",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47506558+MegaRedHand@users.noreply.github.com",
            "name": "Tomás",
            "username": "MegaRedHand"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "8f535ec0925a8637532fff45cdfa30097a1a0d5d",
          "message": "Add support for multiple fields in CUDA (#320)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Extract stark256 kernel specialization\n\n* Add support for multiple fields in CudaState\n\n* Fix compilation errors\n\n* Differentiate specific functions by module name\n\nUsing the function name and the module name is problematic, as the function list when loading a ptx requires a `'static` bound.\n\n* Fix errors\n\n* Fix compile error\n\n* Recompile shader\n\n* Remove commented code\n\n* Fix shader errors\n\n* Fix missing import\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>",
          "timestamp": "2023-05-22T15:28:29Z",
          "tree_id": "0c1a1275a8fc0ba1b5d32b5a7430cc186f472cc0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8f535ec0925a8637532fff45cdfa30097a1a0d5d"
        },
        "date": 1684769657734,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130714297,
            "range": "± 3229934",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 247844638,
            "range": "± 2495702",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 498540729,
            "range": "± 5500391",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 984908375,
            "range": "± 6044919",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34982741,
            "range": "± 431993",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68620056,
            "range": "± 901274",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132641697,
            "range": "± 407595",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 278125125,
            "range": "± 3171193",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31097400,
            "range": "± 230627",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59969446,
            "range": "± 415198",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121057195,
            "range": "± 2227820",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 261610917,
            "range": "± 18500569",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 167196920,
            "range": "± 454651",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 336728208,
            "range": "± 1495062",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 658547500,
            "range": "± 6365227",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1319616646,
            "range": "± 11763839",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 458503739,
            "range": "± 2053224",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 917167792,
            "range": "± 4321296",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1805823291,
            "range": "± 4345636",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3561114874,
            "range": "± 34740485",
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
          "id": "509167c53e54529348beaba144e69a1f02e0d1a8",
          "message": "Add kzg tests (#328)\n\n* Tests\n\n* Fix tests\n\n* Add tests\n\n* Add tests\n\n* Add tests\n\n* Clippy\n\n* Remove unnecesary type definition\n\n* Delete constraint_system.rs",
          "timestamp": "2023-05-22T12:43:20Z",
          "tree_id": "119b1b38b2faad14e078ceb78022c178fe809db6",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/509167c53e54529348beaba144e69a1f02e0d1a8"
        },
        "date": 1684771862188,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2255761096,
            "range": "± 3810202",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3949100542,
            "range": "± 52532089",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4717832350,
            "range": "± 1928224",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8963173025,
            "range": "± 48897172",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 9844074729,
            "range": "± 4409963",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19417799603,
            "range": "± 100078413",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 20524698462,
            "range": "± 5702694",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 41203306935,
            "range": "± 127896306",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1295859678,
            "range": "± 994512",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2720342598,
            "range": "± 1154878",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5697661657,
            "range": "± 2724213",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11901508259,
            "range": "± 3493700",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 71953079,
            "range": "± 1999572",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 148208228,
            "range": "± 1735416",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 296835326,
            "range": "± 1499282",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 605941418,
            "range": "± 2023253",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3568230519,
            "range": "± 3158002",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 7470370542,
            "range": "± 2839415",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 15610144455,
            "range": "± 15492574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 32554504477,
            "range": "± 13329597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 21829539796,
            "range": "± 7814512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 43979711479,
            "range": "± 9384747",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 88659462691,
            "range": "± 20708030",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 178665978774,
            "range": "± 57030555",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 34,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 151,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 86,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 158,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 130,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 5957571,
            "range": "± 1327",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 2010681662,
            "range": "± 3995173",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 11199514458,
            "range": "± 29896881",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 70572576423,
            "range": "± 207807554",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 486954748168,
            "range": "± 2530083472",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 1533242,
            "range": "± 1119",
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
          "id": "cadd83c23e273ee40a13887e1c63b3716e75a834",
          "message": "Implemented parallel FFT on CUDA (#268)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Remove commented code\n\n* Remove allow(dead_code)\n\n* Update comment about limb size\n\n* Update link and change to permalink\n\n* Add Oppen's operator>> bithack\n\n* Fix comment\n\n* Revert \"Fix comment\"\n\nThis reverts commit 7e6ce71571a1f6e0dbd0dff91213db91a0be36b1.\n\n* Remove empty tests module\n\nThis module is tested via `Polynomial::*_fft` in `fft/src/polynomial.rs`\n\n* Use longlong instead of long, and fix u128 mul.\n\n* Address \"reduce branch conditions\" TODOs\n\n---------\n\nCo-authored-by: Tomás <tomas.gruner@lambdaclass.com>\nCo-authored-by: Tomás <47506558+MegaRedHand@users.noreply.github.com>\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>",
          "timestamp": "2023-05-22T14:33:54Z",
          "tree_id": "16128b3842acefefa9858db3638c94203966eadd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/cadd83c23e273ee40a13887e1c63b3716e75a834"
        },
        "date": 1684781292328,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2624612406,
            "range": "± 34519742",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 5734325964,
            "range": "± 35374323",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5422454452,
            "range": "± 54998812",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 12460208112,
            "range": "± 117086215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 11470027019,
            "range": "± 57472601",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 26731491887,
            "range": "± 154635143",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 23651250826,
            "range": "± 186882971",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 56609295520,
            "range": "± 331470378",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1439052767,
            "range": "± 13059292",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 3009842950,
            "range": "± 39634727",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 6291768686,
            "range": "± 62129661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 13236672935,
            "range": "± 125814813",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 105738441,
            "range": "± 1375227",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 218780485,
            "range": "± 1358108",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 452720123,
            "range": "± 8065978",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 895016436,
            "range": "± 18992244",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 4292031176,
            "range": "± 480159569",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 8627209046,
            "range": "± 678669756",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 18006364399,
            "range": "± 151701461",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 37351157588,
            "range": "± 485483628",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 27143506266,
            "range": "± 134210479",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 54951377964,
            "range": "± 249546624",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 110672524681,
            "range": "± 694115165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 222936277026,
            "range": "± 1610276089",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 102,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 435,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 146,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 251,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 189,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 206,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 7590507,
            "range": "± 249649",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 2526853169,
            "range": "± 23157469",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 13968439176,
            "range": "± 50900950",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 86230099483,
            "range": "± 758276244",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 587160590878,
            "range": "± 4746524264",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 1991105,
            "range": "± 27091",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "47506558+MegaRedHand@users.noreply.github.com",
            "name": "Tomás",
            "username": "MegaRedHand"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "8f535ec0925a8637532fff45cdfa30097a1a0d5d",
          "message": "Add support for multiple fields in CUDA (#320)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Extract stark256 kernel specialization\n\n* Add support for multiple fields in CudaState\n\n* Fix compilation errors\n\n* Differentiate specific functions by module name\n\nUsing the function name and the module name is problematic, as the function list when loading a ptx requires a `'static` bound.\n\n* Fix errors\n\n* Fix compile error\n\n* Recompile shader\n\n* Remove commented code\n\n* Fix shader errors\n\n* Fix missing import\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>",
          "timestamp": "2023-05-22T15:28:29Z",
          "tree_id": "0c1a1275a8fc0ba1b5d32b5a7430cc186f472cc0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8f535ec0925a8637532fff45cdfa30097a1a0d5d"
        },
        "date": 1684781295243,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2021793807,
            "range": "± 2555011",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4616608173,
            "range": "± 29084544",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4213078857,
            "range": "± 3978357",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9949490598,
            "range": "± 62225989",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 8816075332,
            "range": "± 19351592",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21214933054,
            "range": "± 64997470",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 18376428565,
            "range": "± 9838779",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 45388185962,
            "range": "± 152734816",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1155776662,
            "range": "± 924841",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2425570853,
            "range": "± 378670",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5076673180,
            "range": "± 3872872",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10605553440,
            "range": "± 5873396",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 82922792,
            "range": "± 498863",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 164048239,
            "range": "± 1103703",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 326697731,
            "range": "± 1699588",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 675535490,
            "range": "± 13857728",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3204554712,
            "range": "± 3100390",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 6702964085,
            "range": "± 17240848",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 14000984202,
            "range": "± 7649382",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 29195194550,
            "range": "± 24092146",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 21683589448,
            "range": "± 6940959",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 43663310989,
            "range": "± 7748616",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 87906225425,
            "range": "± 15618907",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 176956083357,
            "range": "± 35809837",
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
            "value": 343,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 120,
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
            "value": 201,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 156,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 156,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 5990045,
            "range": "± 8414",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 1954567226,
            "range": "± 2885708",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 10783810675,
            "range": "± 24739826",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 66740409190,
            "range": "± 124934746",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 453941060665,
            "range": "± 1240054402",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 1539253,
            "range": "± 462",
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
          "id": "e37a5ddc4ac75a8264374171dd30b8a0919185f4",
          "message": "fix const_le in UnsignedInteger (#366)",
          "timestamp": "2023-05-22T19:37:24Z",
          "tree_id": "681d4275f1618007ae7e33df98720ce86bc65d58",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e37a5ddc4ac75a8264374171dd30b8a0919185f4"
        },
        "date": 1684784594599,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131606741,
            "range": "± 2750054",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248559708,
            "range": "± 3910499",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 495157323,
            "range": "± 5823581",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 989494375,
            "range": "± 20916416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33952393,
            "range": "± 233734",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69422074,
            "range": "± 809894",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133083185,
            "range": "± 844253",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276414656,
            "range": "± 2885056",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30881698,
            "range": "± 293190",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 60003760,
            "range": "± 510217",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121303421,
            "range": "± 4361689",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 245100555,
            "range": "± 22091289",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 166516610,
            "range": "± 468971",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 328620114,
            "range": "± 2146651",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 658960875,
            "range": "± 3147923",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1317270208,
            "range": "± 4562748",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 458354645,
            "range": "± 714625",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 919034458,
            "range": "± 6056382",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1800766312,
            "range": "± 13222684",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3682585187,
            "range": "± 196402832",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "944783fa288e3b1a25a3c434174e2e59be51a2e1",
          "message": "bench: migrate to iai-callgrind (#355)",
          "timestamp": "2023-05-22T19:42:06Z",
          "tree_id": "8dd159d692af7b454599a04e34f9e3acf9acef34",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/944783fa288e3b1a25a3c434174e2e59be51a2e1"
        },
        "date": 1684784869375,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131790885,
            "range": "± 29073006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 252787916,
            "range": "± 4354105",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 491658656,
            "range": "± 4893508",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 983051313,
            "range": "± 6704834",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34659311,
            "range": "± 321517",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68796527,
            "range": "± 533855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132550903,
            "range": "± 675913",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276321302,
            "range": "± 3121752",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30748884,
            "range": "± 287159",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59809612,
            "range": "± 939261",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122672520,
            "range": "± 3906368",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 281761375,
            "range": "± 23165301",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 167360606,
            "range": "± 374121",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 334699281,
            "range": "± 1877047",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 654460208,
            "range": "± 7337235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1316973479,
            "range": "± 27700796",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 451216177,
            "range": "± 4216707",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 906557562,
            "range": "± 12972102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1802081479,
            "range": "± 5478984",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3543147541,
            "range": "± 15340037",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}