window.BENCHMARK_DATA = {
  "lastUpdate": 1711488590782,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
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
          "id": "ac712583f6ae1e599f910bd444b32666a68667ff",
          "message": "Peder optimization #1 (#848)\n\n* Pedersen\n\n* Add random in bench\n\n* Fix seed\n\n* Opt\n\n* Fix bench\n\n* Apply format\n\n* Change version in winerfell adapter\n\n* Fix clippy\n\n---------\n\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>",
          "timestamp": "2024-03-26T19:34:16Z",
          "tree_id": "9caa21c33e332662304f9e9fd4333fa859d784ad",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ac712583f6ae1e599f910bd444b32666a68667ff"
        },
        "date": 1711483548564,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 333465780,
            "range": "± 429401",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 401019092,
            "range": "± 2115411",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 305637785,
            "range": "± 195213",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 700699020,
            "range": "± 597523",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 840972049,
            "range": "± 2983199",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1467441343,
            "range": "± 1591949",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1764824280,
            "range": "± 8765500",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1344975552,
            "range": "± 930558",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3066619195,
            "range": "± 2022108",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3689809138,
            "range": "± 4443068",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6420160414,
            "range": "± 13547092",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7794284837,
            "range": "± 10451314",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5881989051,
            "range": "± 14805491",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10414497,
            "range": "± 4679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10450299,
            "range": "± 18670",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 13204793,
            "range": "± 263605",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 13570448,
            "range": "± 188699",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23576968,
            "range": "± 66385",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23344315,
            "range": "± 62032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34712062,
            "range": "± 1137131",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 35159454,
            "range": "± 1435239",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 46752648,
            "range": "± 100496",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 46801869,
            "range": "± 128312",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 79639489,
            "range": "± 1389959",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 79683809,
            "range": "± 1043177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 93895591,
            "range": "± 147948",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 93622909,
            "range": "± 738368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 161931821,
            "range": "± 513165",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 162855768,
            "range": "± 1544888",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 186933605,
            "range": "± 201216",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 187317483,
            "range": "± 325886",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 330283244,
            "range": "± 2145694",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 328854521,
            "range": "± 1774496",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16581209,
            "range": "± 929302",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35972495,
            "range": "± 781713",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70745936,
            "range": "± 973137",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 141855092,
            "range": "± 881803",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 368539807,
            "range": "± 7000657",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367971130,
            "range": "± 1224201",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 782959443,
            "range": "± 2967055",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1647822051,
            "range": "± 5215934",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3436036345,
            "range": "± 7692597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7246300695,
            "range": "± 8236379",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393594250,
            "range": "± 1569068",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 824651732,
            "range": "± 13071824",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1731090173,
            "range": "± 2076011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3591363704,
            "range": "± 6041085",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7556111072,
            "range": "± 5046634",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 522,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 33478,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 472,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 259,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 740,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 23529,
            "range": "± 115",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1622,
            "range": "± 792",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 78805,
            "range": "± 2160",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 536,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 64,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juanbono94@gmail.com",
            "name": "Juan Bono",
            "username": "juanbono"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0cd3ef57b3c0807316d42c20374ca59ed45df187",
          "message": "Improve docs (#849)\n\n* add usage section for lambdaworks-math\n\n* add docs for crypto crate\n\n* add crates to introduction",
          "timestamp": "2024-03-26T19:56:10Z",
          "tree_id": "f7956253c0f5857cb226820b37905d285ee36e0b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/0cd3ef57b3c0807316d42c20374ca59ed45df187"
        },
        "date": 1711483799905,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 86112755,
            "range": "± 6018623",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 168107323,
            "range": "± 5250147",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 336327875,
            "range": "± 11942119",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 705235500,
            "range": "± 28557535",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 47515577,
            "range": "± 3412026",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 80579182,
            "range": "± 3135378",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 141350035,
            "range": "± 3276309",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 273356645,
            "range": "± 7512054",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 42701506,
            "range": "± 3672405",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 75103400,
            "range": "± 4811299",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 144209367,
            "range": "± 5267362",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 261816135,
            "range": "± 9450491",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 145903367,
            "range": "± 13051391",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 275888354,
            "range": "± 9776737",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 524877750,
            "range": "± 39361805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1095632875,
            "range": "± 46342087",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 411219510,
            "range": "± 7726100",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 842486625,
            "range": "± 27333384",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1635950666,
            "range": "± 59437681",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3353040541,
            "range": "± 53343779",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juanbono94@gmail.com",
            "name": "Juan Bono",
            "username": "juanbono"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0cd3ef57b3c0807316d42c20374ca59ed45df187",
          "message": "Improve docs (#849)\n\n* add usage section for lambdaworks-math\n\n* add docs for crypto crate\n\n* add crates to introduction",
          "timestamp": "2024-03-26T19:56:10Z",
          "tree_id": "f7956253c0f5857cb226820b37905d285ee36e0b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/0cd3ef57b3c0807316d42c20374ca59ed45df187"
        },
        "date": 1711484734057,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 333345168,
            "range": "± 1595876",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 405293022,
            "range": "± 3455371",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 305774841,
            "range": "± 298439",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 701076028,
            "range": "± 752960",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 857098885,
            "range": "± 7361980",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1468884420,
            "range": "± 1552098",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1798041779,
            "range": "± 10675813",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1344929312,
            "range": "± 483836",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3066837983,
            "range": "± 2199092",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3756972379,
            "range": "± 22890911",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6424901550,
            "range": "± 5153090",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7947125852,
            "range": "± 38564971",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5887195777,
            "range": "± 5993234",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10419389,
            "range": "± 4496",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10447655,
            "range": "± 8073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12550356,
            "range": "± 150190",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12721980,
            "range": "± 152956",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23326007,
            "range": "± 117487",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23218124,
            "range": "± 101099",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 33677902,
            "range": "± 1139970",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 33390286,
            "range": "± 885178",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 46868844,
            "range": "± 132504",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 47086507,
            "range": "± 81859",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 82508831,
            "range": "± 1869345",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 80874777,
            "range": "± 1326201",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 94360171,
            "range": "± 343982",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 94292095,
            "range": "± 294009",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 172217911,
            "range": "± 3301412",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 173198441,
            "range": "± 3283592",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 188141683,
            "range": "± 369634",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 187278126,
            "range": "± 758533",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 350485530,
            "range": "± 6430041",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 351930909,
            "range": "± 8467282",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17547256,
            "range": "± 532400",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 38392914,
            "range": "± 2619656",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 78510374,
            "range": "± 6646632",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 160672367,
            "range": "± 5743777",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 405630509,
            "range": "± 9668584",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 369423628,
            "range": "± 1660555",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 791130995,
            "range": "± 2882374",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1659536759,
            "range": "± 7027060",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3468479539,
            "range": "± 6811335",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7303639110,
            "range": "± 13527571",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 394998015,
            "range": "± 3616585",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 832440433,
            "range": "± 5230531",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1744259744,
            "range": "± 9060295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3616195344,
            "range": "± 8626658",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7616869993,
            "range": "± 24819110",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 43,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 378,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 101,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 58,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 167,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 460,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 502,
            "range": "± 51",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2237,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 44,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7755,
            "range": "± 1593",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 30,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mrugiero@gmail.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6e33e203223c6a5be3427f1fd46dd50dd0155593",
          "message": "perf+feat: generalize and constify Pedersen hash (#851)\n\nSlightly more generic: instead of being implemented only for the STARK\ncurve, make it generic over curves on the STARK252 prime field. Future\nwork may make it work for more fields.\n\nInitialization is much faster now: all parameters are computed at\ncompile time.",
          "timestamp": "2024-03-26T21:18:42Z",
          "tree_id": "a066dbb19ffa87a4e151b5b778d085dee8d06261",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6e33e203223c6a5be3427f1fd46dd50dd0155593"
        },
        "date": 1711488586054,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 76251475,
            "range": "± 4548656",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 145797703,
            "range": "± 3094189",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 286968698,
            "range": "± 3244027",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 588032104,
            "range": "± 7852579",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 32373344,
            "range": "± 273213",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 65478729,
            "range": "± 1320617",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 126638157,
            "range": "± 1317520",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 261896812,
            "range": "± 3150322",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 26843764,
            "range": "± 722600",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 51808998,
            "range": "± 3055832",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 116524079,
            "range": "± 3939771",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 251458406,
            "range": "± 18807387",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 109786748,
            "range": "± 829046",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 226366604,
            "range": "± 1417415",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 449141479,
            "range": "± 2333214",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 913756937,
            "range": "± 16760574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 396705218,
            "range": "± 626368",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 782608062,
            "range": "± 7567456",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1547779875,
            "range": "± 7871569",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3045220625,
            "range": "± 36662031",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}