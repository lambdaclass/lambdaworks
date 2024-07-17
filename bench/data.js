window.BENCHMARK_DATA = {
  "lastUpdate": 1721216301068,
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
        "date": 1711489692209,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 332103978,
            "range": "± 2406313",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 401570597,
            "range": "± 892813",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 305907559,
            "range": "± 381895",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 700511069,
            "range": "± 1531650",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 838836804,
            "range": "± 3877627",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1465481004,
            "range": "± 2971635",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1758915381,
            "range": "± 3747546",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1342077946,
            "range": "± 1769867",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3063625389,
            "range": "± 1741499",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3669993128,
            "range": "± 6664825",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6415496164,
            "range": "± 2847323",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7748450949,
            "range": "± 14600762",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5872327311,
            "range": "± 2268054",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10410585,
            "range": "± 90500",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10449148,
            "range": "± 5465",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 16226984,
            "range": "± 1443967",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 13133999,
            "range": "± 379782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23251471,
            "range": "± 91395",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23102636,
            "range": "± 48532",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 41021178,
            "range": "± 1208024",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 39900397,
            "range": "± 3047506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 46272569,
            "range": "± 31834",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 46380576,
            "range": "± 70171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 77736535,
            "range": "± 2389276",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 82283434,
            "range": "± 2062778",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 93318521,
            "range": "± 247665",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 93071969,
            "range": "± 73257",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 160138891,
            "range": "± 830892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 166137039,
            "range": "± 1577086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 186431371,
            "range": "± 357714",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 186637343,
            "range": "± 219818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 323757664,
            "range": "± 767842",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 329159320,
            "range": "± 3774070",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16775588,
            "range": "± 766320",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34144974,
            "range": "± 514047",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69177981,
            "range": "± 1465434",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 140818398,
            "range": "± 3697951",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 372724989,
            "range": "± 4764901",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 370914161,
            "range": "± 1548006",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 787063949,
            "range": "± 4189362",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1654891808,
            "range": "± 6840022",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3428104368,
            "range": "± 11110682",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7242419729,
            "range": "± 15618007",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393028936,
            "range": "± 3577245",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 819349763,
            "range": "± 1677597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1725344441,
            "range": "± 13455081",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3572125599,
            "range": "± 8588611",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7536499818,
            "range": "± 16833353",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 374,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 100,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 63,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 165,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 470,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 529,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2196,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 47,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 15,
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
            "value": 7499,
            "range": "± 993",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
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
          "id": "52042bf3530fbc1392ff0849695bd9a3ad6b255d",
          "message": "Release v0.6.0: Bold Bolognese (#852)",
          "timestamp": "2024-03-26T21:45:12Z",
          "tree_id": "24db9e04162dc866c6932c2773591bde0fe627bd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/52042bf3530fbc1392ff0849695bd9a3ad6b255d"
        },
        "date": 1711490177649,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 72703459,
            "range": "± 4528828",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 144999135,
            "range": "± 1671818",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 286342760,
            "range": "± 1563096",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 577666687,
            "range": "± 8201746",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35147706,
            "range": "± 4494776",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 65755369,
            "range": "± 3772222",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 127929438,
            "range": "± 1437068",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 258642989,
            "range": "± 2704160",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27644185,
            "range": "± 1533141",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 57317951,
            "range": "± 2754434",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 116187051,
            "range": "± 5143233",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 253218146,
            "range": "± 17445320",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 111964537,
            "range": "± 1463028",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 222226618,
            "range": "± 1315327",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 445820781,
            "range": "± 8410167",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 915420625,
            "range": "± 7817345",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 395201000,
            "range": "± 1312278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 787261458,
            "range": "± 5031520",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1540587708,
            "range": "± 9859490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3089022417,
            "range": "± 23566591",
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
          "id": "52042bf3530fbc1392ff0849695bd9a3ad6b255d",
          "message": "Release v0.6.0: Bold Bolognese (#852)",
          "timestamp": "2024-03-26T21:45:12Z",
          "tree_id": "24db9e04162dc866c6932c2773591bde0fe627bd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/52042bf3530fbc1392ff0849695bd9a3ad6b255d"
        },
        "date": 1711491274640,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 332963894,
            "range": "± 2005667",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 400602398,
            "range": "± 1039450",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 307629461,
            "range": "± 6671814",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 700202986,
            "range": "± 1449748",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 842807965,
            "range": "± 1720573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1466226029,
            "range": "± 851733",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1761632481,
            "range": "± 3220675",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1353307178,
            "range": "± 968572",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3064071500,
            "range": "± 1350313",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3693825791,
            "range": "± 6112181",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6415822421,
            "range": "± 11628412",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7806128072,
            "range": "± 11637245",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5921622176,
            "range": "± 8416200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10413516,
            "range": "± 5110",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10442262,
            "range": "± 1577",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12737450,
            "range": "± 773176",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12687318,
            "range": "± 394954",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23247033,
            "range": "± 65366",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23462921,
            "range": "± 107929",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 33130516,
            "range": "± 792436",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 33277562,
            "range": "± 1003807",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 46535790,
            "range": "± 119042",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 46597751,
            "range": "± 88661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 78557546,
            "range": "± 894987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 77486258,
            "range": "± 1676198",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 93646160,
            "range": "± 185763",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 93345362,
            "range": "± 849855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 161198877,
            "range": "± 559118",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 160070983,
            "range": "± 1272205",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 186269990,
            "range": "± 93167",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 186435631,
            "range": "± 213454",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 324216371,
            "range": "± 1199031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 324154771,
            "range": "± 1368923",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15701260,
            "range": "± 220753",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33963483,
            "range": "± 401061",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69196437,
            "range": "± 1160008",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 138717556,
            "range": "± 989345",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 354630314,
            "range": "± 1083700",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 368535037,
            "range": "± 1527317",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 783374515,
            "range": "± 1909581",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1643910073,
            "range": "± 2142034",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3425630847,
            "range": "± 3673154",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7223069053,
            "range": "± 16809374",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 391891539,
            "range": "± 1130797",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 822658562,
            "range": "± 2721521",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1725507547,
            "range": "± 2229674",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3577347517,
            "range": "± 11778860",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7519402136,
            "range": "± 6166978",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 44,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 381,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 103,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 59,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 165,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 473,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 524,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2236,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 47,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
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
            "value": 86,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7572,
            "range": "± 1154",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
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
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "699c12d65e734414f5b09912eb0af42d77e92737",
          "message": "chore: add poseidon benches comparing other implementations (#856)\n\n* Add poseidon comparation benches with pathfinder and starknet-rs\n\n* Minor syntax change\n\n* Remove plonky3 dependency\n\n* Add comments in bench\n\n* remove unused module",
          "timestamp": "2024-04-04T18:30:20Z",
          "tree_id": "4989c195034e82b725bddd0ab881fd828d63fa2e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/699c12d65e734414f5b09912eb0af42d77e92737"
        },
        "date": 1712256207337,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 73609052,
            "range": "± 6638154",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 144660252,
            "range": "± 1069607",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 285739750,
            "range": "± 4307267",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 572748062,
            "range": "± 32452517",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 31760813,
            "range": "± 457502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 63720174,
            "range": "± 1258410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 123487219,
            "range": "± 664528",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 251479052,
            "range": "± 862598",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 25642634,
            "range": "± 270052",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 48842776,
            "range": "± 992793",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 99609375,
            "range": "± 3853243",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 232384791,
            "range": "± 12536988",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 111987446,
            "range": "± 715325",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 225002993,
            "range": "± 730806",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 443931968,
            "range": "± 13945454",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 893053125,
            "range": "± 19262862",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 401485323,
            "range": "± 2161289",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 776368604,
            "range": "± 5845027",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1540688833,
            "range": "± 13009282",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3047359729,
            "range": "± 20110521",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "699c12d65e734414f5b09912eb0af42d77e92737",
          "message": "chore: add poseidon benches comparing other implementations (#856)\n\n* Add poseidon comparation benches with pathfinder and starknet-rs\n\n* Minor syntax change\n\n* Remove plonky3 dependency\n\n* Add comments in bench\n\n* remove unused module",
          "timestamp": "2024-04-04T18:30:20Z",
          "tree_id": "4989c195034e82b725bddd0ab881fd828d63fa2e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/699c12d65e734414f5b09912eb0af42d77e92737"
        },
        "date": 1712257321463,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 334073762,
            "range": "± 2933594",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 404992127,
            "range": "± 1838668",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 306801871,
            "range": "± 1452318",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 701523234,
            "range": "± 438494",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 852469726,
            "range": "± 5210807",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1469514660,
            "range": "± 1681924",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1776289729,
            "range": "± 5280451",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1348688089,
            "range": "± 1330590",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3066521723,
            "range": "± 2071642",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3711936360,
            "range": "± 11362342",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6426611179,
            "range": "± 12922932",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7832570883,
            "range": "± 20488407",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5896072179,
            "range": "± 8148262",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10422224,
            "range": "± 8603",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10451908,
            "range": "± 3079",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14211327,
            "range": "± 333599",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 14029041,
            "range": "± 580895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23570764,
            "range": "± 92174",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23653053,
            "range": "± 130330",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 36695046,
            "range": "± 942311",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 36927880,
            "range": "± 770123",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 46796222,
            "range": "± 95952",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 46815387,
            "range": "± 118337",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 82086898,
            "range": "± 823199",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 81039846,
            "range": "± 543017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 93890533,
            "range": "± 123632",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 93651962,
            "range": "± 219850",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 163897426,
            "range": "± 1190391",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 165682369,
            "range": "± 1200861",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 187343268,
            "range": "± 553829",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 187195701,
            "range": "± 339103",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 327622424,
            "range": "± 1655071",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 327484568,
            "range": "± 2692851",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16461877,
            "range": "± 467394",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34637666,
            "range": "± 171430",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70708240,
            "range": "± 689611",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 142461067,
            "range": "± 4859340",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 357415947,
            "range": "± 3664514",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 372883562,
            "range": "± 1390038",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 788659825,
            "range": "± 2153049",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1650400487,
            "range": "± 10606178",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3425163523,
            "range": "± 4757550",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7233258222,
            "range": "± 23681105",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 398908143,
            "range": "± 3432544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 831003231,
            "range": "± 1726014",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1737837463,
            "range": "± 7530138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3588495445,
            "range": "± 4510808",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7539335753,
            "range": "± 5151831",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 11,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 55,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 25,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 79,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 53,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 264,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 264,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 11,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
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
            "value": 88,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 65,
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
          "id": "a70ec1dad2de2d715ba8a48cbae59f6971811f6f",
          "message": "Poseidon optimization starknet (#855)\n\n* Poseidon\n\n* Add docs\n\n* Clearer docs\n\n* Remove unneeded pub crate\n\n* Clippy\n\n* Revert clippy\n\n* Allow op ref\n\n* Fix wasm target\n\n* Fix wasm target\n\n* Fmt",
          "timestamp": "2024-04-04T19:44:05Z",
          "tree_id": "01a0dad9078f2cf27cb1252efe851d3de0a6e382",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a70ec1dad2de2d715ba8a48cbae59f6971811f6f"
        },
        "date": 1712260492830,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 73489918,
            "range": "± 4342741",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 143033576,
            "range": "± 2435039",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 290022395,
            "range": "± 3682270",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 569995312,
            "range": "± 6529289",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 32179296,
            "range": "± 3192777",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 66852782,
            "range": "± 4753047",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 125467401,
            "range": "± 814800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 262157687,
            "range": "± 1953129",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29709135,
            "range": "± 137045",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 52086607,
            "range": "± 3172133",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 110408005,
            "range": "± 2432254",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 235634881,
            "range": "± 8084961",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 110375312,
            "range": "± 516879",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 224431423,
            "range": "± 877177",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 446586125,
            "range": "± 11019682",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 917176854,
            "range": "± 10391575",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 401783219,
            "range": "± 1752399",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 776815020,
            "range": "± 6227259",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1545251416,
            "range": "± 4581888",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3055807604,
            "range": "± 32672187",
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
          "id": "a70ec1dad2de2d715ba8a48cbae59f6971811f6f",
          "message": "Poseidon optimization starknet (#855)\n\n* Poseidon\n\n* Add docs\n\n* Clearer docs\n\n* Remove unneeded pub crate\n\n* Clippy\n\n* Revert clippy\n\n* Allow op ref\n\n* Fix wasm target\n\n* Fix wasm target\n\n* Fmt",
          "timestamp": "2024-04-04T19:44:05Z",
          "tree_id": "01a0dad9078f2cf27cb1252efe851d3de0a6e382",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a70ec1dad2de2d715ba8a48cbae59f6971811f6f"
        },
        "date": 1712261611403,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 331484637,
            "range": "± 2151881",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 394492591,
            "range": "± 1232843",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 304721897,
            "range": "± 561243",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 698989759,
            "range": "± 1085623",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 840667490,
            "range": "± 3254438",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1465638917,
            "range": "± 750804",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1763976186,
            "range": "± 4342132",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1345721165,
            "range": "± 664857",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3063165186,
            "range": "± 14020962",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3675137502,
            "range": "± 8402759",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6414773524,
            "range": "± 3751274",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7768865673,
            "range": "± 27600641",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5890393015,
            "range": "± 14448193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10412788,
            "range": "± 4668",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10444246,
            "range": "± 3015",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12571671,
            "range": "± 13295",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12616941,
            "range": "± 6981",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 22417097,
            "range": "± 30742",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 22417960,
            "range": "± 31028",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28283605,
            "range": "± 316132",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28284163,
            "range": "± 220846",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 45860274,
            "range": "± 94018",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 46002055,
            "range": "± 90638",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 76730185,
            "range": "± 1121945",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 76344603,
            "range": "± 443634",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 92889297,
            "range": "± 262342",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 92912468,
            "range": "± 168438",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 160790654,
            "range": "± 915934",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 161955299,
            "range": "± 1646178",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 187420298,
            "range": "± 196755",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 187257495,
            "range": "± 107118",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 323584094,
            "range": "± 1878928",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 325592822,
            "range": "± 1043297",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16293957,
            "range": "± 106581",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34372848,
            "range": "± 145390",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69551097,
            "range": "± 412053",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 138656535,
            "range": "± 1543055",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 366823492,
            "range": "± 1919423",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 363043987,
            "range": "± 853295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 775006281,
            "range": "± 1695182",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1639872458,
            "range": "± 12760650",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3436565094,
            "range": "± 11239699",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7242985819,
            "range": "± 15831261",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 386663822,
            "range": "± 501646",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 817756962,
            "range": "± 1057220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1723346571,
            "range": "± 1074334",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3578985229,
            "range": "± 3816153",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7533135964,
            "range": "± 15941976",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 44,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 386,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 103,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 59,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 164,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 495,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 593,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2198,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 46,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 108,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7904,
            "range": "± 1697",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 34,
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
            "email": "38444956+GabrielSac@users.noreply.github.com",
            "name": "GabrielSac",
            "username": "GabrielSac"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1a6d14ee7def60d0692d2b8fd323ca0d494920fd",
          "message": "Optimize poseidon (#860)\n\n* Optimize constant addition for partial rounds\n\n* Remove unnecessary dependancies\n\n* Change unnecessary extra minus operation\n\n* Comment auxiliary functions\n\n* Simplify optimization\n\n* Refactor constants\n\n* Begin constants clean-up\n\n* Add explanations for the new code. Change round to index to better handle optimized constants\n\n* Make mix function inline\n\n* Change multiplication to square\n\n* Update parameters.rs\n\n---------\n\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>",
          "timestamp": "2024-04-08T10:09:48Z",
          "tree_id": "c3869a97de25bd94b1b95361689d21a626778bdb",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1a6d14ee7def60d0692d2b8fd323ca0d494920fd"
        },
        "date": 1712571693434,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 78392812,
            "range": "± 14517089",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 146206543,
            "range": "± 9137503",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 296421166,
            "range": "± 4892725",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 585519770,
            "range": "± 23399620",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34391609,
            "range": "± 6936621",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67879497,
            "range": "± 3316448",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 143206907,
            "range": "± 4961958",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 270759395,
            "range": "± 43091888",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 35787955,
            "range": "± 2108022",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 69015558,
            "range": "± 8296375",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 127359364,
            "range": "± 7529458",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 226864138,
            "range": "± 19417356",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 109384328,
            "range": "± 2686151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 218870374,
            "range": "± 30402849",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 451788426,
            "range": "± 12492512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 911115521,
            "range": "± 47819187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 395015687,
            "range": "± 3222368",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 789931625,
            "range": "± 15999308",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1553508729,
            "range": "± 20938476",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3117586271,
            "range": "± 57127378",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "38444956+GabrielSac@users.noreply.github.com",
            "name": "GabrielSac",
            "username": "GabrielSac"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1a6d14ee7def60d0692d2b8fd323ca0d494920fd",
          "message": "Optimize poseidon (#860)\n\n* Optimize constant addition for partial rounds\n\n* Remove unnecessary dependancies\n\n* Change unnecessary extra minus operation\n\n* Comment auxiliary functions\n\n* Simplify optimization\n\n* Refactor constants\n\n* Begin constants clean-up\n\n* Add explanations for the new code. Change round to index to better handle optimized constants\n\n* Make mix function inline\n\n* Change multiplication to square\n\n* Update parameters.rs\n\n---------\n\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>",
          "timestamp": "2024-04-08T10:09:48Z",
          "tree_id": "c3869a97de25bd94b1b95361689d21a626778bdb",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1a6d14ee7def60d0692d2b8fd323ca0d494920fd"
        },
        "date": 1712572764620,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 332219163,
            "range": "± 2824975",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 395937701,
            "range": "± 2364950",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 305126167,
            "range": "± 1107294",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 699313368,
            "range": "± 751725",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 837603199,
            "range": "± 7624979",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1466603992,
            "range": "± 2418907",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1765920705,
            "range": "± 2155431",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1345082135,
            "range": "± 5525576",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3065897306,
            "range": "± 13892103",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3687341543,
            "range": "± 28706559",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6422036313,
            "range": "± 15399369",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7771485093,
            "range": "± 22340047",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5886962977,
            "range": "± 5036101",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10418314,
            "range": "± 7310",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10445128,
            "range": "± 4593",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12527819,
            "range": "± 116472",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12918510,
            "range": "± 130672",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23058300,
            "range": "± 146372",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 22928912,
            "range": "± 22318",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 31004131,
            "range": "± 481312",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30569022,
            "range": "± 506901",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 46143474,
            "range": "± 61792",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 46524759,
            "range": "± 6281901",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 75672783,
            "range": "± 402065",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 75898758,
            "range": "± 324754",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 93122168,
            "range": "± 206710",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 93120984,
            "range": "± 142654",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 158564746,
            "range": "± 815258",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 157260252,
            "range": "± 263537",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 186373378,
            "range": "± 79484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 186318621,
            "range": "± 90764",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 320955311,
            "range": "± 652692",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 321250653,
            "range": "± 1127078",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15186425,
            "range": "± 327085",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32705936,
            "range": "± 308106",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 67649026,
            "range": "± 624040",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 136095218,
            "range": "± 860988",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 350710055,
            "range": "± 1397612",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367154392,
            "range": "± 4114867",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 778120795,
            "range": "± 2138741",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1637181974,
            "range": "± 2324016",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3419502598,
            "range": "± 13625092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7212247455,
            "range": "± 10087919",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 387358302,
            "range": "± 920398",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 817342745,
            "range": "± 882303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1722496433,
            "range": "± 3098337",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3573554581,
            "range": "± 4721429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7516118660,
            "range": "± 10629183",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 83,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 59,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 85,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 148,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 246,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 691,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 24,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 8,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 173,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10471,
            "range": "± 241",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 186,
            "range": "± 1",
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
          "id": "f6dda1c526c49a33809684c44a948e20dadf5a78",
          "message": "Release v0.7.0: Poached Pepper (#867)",
          "timestamp": "2024-04-24T21:24:28Z",
          "tree_id": "20632e55f192f773863c918728f0e9abdf8cac8d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f6dda1c526c49a33809684c44a948e20dadf5a78"
        },
        "date": 1713995839887,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 333126113,
            "range": "± 1773450",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 403562913,
            "range": "± 9903784",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 303520832,
            "range": "± 315648",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 699803521,
            "range": "± 13641521",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 874050487,
            "range": "± 54435290",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1469923603,
            "range": "± 13681647",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1818935240,
            "range": "± 45095384",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1336626959,
            "range": "± 2993996",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3065884661,
            "range": "± 14015538",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3917386519,
            "range": "± 135330798",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6421607448,
            "range": "± 32781693",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 8401218773,
            "range": "± 164395637",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5861907946,
            "range": "± 8951843",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10422026,
            "range": "± 3899",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10455954,
            "range": "± 7370",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 15536117,
            "range": "± 407323",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 17454604,
            "range": "± 2342540",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 24165940,
            "range": "± 124295",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 24055298,
            "range": "± 338416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 39433850,
            "range": "± 3018396",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 40149936,
            "range": "± 1853556",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 47331425,
            "range": "± 399827",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 47610769,
            "range": "± 546285",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 95238009,
            "range": "± 7909722",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 92129387,
            "range": "± 5713759",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 94258010,
            "range": "± 398366",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 95072954,
            "range": "± 1184291",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 174644361,
            "range": "± 6748902",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 173285569,
            "range": "± 7645524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 188760091,
            "range": "± 2209905",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 187846943,
            "range": "± 2583410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 347688669,
            "range": "± 16908028",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 333264313,
            "range": "± 11649546",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17837541,
            "range": "± 1119843",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34241643,
            "range": "± 2342444",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 84670093,
            "range": "± 9132223",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143597518,
            "range": "± 12603566",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 392338432,
            "range": "± 10514743",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 369955208,
            "range": "± 1441868",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 784064391,
            "range": "± 1833982",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1662487181,
            "range": "± 15168998",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3448886253,
            "range": "± 52908409",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7308882827,
            "range": "± 35911348",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396283602,
            "range": "± 3290138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 842552702,
            "range": "± 10385883",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1761285264,
            "range": "± 15993802",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3617187807,
            "range": "± 23465973",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7642876478,
            "range": "± 68669659",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 11,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 54,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 78,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 51,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 248,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 249,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 9,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 107,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7988,
            "range": "± 981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 34,
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
          "id": "f6dda1c526c49a33809684c44a948e20dadf5a78",
          "message": "Release v0.7.0: Poached Pepper (#867)",
          "timestamp": "2024-04-24T21:24:28Z",
          "tree_id": "20632e55f192f773863c918728f0e9abdf8cac8d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f6dda1c526c49a33809684c44a948e20dadf5a78"
        },
        "date": 1716492718567,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 68800595,
            "range": "± 1553271",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 152423932,
            "range": "± 1068509",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 297591041,
            "range": "± 2789451",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 608674271,
            "range": "± 28285757",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 32699844,
            "range": "± 193059",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 64169576,
            "range": "± 446236",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 126316518,
            "range": "± 2430747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 261893281,
            "range": "± 2533909",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30437359,
            "range": "± 250070",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56594849,
            "range": "± 3054447",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120363979,
            "range": "± 7676346",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 243124510,
            "range": "± 20043030",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 113664052,
            "range": "± 1108971",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 227128756,
            "range": "± 1245290",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 450666969,
            "range": "± 1766879",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 920379687,
            "range": "± 19888594",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 394864145,
            "range": "± 1738286",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 790596687,
            "range": "± 7647271",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1550968166,
            "range": "± 4427401",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3093441187,
            "range": "± 19796324",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e465d7c791dce405c0c630f6afd44b12f277ab0a",
          "message": "Handle single element cases in MerkleTree and update tests (#871)\n\n* Handle single element cases in MerkleTree and update test\n\n* Apply rustfmt to format code\n\n* fix clippy issues\n\n* fix clippy issues\n\n* fix clippy issues\n\n* save work\n\n* save work\n\n* fix deleted coments by mistake\n\n* run cargo fmt\n\n* add test to verify Merkle tree with a single element\n\n* cargo fmt\n\n* handle single element case\n\n* run cargo fmt\n\n* suggested changes\n\n* save work\n\n* change function name to better describe its actual operation",
          "timestamp": "2024-06-12T19:20:09Z",
          "tree_id": "97185614a030b5311be7f08385f40ba56b78ae79",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e465d7c791dce405c0c630f6afd44b12f277ab0a"
        },
        "date": 1718220807000,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 91981613,
            "range": "± 8180845",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 145462009,
            "range": "± 8846657",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 288799874,
            "range": "± 6672976",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 580665875,
            "range": "± 6882727",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 32044005,
            "range": "± 105388",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 64446607,
            "range": "± 629563",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 125042369,
            "range": "± 787992",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 262407125,
            "range": "± 1680278",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30157368,
            "range": "± 350484",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58239270,
            "range": "± 904750",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 114926933,
            "range": "± 5311571",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 268166301,
            "range": "± 11512547",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 113490452,
            "range": "± 667309",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 225388631,
            "range": "± 1225874",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 448874541,
            "range": "± 3425646",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 915644687,
            "range": "± 22074817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 394874781,
            "range": "± 611788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 786263520,
            "range": "± 9045253",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1551797770,
            "range": "± 4514657",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3072606833,
            "range": "± 38452308",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e465d7c791dce405c0c630f6afd44b12f277ab0a",
          "message": "Handle single element cases in MerkleTree and update tests (#871)\n\n* Handle single element cases in MerkleTree and update test\n\n* Apply rustfmt to format code\n\n* fix clippy issues\n\n* fix clippy issues\n\n* fix clippy issues\n\n* save work\n\n* save work\n\n* fix deleted coments by mistake\n\n* run cargo fmt\n\n* add test to verify Merkle tree with a single element\n\n* cargo fmt\n\n* handle single element case\n\n* run cargo fmt\n\n* suggested changes\n\n* save work\n\n* change function name to better describe its actual operation",
          "timestamp": "2024-06-12T19:20:09Z",
          "tree_id": "97185614a030b5311be7f08385f40ba56b78ae79",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e465d7c791dce405c0c630f6afd44b12f277ab0a"
        },
        "date": 1718221832844,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 358536701,
            "range": "± 455290",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 389453945,
            "range": "± 1602074",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 305330552,
            "range": "± 400825",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 754621064,
            "range": "± 674002",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 820339114,
            "range": "± 2125553",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1580992716,
            "range": "± 1508772",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1723239010,
            "range": "± 7139720",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1345683794,
            "range": "± 796811",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3305089274,
            "range": "± 1858290",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3595296147,
            "range": "± 3413478",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6915612111,
            "range": "± 7142057",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7576049866,
            "range": "± 13358268",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5892004530,
            "range": "± 10858286",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 10602903,
            "range": "± 2685",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 10644350,
            "range": "± 3950",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12797938,
            "range": "± 296467",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12854245,
            "range": "± 243872",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23569867,
            "range": "± 95146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23526722,
            "range": "± 144694",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 33193716,
            "range": "± 1083728",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 32936041,
            "range": "± 959774",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 47260096,
            "range": "± 152301",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 47372640,
            "range": "± 110003",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 79513811,
            "range": "± 494146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 80244838,
            "range": "± 455706",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 95182204,
            "range": "± 265339",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 94997191,
            "range": "± 146375",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 166422385,
            "range": "± 1365349",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 166714149,
            "range": "± 1337209",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 189200039,
            "range": "± 240090",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 189065187,
            "range": "± 282898",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 333776910,
            "range": "± 1438190",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 332770334,
            "range": "± 1706902",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17269150,
            "range": "± 156068",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36233562,
            "range": "± 471324",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74989995,
            "range": "± 563018",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 152560062,
            "range": "± 2226588",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 373056418,
            "range": "± 1651868",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 392618984,
            "range": "± 1971129",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 838640287,
            "range": "± 1010699",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1766901946,
            "range": "± 1826867",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3678949849,
            "range": "± 3935416",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7749279387,
            "range": "± 8329365",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 416891951,
            "range": "± 1277612",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 880800618,
            "range": "± 885953",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1849550735,
            "range": "± 3689015",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3839925699,
            "range": "± 4168214",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 8063180063,
            "range": "± 10339884",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 241,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7741,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 283,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 163,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 458,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6152,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1157,
            "range": "± 518",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 24768,
            "range": "± 1509",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 246,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "33334338+PatStiles@users.noreply.github.com",
            "name": "PatStiles",
            "username": "PatStiles"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dfe4f0eaf9b0d62d85e881c87622c37f1e1b2ea2",
          "message": "feat(merkle): Pad single tree leaves to next power of two (#876)\n\n* pad single elements to next power of two\r\n\r\n* fix starknet version\r\n\r\n* address pelitos cmts\r\n\r\n* fmt",
          "timestamp": "2024-07-02T11:29:55-03:00",
          "tree_id": "3e18aebc8a4c68ed7e530287010e7cf4f4bd0bab",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/dfe4f0eaf9b0d62d85e881c87622c37f1e1b2ea2"
        },
        "date": 1719930949164,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 74874510,
            "range": "± 4112696",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 146021660,
            "range": "± 36655151",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 290628458,
            "range": "± 2099430",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 579422229,
            "range": "± 6546325",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 32029196,
            "range": "± 2195516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 64007801,
            "range": "± 491059",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 124596498,
            "range": "± 1254386",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 261052718,
            "range": "± 1986109",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29942444,
            "range": "± 214405",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 57773798,
            "range": "± 3280334",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 116857391,
            "range": "± 4814755",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 234990812,
            "range": "± 20606188",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 107274531,
            "range": "± 3519625",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 226113083,
            "range": "± 971771",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 449233520,
            "range": "± 1915999",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 912416208,
            "range": "± 24865228",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 395274301,
            "range": "± 837885",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 790768604,
            "range": "± 4829163",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1550514937,
            "range": "± 4924583",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3113762604,
            "range": "± 21086875",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "33334338+PatStiles@users.noreply.github.com",
            "name": "PatStiles",
            "username": "PatStiles"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dfe4f0eaf9b0d62d85e881c87622c37f1e1b2ea2",
          "message": "feat(merkle): Pad single tree leaves to next power of two (#876)\n\n* pad single elements to next power of two\r\n\r\n* fix starknet version\r\n\r\n* address pelitos cmts\r\n\r\n* fmt",
          "timestamp": "2024-07-02T11:29:55-03:00",
          "tree_id": "3e18aebc8a4c68ed7e530287010e7cf4f4bd0bab",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/dfe4f0eaf9b0d62d85e881c87622c37f1e1b2ea2"
        },
        "date": 1719931994517,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323219067,
            "range": "± 513602",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 368294537,
            "range": "± 813858",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279538087,
            "range": "± 744472",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 678136073,
            "range": "± 799604",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 785651300,
            "range": "± 7442758",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1422216968,
            "range": "± 934021",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1646625920,
            "range": "± 8625455",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1230500105,
            "range": "± 1578943",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2971866548,
            "range": "± 741455",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3404293864,
            "range": "± 10914266",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6223732022,
            "range": "± 4009545",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7215541701,
            "range": "± 16547071",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5391961307,
            "range": "± 5988785",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7652300,
            "range": "± 3851",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7705209,
            "range": "± 8427",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9795554,
            "range": "± 54032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9875266,
            "range": "± 72188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17727260,
            "range": "± 64702",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17769047,
            "range": "± 58609",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27235688,
            "range": "± 263381",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27163567,
            "range": "± 266200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35589526,
            "range": "± 144947",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35709667,
            "range": "± 61097",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68829738,
            "range": "± 699695",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68680972,
            "range": "± 685359",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72093810,
            "range": "± 107892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71872844,
            "range": "± 126073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 144943056,
            "range": "± 1417824",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 146325219,
            "range": "± 1275166",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 142583185,
            "range": "± 255400",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142869290,
            "range": "± 233272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 298356635,
            "range": "± 5174586",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 298183896,
            "range": "± 2186670",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18327753,
            "range": "± 467949",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 37585912,
            "range": "± 1152803",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73887253,
            "range": "± 1179268",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 147568978,
            "range": "± 1790875",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 366288113,
            "range": "± 4403137",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 352605871,
            "range": "± 925513",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 752184993,
            "range": "± 781075",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1597331754,
            "range": "± 3728200",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3334702265,
            "range": "± 7995262",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7033344532,
            "range": "± 17110482",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 379186749,
            "range": "± 1044377",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 798595873,
            "range": "± 1801381",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1686787402,
            "range": "± 4021181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3474396772,
            "range": "± 4879524",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7306933829,
            "range": "± 12744593",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1112,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 142381,
            "range": "± 154",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 747,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 349,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1097,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 77311,
            "range": "± 697",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2472,
            "range": "± 2578",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 258631,
            "range": "± 3431",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1134,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 13,
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
            "value": 7667,
            "range": "± 1153",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 32,
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
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "712d894462c11084f30ed2cb884078a0970ba9de",
          "message": "Add error to UnsInt from_hex when hex too big (#880)\n\n* Add error to UnsInt from_hex when hex too big\n\n* Format and add function docs\n\n* Format and add function docs\n\n* chore: add same docs in from_hex function in field/element.rs\n\n* Update math/src/field/element.rs\n\n* Update math/src/unsigned_integer/element.rs\n\n* Add tests from hex in montgomery backend\n\n* Lint\n\n---------\n\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>",
          "timestamp": "2024-07-17T11:22:55Z",
          "tree_id": "9e5fb0bc1a3a7ee9da89a722d8cd1c3070b054bc",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/712d894462c11084f30ed2cb884078a0970ba9de"
        },
        "date": 1721216298291,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 73009186,
            "range": "± 1224631",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 145361265,
            "range": "± 11910408",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 289321541,
            "range": "± 8619201",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 585305771,
            "range": "± 7457654",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 31825418,
            "range": "± 103756",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 63827025,
            "range": "± 746429",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 125801057,
            "range": "± 977435",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 270176854,
            "range": "± 5729303",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 36974296,
            "range": "± 4488286",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 70078979,
            "range": "± 4960128",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 136165389,
            "range": "± 7751522",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 261254374,
            "range": "± 13721935",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 134663044,
            "range": "± 5943885",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 239234527,
            "range": "± 13436593",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 494323875,
            "range": "± 23728249",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1007976104,
            "range": "± 47897322",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 413167260,
            "range": "± 29579850",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 832746729,
            "range": "± 19075517",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1571153479,
            "range": "± 34427320",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3140374167,
            "range": "± 167007415",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}