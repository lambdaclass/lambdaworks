window.BENCHMARK_DATA = {
  "lastUpdate": 1767638852524,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "3046f0e663d9182cd8798f8e9488ad300ffa4e46",
          "message": "Remove Metal feature (#993)\n\n* rm all metal features\n\n* fix clippy\n\n* fix attribute\n\n* fix another attribute",
          "timestamp": "2025-04-04T16:50:42Z",
          "tree_id": "2aaf079d5f553baf9a3a9e9e5307664a03c827cc",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3046f0e663d9182cd8798f8e9488ad300ffa4e46"
        },
        "date": 1743787195774,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 320705388,
            "range": "± 1229369",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375578528,
            "range": "± 1347354",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279807033,
            "range": "± 961256",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 672838962,
            "range": "± 1116880",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 790967290,
            "range": "± 2596968",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1410286987,
            "range": "± 882187",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1657890232,
            "range": "± 2605314",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1232780111,
            "range": "± 1180582",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2949072229,
            "range": "± 1321530",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3459300863,
            "range": "± 8619321",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6179583205,
            "range": "± 17065154",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7291301553,
            "range": "± 247093445",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5414236330,
            "range": "± 28395988",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7468724,
            "range": "± 1841",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7537959,
            "range": "± 44546",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9602368,
            "range": "± 562231",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9865255,
            "range": "± 143691",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17834704,
            "range": "± 163536",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17830367,
            "range": "± 88037",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26109838,
            "range": "± 746716",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26475106,
            "range": "± 1141531",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35383803,
            "range": "± 154065",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35704628,
            "range": "± 2810906",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 64691879,
            "range": "± 567840",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 64827620,
            "range": "± 805876",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71253085,
            "range": "± 124218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71178972,
            "range": "± 58657",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 136134606,
            "range": "± 1075568",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 137532919,
            "range": "± 2683461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 139752058,
            "range": "± 466816",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 139712869,
            "range": "± 477014",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 270812589,
            "range": "± 420818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 272761366,
            "range": "± 1375428",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15817354,
            "range": "± 146297",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34006462,
            "range": "± 1676986",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66811860,
            "range": "± 191893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 133633502,
            "range": "± 282515",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 353461137,
            "range": "± 1402597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 351746130,
            "range": "± 1375309",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 753209559,
            "range": "± 2292721",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1579028534,
            "range": "± 2691542",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3285878471,
            "range": "± 3601429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6940267323,
            "range": "± 4428984",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 375302287,
            "range": "± 298354",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 788066352,
            "range": "± 3896669",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1660458229,
            "range": "± 855585",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3439714851,
            "range": "± 4068957",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7242585864,
            "range": "± 4345635",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 60,
            "range": "± 5",
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
            "value": 88,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 161,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 238,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 605,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 22,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 153,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 13668,
            "range": "± 644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 351,
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
          "id": "d9385675aa91e7e99c8c023e74108fe316b243c7",
          "message": "Update Readme (#987)\n\n* update kzg Readme\n\n* KZG: make sure example is ok\n\n* Add Readme for Merkle Tree\n\n* Add Readme for Circle\n\n* fix circle docs\n\n* fix circle readme\n\n* fix typo, remove unused test and fix markdown style\n\n* changes in circle readme\n\n* fix circle readme\n\n* fix clippy\n\n* fix clippy spaces\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Joaquin Carletti <joaquin.carletti@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-04-04T17:10:25Z",
          "tree_id": "74f03eeb691de0aa94763b5baf42b9f4ed23213b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d9385675aa91e7e99c8c023e74108fe316b243c7"
        },
        "date": 1743788268386,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 319852266,
            "range": "± 3805245",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 371104952,
            "range": "± 1740701",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280876259,
            "range": "± 4949283",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 675443325,
            "range": "± 1105947",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 791678995,
            "range": "± 1828994",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1412436337,
            "range": "± 1278948",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1652765386,
            "range": "± 3284626",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233422421,
            "range": "± 10312045",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2952086639,
            "range": "± 10618799",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3459442373,
            "range": "± 4424472",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6182994347,
            "range": "± 8128966",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7254550053,
            "range": "± 17539210",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5409131663,
            "range": "± 16148765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7487159,
            "range": "± 5591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7555693,
            "range": "± 11890",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9601082,
            "range": "± 52661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9656556,
            "range": "± 12599",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17727102,
            "range": "± 20747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17702983,
            "range": "± 47819",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26793247,
            "range": "± 326104",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26627344,
            "range": "± 326648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35393072,
            "range": "± 61487",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35603696,
            "range": "± 80770",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 65545766,
            "range": "± 711173",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 65906135,
            "range": "± 173066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71450949,
            "range": "± 123146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 70882820,
            "range": "± 676315",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139326643,
            "range": "± 996994",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 137803051,
            "range": "± 1177147",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 140531442,
            "range": "± 882189",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 141094402,
            "range": "± 489707",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 278696365,
            "range": "± 2199133",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 272579587,
            "range": "± 526879",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15534075,
            "range": "± 81717",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32852702,
            "range": "± 232320",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66750776,
            "range": "± 387176",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 135963696,
            "range": "± 2016853",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 354161761,
            "range": "± 1124515",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 350899360,
            "range": "± 1206074",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 748718896,
            "range": "± 7440503",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1577322363,
            "range": "± 1531086",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3295011856,
            "range": "± 13650702",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6931654664,
            "range": "± 8755697",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 379027620,
            "range": "± 1035968",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 791817068,
            "range": "± 3980896",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1665178431,
            "range": "± 1568354",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3448918670,
            "range": "± 7856603",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7253263802,
            "range": "± 9209284",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1148,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 146687,
            "range": "± 113",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 770,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 421,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1116,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 82224,
            "range": "± 172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2705,
            "range": "± 2645",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 198588,
            "range": "± 6535",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1167,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 154,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 13738,
            "range": "± 1295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 355,
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
          "id": "a7b22e18784dc80dc755644c0a6e310d544e1dee",
          "message": "Add Binary Field (#984)\n\n* first commit\n\n* define add function\n\n* add more functions and refactor\n\n* save work\n\n* save work\n\n* fix new function,tests and add benches\n\n* change mul function\n\n* test\n\n* change comment\n\n* change mul algorithm. Mul in level 0 and 1 working\n\n* refactor some functions and fix tests\n\n* fix inverse function\n\n* add docs and update README\n\n* fix conflicts\n\n* fix clippy\n\n* fix no_std\n\n* fix tests no std\n\n* small fixex for benches\n\n* fix typo\n\n* remove num_bits from the struct. remove set_num_level. move mul\n\n* improve add_elements()\n\n* impl Eq instead of equalls function\n\n* derive default\n\n* fix prefix in test\n\n* add readme\n\n* omit mul lifetimes\n\n* use vector of random elements for benches\n\n* fix doc\n\n---------\n\nCo-authored-by: Nicole <nicole@Nicoles-MacBook-Air.local>\nCo-authored-by: Nicole <nicole@Nicoles-Air.fibertel.com.ar>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-04-04T22:04:46Z",
          "tree_id": "c004e1007c6a9c30eae0542f5689f1dd6c537503",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a7b22e18784dc80dc755644c0a6e310d544e1dee"
        },
        "date": 1743805940142,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 322536981,
            "range": "± 677708",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 381470548,
            "range": "± 1184480",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282006174,
            "range": "± 5939037",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 675795635,
            "range": "± 659215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 795681046,
            "range": "± 2425038",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1411782190,
            "range": "± 1751973",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1651376887,
            "range": "± 2140153",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233122116,
            "range": "± 3254844",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2946309310,
            "range": "± 1941423",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3432717624,
            "range": "± 11089516",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6173099230,
            "range": "± 8414029",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7267160590,
            "range": "± 13302343",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5403850651,
            "range": "± 9004410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7477272,
            "range": "± 7846",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7567425,
            "range": "± 203683",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9944972,
            "range": "± 20392",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10014529,
            "range": "± 13387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17618702,
            "range": "± 30856",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17571476,
            "range": "± 20902",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 24371876,
            "range": "± 236123",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 24059528,
            "range": "± 206987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35327920,
            "range": "± 386990",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35556992,
            "range": "± 194748",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 65557303,
            "range": "± 809469",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67348023,
            "range": "± 923589",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71434530,
            "range": "± 133214",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71037702,
            "range": "± 216806",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 136201709,
            "range": "± 1189575",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 135977087,
            "range": "± 2000220",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 138991812,
            "range": "± 1846913",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 138849164,
            "range": "± 100573",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 272816251,
            "range": "± 1189382",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 271332338,
            "range": "± 557768",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15574852,
            "range": "± 132202",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32855737,
            "range": "± 144069",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 65461659,
            "range": "± 167480",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 134686502,
            "range": "± 1861292",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 345398332,
            "range": "± 3959000",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 349246068,
            "range": "± 882945",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 746529539,
            "range": "± 7013100",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1578407282,
            "range": "± 3122986",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3290550733,
            "range": "± 7700902",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6920528032,
            "range": "± 9116519",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 374258082,
            "range": "± 2251004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 794782921,
            "range": "± 4374850",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1654737887,
            "range": "± 1901458",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3434244481,
            "range": "± 10075935",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7233266618,
            "range": "± 9545793",
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
            "value": 443,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 104,
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
            "value": 173,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 538,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 612,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2075,
            "range": "± 159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 51,
            "range": "± 1",
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
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 6273,
            "range": "± 110",
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
            "value": 9886,
            "range": "± 449",
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
            "email": "erhany96@gmail.com",
            "name": "erhant",
            "username": "erhant"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7fe890e658ef17cf826c917ca7a71cf6a8e139f4",
          "message": "fix: Circom Adapter (Groth16 over BLS12-381) (#991)\n\n* migrate fixes from other PR, todo tutorial guide fixes\n\n* update tutorial docs as well\n\n* update readme\n\n* fix lint\n\n* fix export witness.json doc\n\n* Remove Metal feature (#993)\n\n* rm all metal features\n\n* fix clippy\n\n* fix attribute\n\n* fix another attribute\n\n* Update Readme (#987)\n\n* update kzg Readme\n\n* KZG: make sure example is ok\n\n* Add Readme for Merkle Tree\n\n* Add Readme for Circle\n\n* fix circle docs\n\n* fix circle readme\n\n* fix typo, remove unused test and fix markdown style\n\n* changes in circle readme\n\n* fix circle readme\n\n* fix clippy\n\n* fix clippy spaces\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Joaquin Carletti <joaquin.carletti@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Joaquin Carletti <56092489+ColoCarletti@users.noreply.github.com>\nCo-authored-by: jotabulacios <45471455+jotabulacios@users.noreply.github.com>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Joaquin Carletti <joaquin.carletti@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-04-07T10:31:41-03:00",
          "tree_id": "97f48d8c59fcf51b4577e7534ef45b2d59210387",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7fe890e658ef17cf826c917ca7a71cf6a8e139f4"
        },
        "date": 1744034142491,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 321437400,
            "range": "± 348215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 380145457,
            "range": "± 1461575",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280568088,
            "range": "± 394822",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 676218495,
            "range": "± 347059",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 802038191,
            "range": "± 3380461",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1415170326,
            "range": "± 1381686",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1667611437,
            "range": "± 3930616",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1235252309,
            "range": "± 1199095",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2957640592,
            "range": "± 2451027",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3514601085,
            "range": "± 13768918",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6195636016,
            "range": "± 8128520",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7407546036,
            "range": "± 28219330",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5419618752,
            "range": "± 3876031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8222883,
            "range": "± 8167",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8305733,
            "range": "± 11553",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11417889,
            "range": "± 307054",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11524311,
            "range": "± 176140",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19698747,
            "range": "± 71650",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19718337,
            "range": "± 97455",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34685528,
            "range": "± 1095145",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 33501042,
            "range": "± 754484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38611651,
            "range": "± 61391",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 39391628,
            "range": "± 104268",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74704448,
            "range": "± 701171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 75474289,
            "range": "± 346220",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78286128,
            "range": "± 136024",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77947828,
            "range": "± 329831",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 154740630,
            "range": "± 2445437",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 154354360,
            "range": "± 963613",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 152519534,
            "range": "± 363491",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152136820,
            "range": "± 187903",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 310582411,
            "range": "± 3168153",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 303174745,
            "range": "± 3898192",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17386870,
            "range": "± 477977",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 37379851,
            "range": "± 726893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73903717,
            "range": "± 887313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 148636900,
            "range": "± 1040597",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 380034545,
            "range": "± 3996013",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 358053518,
            "range": "± 1218181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 758901629,
            "range": "± 1992783",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1589431581,
            "range": "± 2117222",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3334508358,
            "range": "± 7077874",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7026650519,
            "range": "± 13257586",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383769972,
            "range": "± 732942",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 806184179,
            "range": "± 3276873",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1679221092,
            "range": "± 3563109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3493952437,
            "range": "± 7253656",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7350396887,
            "range": "± 9856155",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 526,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 33821,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 412,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 206,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 714,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 24807,
            "range": "± 422",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1659,
            "range": "± 1626",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 61620,
            "range": "± 679",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 543,
            "range": "± 4",
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
            "value": 68,
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
            "value": 9480,
            "range": "± 1249",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 43,
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
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca84326212a16046c25764a04f09fbd25d4a5a5b",
          "message": "Add RSA and Schnorr Examples (#989)\n\n* first commit. adds basic rsa\n\n* refactor and update README\n\n* finish schnorr signature. rand test not working\n\n* fix comments\n\n* Change random sampl in groth16. Schnorr worikng\n\n* change sample field elem\n\n* fix clippy\n\n* fix clippy\n\n* refactor rsa and update README\n\n* remove comments\n\n* refactor schnorr\n\n* fix clippy\n\n* update ubuntu version for ci\n\n* Update README.md\n\n* Update README.md\n\n* change p and q values in the test\n\n* Update README.md\n\n* Update examples/schnorr-signature/README.md\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2025-04-07T20:32:33Z",
          "tree_id": "bae795434cd5db4edc88ea2d19b5698ea4aa7e2c",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ca84326212a16046c25764a04f09fbd25d4a5a5b"
        },
        "date": 1744059616751,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323302174,
            "range": "± 310320",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 378198393,
            "range": "± 1889551",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282302187,
            "range": "± 579703",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679977030,
            "range": "± 555917",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 800838249,
            "range": "± 1952424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1425839785,
            "range": "± 2514275",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1669592523,
            "range": "± 3375890",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1242617718,
            "range": "± 1801842",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2979299365,
            "range": "± 4949856",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3481898841,
            "range": "± 6663047",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6242153472,
            "range": "± 1656559",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7351475668,
            "range": "± 4358052",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5533157933,
            "range": "± 5560622",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8291674,
            "range": "± 8822",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8314536,
            "range": "± 8702",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10334249,
            "range": "± 73145",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10462067,
            "range": "± 41279",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18895744,
            "range": "± 72045",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18786245,
            "range": "± 63210",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28289380,
            "range": "± 553212",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27904926,
            "range": "± 689561",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37860062,
            "range": "± 113373",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38433200,
            "range": "± 140895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69951362,
            "range": "± 690983",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69917365,
            "range": "± 452670",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 77564187,
            "range": "± 33983",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77410617,
            "range": "± 132595",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 144575783,
            "range": "± 666787",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 145391265,
            "range": "± 509476",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151334758,
            "range": "± 230203",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152036489,
            "range": "± 635478",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289779727,
            "range": "± 599516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 292670351,
            "range": "± 685182",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15200285,
            "range": "± 199663",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33801645,
            "range": "± 123245",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69162244,
            "range": "± 228661",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 139832541,
            "range": "± 472688",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 363144425,
            "range": "± 868273",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 355590805,
            "range": "± 1434883",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 755813997,
            "range": "± 1731425",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1594064923,
            "range": "± 2063410",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3328267097,
            "range": "± 3029485",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7024047657,
            "range": "± 2048699",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 379896032,
            "range": "± 943930",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 798572850,
            "range": "± 1282477",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1677234138,
            "range": "± 1460907",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3489203738,
            "range": "± 1426865",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7333133452,
            "range": "± 2950916",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 52,
            "range": "± 0",
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
            "value": 79,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 242,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 44,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 111,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11799,
            "range": "± 1536",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 209,
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
            "email": "70894690+0xLucqs@users.noreply.github.com",
            "name": "0xLucqs",
            "username": "0xLucqs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "201f08d93652006d30cce0dbda83c68fa20f4213",
          "message": "fix(coset): shl overflow when getting the points (#994)\n\n* fix(coset): shl overflow when getting the points\n\n* impl suggestion",
          "timestamp": "2025-04-08T20:16:18Z",
          "tree_id": "f8801b9b7fb0e19c401c46b50ae9b84edce6c97e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/201f08d93652006d30cce0dbda83c68fa20f4213"
        },
        "date": 1744145059078,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 322612516,
            "range": "± 783501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375172800,
            "range": "± 1755920",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 281936890,
            "range": "± 254910",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 677832008,
            "range": "± 13655791",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 794145062,
            "range": "± 13243424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1423013668,
            "range": "± 2821673",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1669372326,
            "range": "± 13723412",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1242990934,
            "range": "± 673322",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2970303244,
            "range": "± 2855584",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3466325400,
            "range": "± 7093704",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6222552366,
            "range": "± 14197329",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7317714054,
            "range": "± 19869092",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5452381623,
            "range": "± 1885502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7533338,
            "range": "± 60821",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7599272,
            "range": "± 18669",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12299973,
            "range": "± 1477743",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10896407,
            "range": "± 297657",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17683720,
            "range": "± 71712",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17227183,
            "range": "± 38677",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28665738,
            "range": "± 2466855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26790666,
            "range": "± 1521028",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 34946833,
            "range": "± 94476",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35188797,
            "range": "± 119149",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 66806435,
            "range": "± 1623807",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66642918,
            "range": "± 3325895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71874222,
            "range": "± 164695",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71365370,
            "range": "± 168413",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139299207,
            "range": "± 962149",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 138070290,
            "range": "± 836855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 142398779,
            "range": "± 457154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142181162,
            "range": "± 176661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 280338474,
            "range": "± 1033178",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 278745190,
            "range": "± 1571992",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15010369,
            "range": "± 626637",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34196270,
            "range": "± 241280",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 67750478,
            "range": "± 416911",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137221681,
            "range": "± 639436",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 361147163,
            "range": "± 11347788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359189887,
            "range": "± 1607505",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 759419329,
            "range": "± 2488516",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1591326735,
            "range": "± 3637246",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3310404885,
            "range": "± 3380166",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6984313813,
            "range": "± 15248299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383772320,
            "range": "± 2871366",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 801865129,
            "range": "± 2779209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1672680649,
            "range": "± 4350940",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3469673705,
            "range": "± 12375245",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7318440621,
            "range": "± 16088436",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 30,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 57,
            "range": "± 0",
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
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 52,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 297,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 285,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 11,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 2",
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
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7132,
            "range": "± 1053",
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
            "value": 9922,
            "range": "± 1124",
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
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "83c041d722f6401d418f487ffc095046cef3ee33",
          "message": "Pohlig Hellman Attack Example (#995)\n\n* save work. Add files\n\n* Save work\n\n* Curve 1 and curve 2 working just for small k\n\n* update ubuntu version for ci\n\n* save work\n\n* save work\n\n* new group over bls12-381\n\n* refactor\n\n* small refactor\n\n* Add Readme and remove comments\n\n* brute force test\n\n* print iterations brute force\n\n* add brute-force comment to readme\n\n* fix clippy\n\n* Update README.md\n\n* change return of function\n\n* fix clippy\n\n* change chinese theorem error\n\n* fix clippy\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\nCo-authored-by: diegokingston <dkingston@fi.uba.ar>",
          "timestamp": "2025-04-08T20:16:51Z",
          "tree_id": "77bf8329b60320dd42d600228559cef253915074",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/83c041d722f6401d418f487ffc095046cef3ee33"
        },
        "date": 1744145122407,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 337032598,
            "range": "± 341044",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 381483433,
            "range": "± 1256221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283310648,
            "range": "± 2804886",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 704873522,
            "range": "± 1115648",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 793288212,
            "range": "± 3021807",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1475980310,
            "range": "± 2772032",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1675442743,
            "range": "± 2776426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1243963546,
            "range": "± 693660",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3085603525,
            "range": "± 3928348",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3489619301,
            "range": "± 5515004",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6469173555,
            "range": "± 12242973",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7345325930,
            "range": "± 14801476",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5458799022,
            "range": "± 5324731",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7535012,
            "range": "± 4997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7589493,
            "range": "± 5072",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9885412,
            "range": "± 244086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10617170,
            "range": "± 756745",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17681854,
            "range": "± 95454",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17663075,
            "range": "± 56199",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 33282154,
            "range": "± 1042732",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 32422792,
            "range": "± 629797",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35741599,
            "range": "± 152664",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35891305,
            "range": "± 124162",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70867350,
            "range": "± 1047052",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72034422,
            "range": "± 814846",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72028951,
            "range": "± 217272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72009037,
            "range": "± 134463",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146744415,
            "range": "± 903954",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144872484,
            "range": "± 1746349",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143579239,
            "range": "± 341996",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 143657769,
            "range": "± 213450",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289237882,
            "range": "± 1526873",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 289097187,
            "range": "± 2251628",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16802329,
            "range": "± 176758",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36269099,
            "range": "± 293885",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72344885,
            "range": "± 455678",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 144187979,
            "range": "± 576753",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 378376989,
            "range": "± 3084053",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 373914343,
            "range": "± 819878",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 792624341,
            "range": "± 2955407",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1659470692,
            "range": "± 2034017",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3444698418,
            "range": "± 2100483",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7276915840,
            "range": "± 8618775",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396680461,
            "range": "± 1630153",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 825378837,
            "range": "± 2117927",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1728524137,
            "range": "± 3164005",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3598299240,
            "range": "± 6189967",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7580173498,
            "range": "± 8160892",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1118,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 143324,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 787,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 395,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1223,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 103113,
            "range": "± 6851",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2733,
            "range": "± 2190",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 198259,
            "range": "± 2451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1140,
            "range": "± 4",
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
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 8708,
            "range": "± 1263",
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
            "email": "43053772+diegokingston@users.noreply.github.com",
            "name": "Diego K",
            "username": "diegokingston"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "3e413875bb4f9b45883678cf05a6184c2fa344a7",
          "message": "Update README.md (#996)",
          "timestamp": "2025-04-09T15:03:41Z",
          "tree_id": "d9561f498cfb89b926cd67b1b68143bc679e0227",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3e413875bb4f9b45883678cf05a6184c2fa344a7"
        },
        "date": 1744212724057,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324920723,
            "range": "± 714090",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 389038579,
            "range": "± 2439642",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280658658,
            "range": "± 155014",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681160668,
            "range": "± 248528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 812031122,
            "range": "± 3655666",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1425623370,
            "range": "± 2525518",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1690466477,
            "range": "± 9088668",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233969790,
            "range": "± 2262847",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2979013463,
            "range": "± 1909816",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3537440924,
            "range": "± 16510532",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6248667893,
            "range": "± 3209709",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7418915661,
            "range": "± 21389826",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5414230850,
            "range": "± 4142559",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8224964,
            "range": "± 3977",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8327296,
            "range": "± 14214",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 15235837,
            "range": "± 613764",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 15942800,
            "range": "± 1397765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19346030,
            "range": "± 80631",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19246494,
            "range": "± 158896",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 36265454,
            "range": "± 861308",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 36820588,
            "range": "± 715920",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38403885,
            "range": "± 151439",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38858659,
            "range": "± 101715",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74550757,
            "range": "± 756336",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73759444,
            "range": "± 762117",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78384524,
            "range": "± 224747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77856418,
            "range": "± 220158",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 149600278,
            "range": "± 687720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 149696164,
            "range": "± 1468261",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153582268,
            "range": "± 474294",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 154268345,
            "range": "± 287543",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294645286,
            "range": "± 1943735",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 295609179,
            "range": "± 3411762",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18537299,
            "range": "± 896767",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36569058,
            "range": "± 1253456",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74686485,
            "range": "± 3202358",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 149105815,
            "range": "± 8062333",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 377799909,
            "range": "± 7356768",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 363425586,
            "range": "± 2688926",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 766702373,
            "range": "± 1698843",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1600311370,
            "range": "± 4699138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3335901351,
            "range": "± 8207817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7067173707,
            "range": "± 19563159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390649595,
            "range": "± 1307099",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 808225955,
            "range": "± 3076283",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1688784363,
            "range": "± 3784791",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3507734586,
            "range": "± 9986787",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7397674088,
            "range": "± 14892021",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 57,
            "range": "± 0",
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
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 50,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 420,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 396,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 33",
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 114,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11709,
            "range": "± 141",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 181,
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
            "email": "70894690+0xLucqs@users.noreply.github.com",
            "name": "0xLucqs",
            "username": "0xLucqs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28d98d12796324d4e5ac0bd65822b60044833f4f",
          "message": "feat(fft): add fast mul for fft friendly fields (#997)\n\n* feat(fft): add fast multiplication\n\n* chore: add fast_mul benchmark\n\n* impl suggestion\n\n* chore: impl suggestion",
          "timestamp": "2025-04-24T11:42:11-03:00",
          "tree_id": "6ca8fe89487eb084ce66efe5ed933d8a0469ee79",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/28d98d12796324d4e5ac0bd65822b60044833f4f"
        },
        "date": 1745507172482,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323095920,
            "range": "± 1261552",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 381310400,
            "range": "± 872291",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 284164188,
            "range": "± 4116159",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679369981,
            "range": "± 810523",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 797459457,
            "range": "± 3780285",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1423521246,
            "range": "± 5238649",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1679774753,
            "range": "± 4278875",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1245748884,
            "range": "± 700869",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2975875355,
            "range": "± 2490922",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3489698546,
            "range": "± 10616458",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6234688392,
            "range": "± 10017355",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7338377516,
            "range": "± 18712246",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5463540080,
            "range": "± 5381404",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7535564,
            "range": "± 13506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7610741,
            "range": "± 9731",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9874940,
            "range": "± 38479",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9957001,
            "range": "± 39111",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17565090,
            "range": "± 59018",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17596298,
            "range": "± 106852",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29561814,
            "range": "± 785516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28986776,
            "range": "± 838003",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35319624,
            "range": "± 141279",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35863407,
            "range": "± 150121",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69515687,
            "range": "± 1070546",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72622649,
            "range": "± 919373",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72790486,
            "range": "± 380722",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72645348,
            "range": "± 305096",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147070723,
            "range": "± 2338026",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 147494705,
            "range": "± 3491661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143530041,
            "range": "± 332839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142656895,
            "range": "± 867997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 292109333,
            "range": "± 1138652",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 291708799,
            "range": "± 3143061",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17541361,
            "range": "± 212414",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36253166,
            "range": "± 1459691",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73590100,
            "range": "± 2807304",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 150340022,
            "range": "± 2419909",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 374286957,
            "range": "± 2555151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 356441569,
            "range": "± 1384057",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 761023412,
            "range": "± 1855627",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1603843893,
            "range": "± 6280452",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3344561007,
            "range": "± 7075151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7036811867,
            "range": "± 16790565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 382696252,
            "range": "± 1571526",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 806443815,
            "range": "± 2875034",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1688585365,
            "range": "± 3490387",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3502721466,
            "range": "± 9651174",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7396696375,
            "range": "± 19519533",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 541,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 34669,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 491,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 225,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 704,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 22367,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 143710,
            "range": "± 982",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1640725,
            "range": "± 10187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1764,
            "range": "± 1524",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 64755,
            "range": "± 689",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 558,
            "range": "± 4",
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
            "value": 10454,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10177,
            "range": "± 1624",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 68,
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
            "email": "43053772+diegokingston@users.noreply.github.com",
            "name": "Diego K",
            "username": "diegokingston"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "7a336ca3ba8d64838b443bd8efca463c8f9c73b9",
          "message": "Improve documentation (#999)\n\n* add explanation\n\n* fix format\n\n* add more comments\n\n* explain fft\n\n* continue docs\n\n* add more docs\n\n* Update README.md",
          "timestamp": "2025-04-25T13:14:46Z",
          "tree_id": "022c7aafee18c3201d20dc2f72866b2976de4463",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7a336ca3ba8d64838b443bd8efca463c8f9c73b9"
        },
        "date": 1745588590282,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325167512,
            "range": "± 357543",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 383158347,
            "range": "± 1617891",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283667452,
            "range": "± 673043",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681873524,
            "range": "± 709042",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 800844884,
            "range": "± 2440366",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1426885051,
            "range": "± 2199439",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1685459138,
            "range": "± 6215426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1245741005,
            "range": "± 1225876",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2979830104,
            "range": "± 2748076",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3512155368,
            "range": "± 11844782",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6243942783,
            "range": "± 3071006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7412213765,
            "range": "± 12457020",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5459712657,
            "range": "± 4943137",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8290321,
            "range": "± 4696",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8370760,
            "range": "± 3293",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14170538,
            "range": "± 818168",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 13933441,
            "range": "± 1034893",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19307683,
            "range": "± 59690",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19274290,
            "range": "± 88828",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 36390917,
            "range": "± 581987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 35852131,
            "range": "± 584930",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38590353,
            "range": "± 97914",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38887320,
            "range": "± 154665",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74765705,
            "range": "± 999177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 75525460,
            "range": "± 869899",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78438497,
            "range": "± 366840",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 78001815,
            "range": "± 258709",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 151883276,
            "range": "± 708111",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151727353,
            "range": "± 1254970",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 154930578,
            "range": "± 417839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 154533924,
            "range": "± 739885",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 300723530,
            "range": "± 2506367",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 301823566,
            "range": "± 1642341",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18338046,
            "range": "± 410092",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 37397722,
            "range": "± 478214",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74264302,
            "range": "± 1059452",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 150819057,
            "range": "± 2524344",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 381178755,
            "range": "± 1610092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367148607,
            "range": "± 1102936",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 774064122,
            "range": "± 2943004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612194695,
            "range": "± 2441785",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3354505499,
            "range": "± 6630940",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7088948617,
            "range": "± 8311963",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396038895,
            "range": "± 1603365",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 818248907,
            "range": "± 2089020",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1700098707,
            "range": "± 4653184",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3512063378,
            "range": "± 10489721",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7399620535,
            "range": "± 8445734",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 411,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 106,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 178,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 525,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 143348,
            "range": "± 925",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1634355,
            "range": "± 19974",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 539,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2030,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 50,
            "range": "± 0",
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
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7560,
            "range": "± 423",
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
            "email": "76252340+MarcosNicolau@users.noreply.github.com",
            "name": "Marcos Nicolau",
            "username": "MarcosNicolau"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5f8f2cfcc8a1a22f77e8dff2d581f1166eefb80b",
          "message": "refactor: remove default contraint in merkle tree trait (#1004)",
          "timestamp": "2025-05-05T17:24:46Z",
          "tree_id": "99bb8089f5ac9fb827806203d60745320423ba07",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5f8f2cfcc8a1a22f77e8dff2d581f1166eefb80b"
        },
        "date": 1746467642382,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 322695457,
            "range": "± 611075",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 379666224,
            "range": "± 1708063",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283159377,
            "range": "± 280436",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 678621383,
            "range": "± 1024919",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 801010091,
            "range": "± 3717688",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1422804968,
            "range": "± 1623345",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1685157329,
            "range": "± 4821781",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1248506328,
            "range": "± 1446382",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2970836960,
            "range": "± 4518248",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3511009894,
            "range": "± 13818676",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6218981042,
            "range": "± 2679661",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7348890928,
            "range": "± 45362648",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5471409863,
            "range": "± 5267401",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7547554,
            "range": "± 3567",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7613059,
            "range": "± 17813",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11110302,
            "range": "± 986468",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10456885,
            "range": "± 935224",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17720959,
            "range": "± 117143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17509571,
            "range": "± 60419",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26884118,
            "range": "± 3893736",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27320635,
            "range": "± 1262458",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35282223,
            "range": "± 88192",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35364523,
            "range": "± 263505",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70209580,
            "range": "± 2554368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68429936,
            "range": "± 3239328",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71839950,
            "range": "± 358314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71613864,
            "range": "± 166352",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 143328198,
            "range": "± 2573163",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144476560,
            "range": "± 2732008",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 142982342,
            "range": "± 358916",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142746211,
            "range": "± 204361",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 293947899,
            "range": "± 5164140",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 285371374,
            "range": "± 2739342",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15455079,
            "range": "± 1100988",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34623605,
            "range": "± 876922",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 79039795,
            "range": "± 2884411",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 155956312,
            "range": "± 4109799",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 386660504,
            "range": "± 4930568",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367288520,
            "range": "± 5024959",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 761519817,
            "range": "± 4583372",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1602598212,
            "range": "± 5446278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3339587740,
            "range": "± 9652549",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7032887949,
            "range": "± 15356491",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 381480913,
            "range": "± 1204371",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 800032601,
            "range": "± 4976345",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1679026410,
            "range": "± 3301481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3498004477,
            "range": "± 10242867",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7363066169,
            "range": "± 23772139",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 260,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8397,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 288,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 174,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 410,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 7117,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142690,
            "range": "± 370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1643673,
            "range": "± 9220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1297,
            "range": "± 1151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22317,
            "range": "± 451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 266,
            "range": "± 1",
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
            "value": 76,
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
            "value": 10357,
            "range": "± 1017",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 47,
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
          "id": "c9d02b585f9dafee732eb6f2e30b7028b546e79e",
          "message": "Add big uint conversion (#1002)\n\n* first iteration to add BigUint conversion\n\n* remove comments\n\n* refactor\n\n* refactor\n\n* save work. tests working for different fields\n\n* small refactor\n\n* conversion not working for baby bear u32\n\n* all tests working for every field\n\n* remove comments in u64_prime_field file\n\n* remove commented code\n\n* fix cargo check no-default-features\n\n* remove changes in unsigned_integer\n\n* fix ensure-no-std\n\n* remove Ok and unwrap\n\n* fix ci set up job\n\n* fix compile error\n\n* add BN254 conversion test\n\n* change test for bn254\n\n---------\n\nCo-authored-by: Nicole <nicole@Nicoles-MacBook-Air.local>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-05-09T17:05:53-03:00",
          "tree_id": "411624d8e8ce1278066bf25ab94b6ec368be30ab",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c9d02b585f9dafee732eb6f2e30b7028b546e79e"
        },
        "date": 1746822596607,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323451059,
            "range": "± 268299",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 376807613,
            "range": "± 1313548",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282823402,
            "range": "± 4229959",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679629634,
            "range": "± 388299",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 793103466,
            "range": "± 1689426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1423896294,
            "range": "± 4353360",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1667357554,
            "range": "± 2933823",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1243888081,
            "range": "± 1182913",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2981132872,
            "range": "± 14664835",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3491020927,
            "range": "± 12755257",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6237050535,
            "range": "± 3669578",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7378530124,
            "range": "± 8954215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5455713687,
            "range": "± 3948262",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8283673,
            "range": "± 12800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8365016,
            "range": "± 10269",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10708471,
            "range": "± 263766",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10559195,
            "range": "± 84566",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18851283,
            "range": "± 78314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18851605,
            "range": "± 91511",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29964823,
            "range": "± 1297784",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29475991,
            "range": "± 1098928",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37847179,
            "range": "± 78574",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38543573,
            "range": "± 123975",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70124996,
            "range": "± 889107",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71117875,
            "range": "± 1337717",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78066603,
            "range": "± 191719",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77446273,
            "range": "± 103521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147490149,
            "range": "± 1618457",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 148394439,
            "range": "± 867443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153641567,
            "range": "± 404252",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152971383,
            "range": "± 191232",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294984697,
            "range": "± 1790160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 295303295,
            "range": "± 1052874",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16497744,
            "range": "± 258775",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35447765,
            "range": "± 817360",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71884891,
            "range": "± 759761",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 145665541,
            "range": "± 2061222",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 362358587,
            "range": "± 3427299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357871222,
            "range": "± 1944202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 762395212,
            "range": "± 1794992",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1598158132,
            "range": "± 2698045",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3342636358,
            "range": "± 6681481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7032456476,
            "range": "± 8214333",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 384676977,
            "range": "± 1797321",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 800995608,
            "range": "± 2764245",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1684298581,
            "range": "± 3068358",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3480717808,
            "range": "± 3908788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7347177147,
            "range": "± 15737995",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 114,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1785,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 189,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 103,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 306,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1664,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 143733,
            "range": "± 348",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1634228,
            "range": "± 13567",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 838,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7191,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 119,
            "range": "± 1",
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 133,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11765,
            "range": "± 1220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 192,
            "range": "± 3",
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
            "email": "70894690+0xLucqs@users.noreply.github.com",
            "name": "0xLucqs",
            "username": "0xLucqs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3a04406abfa059bd5688dd0622188d516b9207bd",
          "message": "Feat/fast division (#1001)\n\n* feat(fft): add fast multiplication\n\n* chore: add fast_mul benchmark\n\n* impl suggestion\n\n* chore: impl suggestion\n\n* chore: add fast_mul benchmark\n\n* feat(fft): fast division\n\n* refacto: proper error handling\n\n* refacto: rename inversion function\n\n* test: add tests for fast division helper functions\n\n* doc: explain where fast division is taken from\n\n---------\n\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-05-15T11:21:27-03:00",
          "tree_id": "a401827c78813d6302976cb93c864f92ff017538",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3a04406abfa059bd5688dd0622188d516b9207bd"
        },
        "date": 1747320346474,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323651543,
            "range": "± 1347806",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 377760438,
            "range": "± 5438045",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 281258420,
            "range": "± 414580",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 680871017,
            "range": "± 1593472",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 794346637,
            "range": "± 3680509",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1426878837,
            "range": "± 2642264",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1664871175,
            "range": "± 7061273",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1234675450,
            "range": "± 781125",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2980746943,
            "range": "± 2749213",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3472252181,
            "range": "± 15622923",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6241157860,
            "range": "± 6483124",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7328442952,
            "range": "± 21082462",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5418122047,
            "range": "± 3827502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8294349,
            "range": "± 10955",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8324635,
            "range": "± 7575",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10548718,
            "range": "± 565785",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10565931,
            "range": "± 85847",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19237325,
            "range": "± 99834",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19090906,
            "range": "± 87802",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 30147093,
            "range": "± 889764",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29569539,
            "range": "± 698940",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38045423,
            "range": "± 412428",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38700087,
            "range": "± 111030",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70792684,
            "range": "± 1858007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71763221,
            "range": "± 1104188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 77901976,
            "range": "± 149738",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77577288,
            "range": "± 70800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 151523400,
            "range": "± 2341193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151679560,
            "range": "± 5131143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153066959,
            "range": "± 328225",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152323824,
            "range": "± 158579",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 298841401,
            "range": "± 3845369",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 297299902,
            "range": "± 4046873",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16843391,
            "range": "± 371024",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36288315,
            "range": "± 661660",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73578024,
            "range": "± 1583633",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 151147337,
            "range": "± 3069945",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 384451913,
            "range": "± 2802015",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 360296155,
            "range": "± 2186425",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764778711,
            "range": "± 2073004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1607611338,
            "range": "± 3010162",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3352624023,
            "range": "± 15148626",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7078324717,
            "range": "± 15005930",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385570603,
            "range": "± 1541553",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 805398869,
            "range": "± 7281180",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1693593662,
            "range": "± 3103706",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3510503667,
            "range": "± 14395872",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7377033939,
            "range": "± 11796395",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 117,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1903,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 178,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 116,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 301,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1642,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144007,
            "range": "± 403",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1571401,
            "range": "± 99375",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 810686,
            "range": "± 5528",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1639385,
            "range": "± 24018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 872,
            "range": "± 378",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7335,
            "range": "± 117",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 121,
            "range": "± 0",
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
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 128,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11837,
            "range": "± 1418",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 185,
            "range": "± 3",
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
          "id": "055dc6f600786dd49f4ff49383fb94ff5f629992",
          "message": "Add quadratic and cubic sumcheck (#1003)\n\n* first commit, add quadratic and cubic\n\n* fix quadratic and cubic sumcheck implementation\n\n* refactor code to use wrappers\n\n* save work\n\n* small refactor\n\n* implement suggestion\n\n* add number of factors and number of variables to transcript\n\n* fix clippy\n\n* fix clippy\n\n---------\n\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-05-23T21:05:22Z",
          "tree_id": "d7ece2de75dabfc66391f1d65ae7ac0e5f588916",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/055dc6f600786dd49f4ff49383fb94ff5f629992"
        },
        "date": 1748036131031,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326496974,
            "range": "± 438535",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370473084,
            "range": "± 2057600",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279833211,
            "range": "± 249176",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685084167,
            "range": "± 1356756",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778619601,
            "range": "± 1773755",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1436109821,
            "range": "± 750294",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1642109086,
            "range": "± 5425699",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1230921589,
            "range": "± 6874242",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3003171093,
            "range": "± 3646826",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3434759053,
            "range": "± 11654593",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6281247560,
            "range": "± 6053618",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7255101011,
            "range": "± 31926338",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5391957695,
            "range": "± 2193662",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7982502,
            "range": "± 9125",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7988857,
            "range": "± 6945",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10760900,
            "range": "± 461432",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10239695,
            "range": "± 62862",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18148284,
            "range": "± 77625",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18180202,
            "range": "± 48963",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27906144,
            "range": "± 1798682",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27001470,
            "range": "± 509727",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37256775,
            "range": "± 532440",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37008792,
            "range": "± 143783",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67117391,
            "range": "± 544298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67528725,
            "range": "± 1092948",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74617386,
            "range": "± 140251",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74609511,
            "range": "± 300154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 140176752,
            "range": "± 1681715",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142683585,
            "range": "± 3404708",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151740687,
            "range": "± 159612",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151544508,
            "range": "± 207066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 293685774,
            "range": "± 4258666",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 289174893,
            "range": "± 3285835",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15739747,
            "range": "± 294617",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33651440,
            "range": "± 643980",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 68485500,
            "range": "± 962095",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 136081483,
            "range": "± 1168278",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 357297181,
            "range": "± 4718925",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 356095104,
            "range": "± 926830",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 763605224,
            "range": "± 2668117",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1608294675,
            "range": "± 2081607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3349213751,
            "range": "± 5320388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7084137048,
            "range": "± 8029157",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389087602,
            "range": "± 1085388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 808083947,
            "range": "± 1913828",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1697281784,
            "range": "± 6104393",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3509655096,
            "range": "± 6328631",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7393626168,
            "range": "± 17853875",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 115,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1840,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 173,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 104,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 311,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1570,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144982,
            "range": "± 1136",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1547321,
            "range": "± 5389",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 815367,
            "range": "± 1753",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1672401,
            "range": "± 8214",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 807,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7399,
            "range": "± 233",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 119,
            "range": "± 0",
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
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 6863,
            "range": "± 695",
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
          "id": "ad2a380641758cfbbe2de6f4a2d8e27684f0fed7",
          "message": "fix clippy for new rust version (#1013)",
          "timestamp": "2025-08-01T21:05:21Z",
          "tree_id": "f84f62fc85c1e1ea5772038679bacf93f88bdb68",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ad2a380641758cfbbe2de6f4a2d8e27684f0fed7"
        },
        "date": 1754084121312,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326255928,
            "range": "± 939803",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369905715,
            "range": "± 594165",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279918677,
            "range": "± 517863",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685061449,
            "range": "± 401403",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784183788,
            "range": "± 1889981",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1435415520,
            "range": "± 471741",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1652110374,
            "range": "± 10000424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1229965637,
            "range": "± 637220",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3000409988,
            "range": "± 1215925",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3462072077,
            "range": "± 23803410",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6282392621,
            "range": "± 3316907",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7302940268,
            "range": "± 51380774",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5401374650,
            "range": "± 1015956",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7984778,
            "range": "± 5238",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8002408,
            "range": "± 8754",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10117812,
            "range": "± 143285",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10105146,
            "range": "± 171507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18284352,
            "range": "± 43257",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18263368,
            "range": "± 126630",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28601307,
            "range": "± 676882",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28027605,
            "range": "± 549105",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37646908,
            "range": "± 104867",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37105755,
            "range": "± 74237",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 73362296,
            "range": "± 2625624",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70230220,
            "range": "± 2546897",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74917747,
            "range": "± 221190",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74634070,
            "range": "± 124648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 144976371,
            "range": "± 1219601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144467825,
            "range": "± 1143031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153115114,
            "range": "± 660368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153397623,
            "range": "± 1002267",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294342794,
            "range": "± 1591177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 297526550,
            "range": "± 3486689",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16776380,
            "range": "± 143093",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35856901,
            "range": "± 441807",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72059970,
            "range": "± 1421281",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 140824082,
            "range": "± 890820",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 368940861,
            "range": "± 2559109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359683099,
            "range": "± 1662306",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 765859213,
            "range": "± 1812981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1610462797,
            "range": "± 2047693",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3353514823,
            "range": "± 1999644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7084306723,
            "range": "± 4945241",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 386847105,
            "range": "± 4092238",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 808214464,
            "range": "± 2599220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1700979431,
            "range": "± 7115410",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3550286267,
            "range": "± 10650763",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7463239684,
            "range": "± 28967920",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 57,
            "range": "± 0",
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
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 53,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 145307,
            "range": "± 729",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1545362,
            "range": "± 7383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 816282,
            "range": "± 2472",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1744462,
            "range": "± 15641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 293,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 291,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 36",
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
            "value": 10408,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10176,
            "range": "± 477",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 69,
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
            "email": "pdeymon@fi.uba.ar",
            "name": "Pablo Deymonnaz",
            "username": "pablodeymo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b018546d485d2183229af08050da6828e44727f",
          "message": "Fix bincode repo change (#1023)\n\n* Fix bincode inclusion\n\n* Add feature serde\n\n* fix clippy\n\n* fix warnings in benches and solve bincode dependency\n\n* use crates io for bincode\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2025-09-04T10:39:44-03:00",
          "tree_id": "92dc6ad2e4737096fc389a707999cd69ee64d351",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4b018546d485d2183229af08050da6828e44727f"
        },
        "date": 1756994647048,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325991755,
            "range": "± 681598",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 374448513,
            "range": "± 3757645",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277883420,
            "range": "± 2144164",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685338373,
            "range": "± 467234",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 805259952,
            "range": "± 4998892",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1436433999,
            "range": "± 1251909",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1679420004,
            "range": "± 12453102",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223299184,
            "range": "± 4274812",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3001160329,
            "range": "± 2126221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3496676362,
            "range": "± 28102858",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6285745381,
            "range": "± 3353500",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7399146270,
            "range": "± 26025662",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5390031430,
            "range": "± 5988611",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7975984,
            "range": "± 4911",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7995500,
            "range": "± 5253",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10100023,
            "range": "± 36995",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10340819,
            "range": "± 438263",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18150525,
            "range": "± 74325",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18224592,
            "range": "± 107672",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27015866,
            "range": "± 1754348",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26961971,
            "range": "± 945042",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37349926,
            "range": "± 69767",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37078828,
            "range": "± 73917",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71806229,
            "range": "± 810077",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71588700,
            "range": "± 963589",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75228231,
            "range": "± 309974",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75085703,
            "range": "± 131621",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 153828378,
            "range": "± 3316296",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151642432,
            "range": "± 2680587",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153118461,
            "range": "± 366388",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153301852,
            "range": "± 228070",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 309341257,
            "range": "± 5463317",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 314997063,
            "range": "± 6437859",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18671865,
            "range": "± 714154",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 39585478,
            "range": "± 1104319",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 81110671,
            "range": "± 3985246",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 159675731,
            "range": "± 4888886",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 396307028,
            "range": "± 4988328",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357561710,
            "range": "± 1809150",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 768588543,
            "range": "± 3089943",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1623667653,
            "range": "± 5221858",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3396026948,
            "range": "± 5832406",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7144512983,
            "range": "± 7067355",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385163105,
            "range": "± 2373397",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 811093379,
            "range": "± 2634027",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1705919182,
            "range": "± 4690839",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3540674157,
            "range": "± 9615678",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7463924191,
            "range": "± 17757604",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 51,
            "range": "± 0",
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
            "value": 75,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 145308,
            "range": "± 298",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1543587,
            "range": "± 14118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 814917,
            "range": "± 1820",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1713758,
            "range": "± 33265",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 251,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 44,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 60",
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
            "value": 10334,
            "range": "± 25",
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
            "value": 10157,
            "range": "± 961",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 64,
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
            "value": 9727,
            "range": "± 1211",
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
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "575c4a208e4182fa5998965808b953f512124cb7",
          "message": "Add GKR Protocol (#1011)\n\n* save work, protocol structure now is following the post style\n\n* add prints to debug implementation\n\n* refactor to avoid unwraps\n\n* more refactor\n\n* more refactor\n\n* refactor\n\n* add readme\n\n* Fix readme. Add struct Prover\n\n* remove extra cargo.toml\n\n* remove claimed_sum from sumcheck_proof. Fix test. Add documentation\n\n* remove clones and fix clippy\n\n* add documentation\n\n* add check mark for gkr in readme\n\n* add degree check on g_j\n\n* add verifier checks: proof structure match circuit structure\n\n* fix clippy for new rust version (#1013)\n\n* fix clippy\n\n* fix readme\n\n* rename modulus\n\n* avoid repetitive computation\n\n* check terms has len 2\n\n* remove clone term_1 and term_2\n\n* add check number of inputs\n\n* fix type complexity\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: jotabulacios <45471455+jotabulacios@users.noreply.github.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-09-04T17:41:15Z",
          "tree_id": "4a0dbcbe5f6bff84ac8a4a145fdbd56730252eb7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/575c4a208e4182fa5998965808b953f512124cb7"
        },
        "date": 1757009404290,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327349357,
            "range": "± 225229",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369173127,
            "range": "± 944363",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278358638,
            "range": "± 3996838",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 686881448,
            "range": "± 450374",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 777632938,
            "range": "± 1958864",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1439667581,
            "range": "± 6619044",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1632517436,
            "range": "± 6433470",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1221655289,
            "range": "± 937573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3007052303,
            "range": "± 7970711",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3424299583,
            "range": "± 4418343",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6293588909,
            "range": "± 2019723",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7240206657,
            "range": "± 21137611",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5357256651,
            "range": "± 10105561",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8005486,
            "range": "± 4484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8005962,
            "range": "± 12041",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10621194,
            "range": "± 98679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10575369,
            "range": "± 67678",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19365209,
            "range": "± 125406",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19234876,
            "range": "± 103713",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 32380989,
            "range": "± 1786760",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30467225,
            "range": "± 1327679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38214504,
            "range": "± 49895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38218622,
            "range": "± 83004",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71712421,
            "range": "± 1102960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69059180,
            "range": "± 523308",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75501772,
            "range": "± 353384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75331493,
            "range": "± 94651",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 141133423,
            "range": "± 1640193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142705663,
            "range": "± 566601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151238136,
            "range": "± 228972",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151140655,
            "range": "± 98154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 285776897,
            "range": "± 1171319",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 289713741,
            "range": "± 700536",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15281536,
            "range": "± 358394",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33943059,
            "range": "± 191313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 68260684,
            "range": "± 330574",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137590252,
            "range": "± 1099558",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 350515501,
            "range": "± 2421887",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 362356571,
            "range": "± 844229",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 769775233,
            "range": "± 2815799",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612756789,
            "range": "± 2208520",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3354180619,
            "range": "± 7685594",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7081259770,
            "range": "± 10563601",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389717837,
            "range": "± 2412829",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 810352827,
            "range": "± 2141090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1691467127,
            "range": "± 4272275",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3509554173,
            "range": "± 2748299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7386491497,
            "range": "± 5383268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 242,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7794,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 278,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 195,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 466,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6462,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 145173,
            "range": "± 193",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578950,
            "range": "± 5956",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 816826,
            "range": "± 1654",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1675230,
            "range": "± 12407",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1245,
            "range": "± 1077",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22756,
            "range": "± 313",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 249,
            "range": "± 0",
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
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7236,
            "range": "± 525",
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
            "value": 9669,
            "range": "± 262",
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
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "160bb5ca435696a41387c8cd95ad1f451e497a73",
          "message": "Fix clippy (#1029)\n\n* fix clippy\n\n* remove unused import\n\n* fix",
          "timestamp": "2025-09-22T13:55:28Z",
          "tree_id": "35c4006d18ca2209259c1fd0ec20c1ae7f75ef99",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/160bb5ca435696a41387c8cd95ad1f451e497a73"
        },
        "date": 1758551006111,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325335444,
            "range": "± 125867",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369983647,
            "range": "± 985942",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277303546,
            "range": "± 165830",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684694098,
            "range": "± 678642",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778176882,
            "range": "± 1775762",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1434791074,
            "range": "± 2940528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1638901678,
            "range": "± 4906497",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1220013815,
            "range": "± 2853602",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2995143977,
            "range": "± 3485241",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3421099090,
            "range": "± 7458792",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6266438925,
            "range": "± 3331942",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7254859249,
            "range": "± 10861104",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5358648266,
            "range": "± 7326848",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8168692,
            "range": "± 5196",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8179494,
            "range": "± 5922",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10447423,
            "range": "± 332621",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11518927,
            "range": "± 843071",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18489389,
            "range": "± 69462",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18383582,
            "range": "± 84599",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27534711,
            "range": "± 250836",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27370098,
            "range": "± 321637",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37211194,
            "range": "± 116104",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37496783,
            "range": "± 174741",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69841715,
            "range": "± 414266",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69643673,
            "range": "± 1306146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75099049,
            "range": "± 132341",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75235898,
            "range": "± 171668",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146259185,
            "range": "± 813673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 147469116,
            "range": "± 992128",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151743604,
            "range": "± 173893",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151569894,
            "range": "± 245275",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 295296106,
            "range": "± 1103585",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 294863150,
            "range": "± 429504",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17265704,
            "range": "± 263477",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36519140,
            "range": "± 366020",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74608768,
            "range": "± 5225600",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 146583199,
            "range": "± 751191",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 380952927,
            "range": "± 2779776",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 363822408,
            "range": "± 2322157",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 774116277,
            "range": "± 2971178",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1614958369,
            "range": "± 6043447",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3360972770,
            "range": "± 2582537",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7082677363,
            "range": "± 8344820",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390025306,
            "range": "± 1313770",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 811466603,
            "range": "± 2131396",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1697281333,
            "range": "± 889175",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3513056901,
            "range": "± 11513542",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7407719018,
            "range": "± 5551583",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 541,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 34591,
            "range": "± 379",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 436,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 237,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 743,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 27471,
            "range": "± 202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149369,
            "range": "± 332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568268,
            "range": "± 7377",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837200,
            "range": "± 1554",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1736615,
            "range": "± 10807",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1677,
            "range": "± 1405",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 68220,
            "range": "± 825",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 556,
            "range": "± 6",
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
          "id": "060dbf6af6e0c293e80f143fc84c01df47984304",
          "message": "Add new unchecked (#1024)\n\n* add new_uncheckes for short weiestrass\n\n* remove unused files\n\n* use unchecked for stark curve generator\n\n* revert changes for vesta curve\n\n* add missing comment\n\n* make the point field private\n\n* fix clippy\n\n* add safety comments and rename function\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>",
          "timestamp": "2025-09-26T12:41:53Z",
          "tree_id": "58fde0f7f49b7bb1c20f6adb2b69cfd487ca2a32",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/060dbf6af6e0c293e80f143fc84c01df47984304"
        },
        "date": 1758892260395,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325215580,
            "range": "± 847396",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373657068,
            "range": "± 2434340",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278297585,
            "range": "± 216465",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682379484,
            "range": "± 1547866",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 785330520,
            "range": "± 5010239",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1430639896,
            "range": "± 2018894",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1641789202,
            "range": "± 4857329",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223414376,
            "range": "± 926179",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2990728001,
            "range": "± 1673318",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3462773251,
            "range": "± 27530512",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6254017292,
            "range": "± 6648237",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7276449166,
            "range": "± 28044908",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5363226478,
            "range": "± 2129329",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7802390,
            "range": "± 7737",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7855133,
            "range": "± 3591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12414636,
            "range": "± 329671",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12445642,
            "range": "± 406244",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18406185,
            "range": "± 25581",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18406602,
            "range": "± 26824",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28133954,
            "range": "± 1611770",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 25844952,
            "range": "± 1778691",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36405996,
            "range": "± 273027",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36731186,
            "range": "± 194837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71598287,
            "range": "± 2491460",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73062654,
            "range": "± 2116195",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74007339,
            "range": "± 182959",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73830442,
            "range": "± 290963",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145939467,
            "range": "± 2058601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144493665,
            "range": "± 1143129",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 147382138,
            "range": "± 149852",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 147205967,
            "range": "± 179827",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289140208,
            "range": "± 1155299",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 296248833,
            "range": "± 2802542",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16989761,
            "range": "± 659266",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35842569,
            "range": "± 462530",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69181791,
            "range": "± 547365",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143995557,
            "range": "± 1552718",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 378716909,
            "range": "± 5075004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 355185973,
            "range": "± 4450696",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764245901,
            "range": "± 5096431",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612145494,
            "range": "± 3757159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3339393239,
            "range": "± 3569650",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7105842097,
            "range": "± 14807527",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 382037081,
            "range": "± 1024800",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 804546863,
            "range": "± 1461616",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1700177771,
            "range": "± 3290557",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3544380419,
            "range": "± 8550266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7422766439,
            "range": "± 23445052",
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
            "value": 94,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 63,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 91,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 165,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149398,
            "range": "± 235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1577888,
            "range": "± 15509",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837095,
            "range": "± 1089",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1750395,
            "range": "± 19949",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 266,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 646,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 22,
            "range": "± 0",
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
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7157,
            "range": "± 647",
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
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "5c451b961aacd957e6d837f23550cfa3144241d0",
          "message": "Release 0.13.0 (#1032)\n\n* update version\n\n* add publish workflow\n\n* update dependencies versions\n\n* fix name\n\n* add --workspace",
          "timestamp": "2025-09-26T22:11:09Z",
          "tree_id": "f6fa21ee6bc287b148793d5475cf7feb52221341",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5c451b961aacd957e6d837f23550cfa3144241d0"
        },
        "date": 1758926412276,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325963016,
            "range": "± 3399029",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 371738149,
            "range": "± 1561326",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278814470,
            "range": "± 7929519",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 683930195,
            "range": "± 678225",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784875772,
            "range": "± 4714139",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1431549352,
            "range": "± 866932",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1640861027,
            "range": "± 5503930",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224447072,
            "range": "± 1196037",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993522190,
            "range": "± 11282622",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3412001808,
            "range": "± 15221984",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6259757545,
            "range": "± 8681837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7226452713,
            "range": "± 8290036",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5365975769,
            "range": "± 2925741",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7806927,
            "range": "± 5786",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7852937,
            "range": "± 20188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11733822,
            "range": "± 1335452",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11315285,
            "range": "± 278017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18110453,
            "range": "± 49818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18093574,
            "range": "± 40461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 30587502,
            "range": "± 1034200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30614571,
            "range": "± 767048",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36459325,
            "range": "± 155532",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36610808,
            "range": "± 46152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70282343,
            "range": "± 726581",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70748278,
            "range": "± 952778",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73801286,
            "range": "± 184280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73530496,
            "range": "± 78973",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 143704585,
            "range": "± 469241",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143309730,
            "range": "± 405695",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 146940510,
            "range": "± 221954",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 146619131,
            "range": "± 231523",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 286076963,
            "range": "± 1299961",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 286364706,
            "range": "± 821430",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16264109,
            "range": "± 257096",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35221517,
            "range": "± 172517",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71058491,
            "range": "± 338395",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 140004925,
            "range": "± 622099",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 367674065,
            "range": "± 818266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 361810685,
            "range": "± 1080391",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 769739185,
            "range": "± 10838758",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1607312731,
            "range": "± 1350613",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3343970614,
            "range": "± 9667806",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7069251321,
            "range": "± 8776698",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 388927278,
            "range": "± 1511521",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 812653115,
            "range": "± 2123301",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699394986,
            "range": "± 1993612",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3500682480,
            "range": "± 3594168",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7373802937,
            "range": "± 5009223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 30,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 57,
            "range": "± 0",
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
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 51,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149089,
            "range": "± 376",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1553940,
            "range": "± 7253",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 834314,
            "range": "± 1458",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1688453,
            "range": "± 17068",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 269,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 269,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 38",
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
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7159,
            "range": "± 985",
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
            "value": 9754,
            "range": "± 121",
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
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "980313e6e5d3962aa15cbfbe9c73e534af2dfc82",
          "message": "fix publish workflow (#1033)",
          "timestamp": "2025-09-29T14:07:47Z",
          "tree_id": "1715dc3b4f1804b05b1bdbb785bf06720cfd136f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/980313e6e5d3962aa15cbfbe9c73e534af2dfc82"
        },
        "date": 1759156644327,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325276909,
            "range": "± 299122",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373799212,
            "range": "± 1280792",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 276761830,
            "range": "± 3240462",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684123108,
            "range": "± 458835",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784548181,
            "range": "± 860109",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432390908,
            "range": "± 2789501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1648173519,
            "range": "± 1888834",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1217575532,
            "range": "± 3284841",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993699711,
            "range": "± 719515",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3440730954,
            "range": "± 10184483",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6268181109,
            "range": "± 3354598",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7305123303,
            "range": "± 13595186",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5340979849,
            "range": "± 11014483",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7792357,
            "range": "± 26787",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7845413,
            "range": "± 4191",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9905692,
            "range": "± 54020",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9935119,
            "range": "± 37029",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17946159,
            "range": "± 72651",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17854317,
            "range": "± 29782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27077437,
            "range": "± 1066053",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26842568,
            "range": "± 504628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36099204,
            "range": "± 25086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36296376,
            "range": "± 29384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67316601,
            "range": "± 696305",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67900197,
            "range": "± 610414",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73460960,
            "range": "± 125364",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73239280,
            "range": "± 97542",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 142108141,
            "range": "± 990596",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143959311,
            "range": "± 2673369",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 146531409,
            "range": "± 321114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 146469282,
            "range": "± 105839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 284632593,
            "range": "± 945734",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 283681688,
            "range": "± 1164461",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15729115,
            "range": "± 66530",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35615464,
            "range": "± 669731",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70791379,
            "range": "± 556050",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 139936815,
            "range": "± 910885",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 365484553,
            "range": "± 5490571",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 361663058,
            "range": "± 2990451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 762404414,
            "range": "± 2192560",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1604253640,
            "range": "± 3479460",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3343212740,
            "range": "± 3525470",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7061624679,
            "range": "± 9724140",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385896349,
            "range": "± 2498005",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 805836010,
            "range": "± 2063810",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1693259461,
            "range": "± 1272833",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3502638453,
            "range": "± 9288862",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7379752116,
            "range": "± 5243622",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 56,
            "range": "± 0",
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
            "value": 84,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 50,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149332,
            "range": "± 202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576574,
            "range": "± 14090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837195,
            "range": "± 3444",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1749773,
            "range": "± 5302",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 279,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 278,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 9,
            "range": "± 37",
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
            "value": 10383,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 92,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10130,
            "range": "± 952",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 62,
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
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f03bf1ba13c4207faafb5b0c639b68f8b60d55f",
          "message": "Remove all-features in publish (#1034)\n\n* remove all-features in publish\n\n* rm mac specific features",
          "timestamp": "2025-09-29T15:41:05Z",
          "tree_id": "3627df9a79145407673b45e92a794163a751866f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6f03bf1ba13c4207faafb5b0c639b68f8b60d55f"
        },
        "date": 1759162230136,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326052587,
            "range": "± 327303",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 378794070,
            "range": "± 1060771",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279230377,
            "range": "± 6382372",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684888426,
            "range": "± 7314944",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 797613594,
            "range": "± 9744927",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1433431586,
            "range": "± 645093",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1667154408,
            "range": "± 11292086",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1221477059,
            "range": "± 6207384",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993403094,
            "range": "± 1566266",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3480509202,
            "range": "± 21420319",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6276792127,
            "range": "± 11290789",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7341362876,
            "range": "± 14007302",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5353857536,
            "range": "± 12102265",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7800649,
            "range": "± 20792",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7867376,
            "range": "± 97685",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14969400,
            "range": "± 510732",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 15421656,
            "range": "± 388852",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18548995,
            "range": "± 65264",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18477744,
            "range": "± 49657",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 35407419,
            "range": "± 369150",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 35945733,
            "range": "± 254154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37047255,
            "range": "± 371682",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37149369,
            "range": "± 905982",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 72676056,
            "range": "± 785625",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72988400,
            "range": "± 220160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74992312,
            "range": "± 118634",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74678301,
            "range": "± 76627",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146134758,
            "range": "± 419783",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144749031,
            "range": "± 361141",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 148144466,
            "range": "± 506503",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 148525673,
            "range": "± 175507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 290099351,
            "range": "± 1499140",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 291142433,
            "range": "± 1259193",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18126240,
            "range": "± 529617",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36343143,
            "range": "± 936185",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70890838,
            "range": "± 746462",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 144304341,
            "range": "± 938313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 377452695,
            "range": "± 7593715",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 370578339,
            "range": "± 875132",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 776197736,
            "range": "± 762739",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1613468970,
            "range": "± 1173146",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3355807362,
            "range": "± 2133974",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7078061283,
            "range": "± 5411817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 397337274,
            "range": "± 3375303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 814209726,
            "range": "± 19613983",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1696964267,
            "range": "± 17870997",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3508199492,
            "range": "± 8024534",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7379264858,
            "range": "± 23020004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 257,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8230,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 287,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 184,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 439,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6177,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149213,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1554409,
            "range": "± 11862",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 836692,
            "range": "± 1413",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1658194,
            "range": "± 9165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1255,
            "range": "± 1086",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22172,
            "range": "± 1776",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 266,
            "range": "± 15",
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
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 78,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10080,
            "range": "± 197",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 38,
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
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d9c3c04b61cda15e4b416f00ee8be3594b2b5708",
          "message": "Add Plonky3 in the Readme Table (#1035)\n\n* add plonky3\n\n* add circle for plonky3\n\n* exchange columns hal2 and plonky3\n\n* fix clippy\n\n* fix fmt\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2025-10-13T19:41:14Z",
          "tree_id": "a2dd8768cdb51093a256f03d5efa08eb67d1951a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d9c3c04b61cda15e4b416f00ee8be3594b2b5708"
        },
        "date": 1760386215632,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327003901,
            "range": "± 949477",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 372022574,
            "range": "± 1166774",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277600530,
            "range": "± 4350259",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687967982,
            "range": "± 1073563",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 785086018,
            "range": "± 4227415",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1442572902,
            "range": "± 1855986",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1646686277,
            "range": "± 9313311",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1221192814,
            "range": "± 892421",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3017093305,
            "range": "± 1301463",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3446167998,
            "range": "± 21249847",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6314248004,
            "range": "± 1965320",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7325057350,
            "range": "± 17558812",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5356191949,
            "range": "± 2566175",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8215384,
            "range": "± 6980",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8286234,
            "range": "± 10374",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10339980,
            "range": "± 40292",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10461441,
            "range": "± 43933",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18975059,
            "range": "± 85143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19123153,
            "range": "± 97328",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29601906,
            "range": "± 753768",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29007544,
            "range": "± 625723",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37952887,
            "range": "± 232341",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38089513,
            "range": "± 107115",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70829695,
            "range": "± 1072510",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70545049,
            "range": "± 1064771",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 76952903,
            "range": "± 150406",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77525153,
            "range": "± 202212",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145733558,
            "range": "± 1678057",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 146793590,
            "range": "± 2083820",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153579462,
            "range": "± 433381",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153146846,
            "range": "± 410654",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 299082342,
            "range": "± 3417229",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 299439096,
            "range": "± 4130444",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16791018,
            "range": "± 468063",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35837178,
            "range": "± 623944",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72010386,
            "range": "± 1448005",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143570805,
            "range": "± 3183094",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 364989740,
            "range": "± 3545268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 364073093,
            "range": "± 2471830",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 770399434,
            "range": "± 5166278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1622037508,
            "range": "± 2128959",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3385749529,
            "range": "± 10682648",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7134777766,
            "range": "± 12866778",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393985161,
            "range": "± 2542414",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813949176,
            "range": "± 3791740",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1712152356,
            "range": "± 4667565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3540453825,
            "range": "± 12449434",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7434124041,
            "range": "± 12618574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 51,
            "range": "± 0",
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
            "value": 77,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 153972,
            "range": "± 329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1577286,
            "range": "± 118712",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 858628,
            "range": "± 800",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1661229,
            "range": "± 25863",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 253,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 44,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 60",
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
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7456,
            "range": "± 749",
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
            "value": 9845,
            "range": "± 449",
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
            "email": "34384633+tdelabro@users.noreply.github.com",
            "name": "Timothée Delabrouille",
            "username": "tdelabro"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3b96e92c046dc7088a53dfb6f4772489a05cc11b",
          "message": "feat: support formatter alternate display (#1049)\n\n* feat: support formatter alternate display\n\n* feat: impl LowerHex and UpperHex on UnsignedElement",
          "timestamp": "2025-10-17T15:49:47-03:00",
          "tree_id": "574bc70ac6a625145ec24c1cb3e7998ea0ef98c3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3b96e92c046dc7088a53dfb6f4772489a05cc11b"
        },
        "date": 1760728468268,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326213869,
            "range": "± 1078415",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370570059,
            "range": "± 1115020",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279011883,
            "range": "± 629632",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684628464,
            "range": "± 1113127",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 783866900,
            "range": "± 2893513",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432018970,
            "range": "± 1802752",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1647858640,
            "range": "± 15739586",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1227111175,
            "range": "± 13531387",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2992092458,
            "range": "± 3553204",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3448239716,
            "range": "± 19567647",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6265565996,
            "range": "± 2482725",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7334672606,
            "range": "± 23616032",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5378243807,
            "range": "± 5991506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7805391,
            "range": "± 29109",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7859042,
            "range": "± 7321",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10431971,
            "range": "± 416148",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12191920,
            "range": "± 947698",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17409705,
            "range": "± 44579",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17399479,
            "range": "± 77643",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 31808193,
            "range": "± 2282209",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30773175,
            "range": "± 1901507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 34959431,
            "range": "± 352689",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35180008,
            "range": "± 233535",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71316851,
            "range": "± 1033492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71278099,
            "range": "± 1137563",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 70258482,
            "range": "± 467684",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 70190864,
            "range": "± 265152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145678862,
            "range": "± 2495058",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143040457,
            "range": "± 2294540",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 140790093,
            "range": "± 343259",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 140634133,
            "range": "± 332762",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289069488,
            "range": "± 4781889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 292111570,
            "range": "± 5331905",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17607601,
            "range": "± 560764",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 39204493,
            "range": "± 1652486",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74654805,
            "range": "± 3320383",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 158644283,
            "range": "± 5602546",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 402584068,
            "range": "± 12516929",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 365543707,
            "range": "± 2434703",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 774183625,
            "range": "± 3035734",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612825566,
            "range": "± 7261155",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3360633517,
            "range": "± 10701187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7059319828,
            "range": "± 24622592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390372317,
            "range": "± 1302567",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 805640471,
            "range": "± 3562802",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1692172362,
            "range": "± 3510480",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3491686087,
            "range": "± 19636439",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7348042954,
            "range": "± 12157141",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1114,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 142446,
            "range": "± 160",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 774,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 406,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1220,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 108282,
            "range": "± 376",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 153870,
            "range": "± 515",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578704,
            "range": "± 32011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 858552,
            "range": "± 4529",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1702629,
            "range": "± 19172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2663,
            "range": "± 2613",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 201902,
            "range": "± 3115",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1132,
            "range": "± 10",
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
            "value": 47,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 61,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 29,
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
            "value": 11357,
            "range": "± 460",
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
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "762daeb337fd21191c514ffcd398b51d7208b5e3",
          "message": "fix link (#1052)",
          "timestamp": "2025-10-27T15:54:10Z",
          "tree_id": "38d9d80e3ae9784c0f811ed8ed80ea33696dc3af",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/762daeb337fd21191c514ffcd398b51d7208b5e3"
        },
        "date": 1761582236999,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324602915,
            "range": "± 234628",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370683155,
            "range": "± 921586",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278020704,
            "range": "± 3374680",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682977162,
            "range": "± 1460159",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 780804267,
            "range": "± 1292401",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1431091774,
            "range": "± 4722553",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1637527791,
            "range": "± 4552033",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224147816,
            "range": "± 888916",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2990094388,
            "range": "± 3467514",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3439084034,
            "range": "± 30141218",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6263569665,
            "range": "± 6275905",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7288746098,
            "range": "± 16693646",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5373165404,
            "range": "± 3653270",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7823904,
            "range": "± 6837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7862765,
            "range": "± 3635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10358353,
            "range": "± 48972",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10413866,
            "range": "± 21368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17936744,
            "range": "± 33064",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17951752,
            "range": "± 22620",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27998817,
            "range": "± 392304",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27188450,
            "range": "± 252420",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36433239,
            "range": "± 157007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36703375,
            "range": "± 137965",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67074439,
            "range": "± 465092",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66821414,
            "range": "± 427415",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73676610,
            "range": "± 72040",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73260795,
            "range": "± 155635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 140085368,
            "range": "± 1694492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 141353068,
            "range": "± 3234017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 146818070,
            "range": "± 180678",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 146706529,
            "range": "± 92217",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 280208438,
            "range": "± 3884471",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 280130737,
            "range": "± 803742",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15950586,
            "range": "± 42551",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32887501,
            "range": "± 83654",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66436958,
            "range": "± 371453",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 135233762,
            "range": "± 1779980",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 349207971,
            "range": "± 1816323",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357384381,
            "range": "± 892621",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 760906384,
            "range": "± 4498981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1600783894,
            "range": "± 1839298",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3335351012,
            "range": "± 2637257",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7044910654,
            "range": "± 5611240",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383612613,
            "range": "± 633508",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 803257524,
            "range": "± 2166145",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1681506845,
            "range": "± 2771614",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3489377969,
            "range": "± 2883163",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7331910771,
            "range": "± 6829252",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 45,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 382,
            "range": "± 0",
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
            "value": 61,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 172,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 431,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149119,
            "range": "± 211",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1575772,
            "range": "± 13726",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 835948,
            "range": "± 1617",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1671203,
            "range": "± 62196",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 547,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2091,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 13109,
            "range": "± 841",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 333,
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
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fade1668e234e575138056df99a96469e2f3d0dd",
          "message": "Implement HasDefaultTranscript for baby bear u32 degree 4 extension field (#1055)\n\n* implement HasDefaultTranscript for degree 4  baby bear u32\n\n* fix lint\n\n* fix clippy",
          "timestamp": "2025-12-10T14:35:46Z",
          "tree_id": "e08fc5ead7534c269f8625bb19afd969a17d32f7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fade1668e234e575138056df99a96469e2f3d0dd"
        },
        "date": 1765379135578,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 329259381,
            "range": "± 170664",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375363779,
            "range": "± 981705",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279579550,
            "range": "± 257626",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 690709284,
            "range": "± 3398743",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 797750355,
            "range": "± 3153216",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1447985294,
            "range": "± 1487267",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1661230128,
            "range": "± 2479831",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1227907587,
            "range": "± 3742643",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3017968352,
            "range": "± 4299935",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3460925092,
            "range": "± 10610846",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6316506173,
            "range": "± 5789717",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7285519863,
            "range": "± 19810879",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5377086721,
            "range": "± 8191740",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7806477,
            "range": "± 9812",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7862283,
            "range": "± 6166",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11518150,
            "range": "± 525871",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11368146,
            "range": "± 725790",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18286683,
            "range": "± 63076",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18254642,
            "range": "± 56049",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34000634,
            "range": "± 2224177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 33162877,
            "range": "± 1187917",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36746065,
            "range": "± 143475",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36790808,
            "range": "± 198123",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71917048,
            "range": "± 2475023",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73226399,
            "range": "± 1276664",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74477374,
            "range": "± 381499",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73862976,
            "range": "± 319301",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147661610,
            "range": "± 1226086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 148563739,
            "range": "± 1473218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 148675450,
            "range": "± 919179",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 148514398,
            "range": "± 344326",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 302831100,
            "range": "± 2896022",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 298804076,
            "range": "± 2281683",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18969703,
            "range": "± 557282",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 38199448,
            "range": "± 502562",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 75562977,
            "range": "± 756339",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 153886602,
            "range": "± 3409392",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 369325593,
            "range": "± 4927486",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 372084137,
            "range": "± 1568904",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 786721796,
            "range": "± 6714204",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1636364491,
            "range": "± 4129420",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3392721170,
            "range": "± 7057590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7142151105,
            "range": "± 28328583",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393501449,
            "range": "± 1722544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 822561113,
            "range": "± 5070804",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1717966672,
            "range": "± 6552560",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3591037939,
            "range": "± 33375102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7469020805,
            "range": "± 98043184",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 92,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 63,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 91,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 168,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 148918,
            "range": "± 293",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576121,
            "range": "± 14964",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 835896,
            "range": "± 2109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1840219,
            "range": "± 12558",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 271,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 667,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 22,
            "range": "± 2",
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
            "value": 10372,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 89,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10300,
            "range": "± 470",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 70,
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
            "value": 9997,
            "range": "± 703",
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
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e17bcc29450cdd6d908e82ca998b20823253722a",
          "message": "Add copy to BabyBear u32 Extension Field struct (#1056)\n\n* add copy\n\n* fix clippy\n\n* fix clippy\n\n* fix clippy",
          "timestamp": "2025-12-11T20:08:28Z",
          "tree_id": "6f90e690ace7a93615dc9147c7a988b6a191ad9d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e17bcc29450cdd6d908e82ca998b20823253722a"
        },
        "date": 1765485385936,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325406357,
            "range": "± 1449160",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367525199,
            "range": "± 1396848",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278242297,
            "range": "± 367994",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685018077,
            "range": "± 363005",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 777713631,
            "range": "± 1089316",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432844587,
            "range": "± 902600",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1639458180,
            "range": "± 1564130",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223808194,
            "range": "± 721911",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2994213970,
            "range": "± 1781865",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3430968624,
            "range": "± 8243283",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6271091231,
            "range": "± 5127099",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7276624083,
            "range": "± 20112865",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5369050830,
            "range": "± 6050521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8065219,
            "range": "± 18642",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8081004,
            "range": "± 15222",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10448562,
            "range": "± 161469",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11560672,
            "range": "± 337863",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18474816,
            "range": "± 83177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18648780,
            "range": "± 70076",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28949114,
            "range": "± 851443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28173462,
            "range": "± 777969",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37400728,
            "range": "± 92193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37308523,
            "range": "± 111876",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69009334,
            "range": "± 879822",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68155552,
            "range": "± 580314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75255742,
            "range": "± 102992",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75080610,
            "range": "± 183416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 142452756,
            "range": "± 795215",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142616483,
            "range": "± 734250",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 150887530,
            "range": "± 240194",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 150814659,
            "range": "± 145358",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 286192501,
            "range": "± 977218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 287260855,
            "range": "± 1300607",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15826397,
            "range": "± 222037",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33707776,
            "range": "± 399382",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 67798973,
            "range": "± 350093",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137835337,
            "range": "± 1438211",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 355163739,
            "range": "± 1454605",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359807250,
            "range": "± 1312720",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764402117,
            "range": "± 2366022",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1605633261,
            "range": "± 4717535",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3338491693,
            "range": "± 3021876",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7055355111,
            "range": "± 6787329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389630755,
            "range": "± 1533826",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813572698,
            "range": "± 4922383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1695252408,
            "range": "± 6333615",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3498384910,
            "range": "± 3543894",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7358980115,
            "range": "± 8331080",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 111,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1779,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 184,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 123,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 315,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1907,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 151613,
            "range": "± 414",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576668,
            "range": "± 5092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 832975,
            "range": "± 1700",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1694880,
            "range": "± 11051",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 949,
            "range": "± 551",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7635,
            "range": "± 118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 114,
            "range": "± 0",
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
          "id": "5998a043d7613282977a92fa4bbb8456b4a68117",
          "message": "Refactor air for dyn (#1058)",
          "timestamp": "2025-12-22T17:26:08Z",
          "tree_id": "0c2de7ed3a51bd2cb5c0011de4c49287d149fed2",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5998a043d7613282977a92fa4bbb8456b4a68117"
        },
        "date": 1766426148646,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324724647,
            "range": "± 449404",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 372250927,
            "range": "± 923871",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277838295,
            "range": "± 272783",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682101775,
            "range": "± 473121",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 792027251,
            "range": "± 1638737",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1430810534,
            "range": "± 1782538",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1680508438,
            "range": "± 3361339",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223566036,
            "range": "± 2708180",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2989018979,
            "range": "± 705364",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3511929369,
            "range": "± 5720551",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6259761299,
            "range": "± 2466897",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7442062824,
            "range": "± 8018389",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5366228136,
            "range": "± 2890874",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8156053,
            "range": "± 6759",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8156823,
            "range": "± 9313",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10647376,
            "range": "± 18550",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10644468,
            "range": "± 27822",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18534816,
            "range": "± 73583",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18507097,
            "range": "± 53521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26483422,
            "range": "± 307845",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27170659,
            "range": "± 384524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37362158,
            "range": "± 26760",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37397085,
            "range": "± 224019",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67311486,
            "range": "± 1152236",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66350858,
            "range": "± 662558",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75482996,
            "range": "± 42785",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75456592,
            "range": "± 36889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139584842,
            "range": "± 924681",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 140670216,
            "range": "± 848827",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151568234,
            "range": "± 435057",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151459889,
            "range": "± 144975",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 281865282,
            "range": "± 662773",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 281986729,
            "range": "± 597576",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14425380,
            "range": "± 286713",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32808339,
            "range": "± 363945",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 64888193,
            "range": "± 202035",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 133349058,
            "range": "± 3284424",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 355152160,
            "range": "± 3142192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 353300218,
            "range": "± 1740276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 756602876,
            "range": "± 1030408",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1597912231,
            "range": "± 3195584",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3329693128,
            "range": "± 4514017",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7017592094,
            "range": "± 28216237",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 381658121,
            "range": "± 1477383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 810895684,
            "range": "± 3499205",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699091898,
            "range": "± 7235972",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3516212716,
            "range": "± 4940657",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7396557256,
            "range": "± 7411707",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 254,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8123,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 307,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 181,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 462,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6581,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149750,
            "range": "± 460",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568963,
            "range": "± 8528",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 845524,
            "range": "± 3884",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1817579,
            "range": "± 42040",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1314,
            "range": "± 1266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 23447,
            "range": "± 649",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 263,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 131,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11576,
            "range": "± 544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 190,
            "range": "± 5",
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
          "id": "bc6dd545e9daf5bac32e24467279b4b90367744c",
          "message": "Make AIR Dynamic, and embed them along the Prover (#1059)\n\n* make IsStarkTranscript dyn\n\n* multi_prove first draft\n\n* fix\n\n* Make PublicInputs generic again\n\n* fixes\n\n* re-enable debug code\n\n* restore commented code\n\n* clippy\n\n* fix parallel feature : swap &dyn AIR -> Box<dyn AIR>\n\n* get rid of Box\n\n* remove uneeded code\n\n* cleanup\n\n* cleanup\n\n* cleanup\n\n* Remove duplicate AsBytes trait bound\n\n* Remove uneeded trait bounds\n\n* Remove uneeded trait bounds\n\n* Remove Send + Sync bounds from PublicInputs generic PI\n\n* Remove Send + Sync bounds from PublicInputs generic PI\n\n* Apply same trait refactor to IsStarkVerifier\n\n* Remove proof_options argument from verify\n\n* Add multi_verify\n\n---------\n\nCo-authored-by: fmoletta <fedemoletta@hotmail.com>",
          "timestamp": "2026-01-05T18:18:24Z",
          "tree_id": "1b3b2c961402d83ab01ffcf92a5f577782fa9a53",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/bc6dd545e9daf5bac32e24467279b4b90367744c"
        },
        "date": 1767638851325,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323862335,
            "range": "± 1171619",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367718544,
            "range": "± 1392692",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 276231131,
            "range": "± 88572",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681703306,
            "range": "± 601894",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 781667595,
            "range": "± 2586657",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1429458047,
            "range": "± 794209",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1650271155,
            "range": "± 1717573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1217710128,
            "range": "± 460324",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2987988502,
            "range": "± 1453443",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3452850789,
            "range": "± 2351416",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6258219352,
            "range": "± 1871858",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7314189707,
            "range": "± 5963561",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5345794983,
            "range": "± 2117543",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8080674,
            "range": "± 3563",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8091097,
            "range": "± 7541",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10495095,
            "range": "± 7830",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10504075,
            "range": "± 19453",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17994703,
            "range": "± 22353",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18032860,
            "range": "± 38233",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 23764830,
            "range": "± 150482",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 24029969,
            "range": "± 134656",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37109895,
            "range": "± 59555",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36885439,
            "range": "± 42345",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 64727494,
            "range": "± 226921",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 64758643,
            "range": "± 225171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74660075,
            "range": "± 45349",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74913720,
            "range": "± 39341",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 138473517,
            "range": "± 187177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 139125395,
            "range": "± 446667",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 150885069,
            "range": "± 43112",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 150150423,
            "range": "± 155137",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 282529556,
            "range": "± 443384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 280625971,
            "range": "± 580365",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14568391,
            "range": "± 172779",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32278367,
            "range": "± 107039",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 65512226,
            "range": "± 266290",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 133453488,
            "range": "± 510587",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 328709380,
            "range": "± 678512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 351561101,
            "range": "± 726552",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 753929304,
            "range": "± 571886",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1590202551,
            "range": "± 1047511",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3324195650,
            "range": "± 1504223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7024808199,
            "range": "± 6859644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 380769692,
            "range": "± 548259",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 795857485,
            "range": "± 442585",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1677480610,
            "range": "± 1038150",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3481691284,
            "range": "± 1980752",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7331646469,
            "range": "± 5693316",
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
            "value": 96,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 152,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149935,
            "range": "± 426",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568168,
            "range": "± 9966",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837928,
            "range": "± 1303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1746608,
            "range": "± 6662",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 384,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 758,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 21,
            "range": "± 2",
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
            "value": 10419,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 89,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10044,
            "range": "± 1525",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 66,
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
      }
    ]
  }
}