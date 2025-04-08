window.BENCHMARK_DATA = {
  "lastUpdate": 1744145067778,
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
      }
    ]
  }
}