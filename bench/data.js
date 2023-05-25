window.BENCHMARK_DATA = {
  "lastUpdate": 1685014092154,
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
          "id": "943963cdfb977c2f54bf6dca60e5475985c30606",
          "message": "Add cairo memory and range check constraints (#337)\n\n* add trace commitments to transcript\n\n* move sampling of z to round 3\n\n* add batch_commit function\n\n* add commitments of the composition polynomial to the transcript\n\n* refactor sampling of boundary and transition coefficients\n\n* add ood evaluations to transcript\n\n* minor refactor\n\n* move sample batch to lib\n\n* extract deep composition randomness to round 4\n\n* refactor fri commitment phase\n\n* refactor next_fri_layer\n\n* remove last iteration of fri commit phase\n\n* refactor fri_commit_phase\n\n* move sampling of q_0 to query phase. Rename\n\n* refactor fri decommitment\n\n* add fri last value to proof and remove it from decommitments\n\n* remove layers commitments from decommitment\n\n* remove unused FriQuery struct\n\n* leave only symmetric points in the proof\n\n* remove unnecesary last fri layer\n\n* reuse composition poly evaluations from round_1 in the consistency check\n\n* minor refactor\n\n* fix trace ood commitments and small refactor\n\n* move fri_query_phase to fri mod\n\n* minor refactor in build deeep composition poly function\n\n* refactor deep composition poly related code\n\n* minor modifications to comments\n\n* clippy\n\n* add comments\n\n* move iota sampling to step 0 and rename challenges\n\n* minor refactor and add missing opening checks\n\n* minor refactor\n\n* move transition divisor computation outside of the main loop of the computation of the composition polynomial\n\n* add protocol to docs\n\n* clippy and fmt\n\n* remove old test\n\n* fix typo\n\n* fmt\n\n* remove unnecessary public attribute\n\n* move build_execution trace to round 1\n\n* add rap support to air\n\n* remove commit_original_trace function\n\n* Add auxiliary trace polynomials and commitments in the first round\n\n* Test simple fibonacci\n\n* fix format in starks protocol docs\n\n* fmt, clippy\n\n* remove comments, add sampling of cairo rap\n\n* add commented test\n\n* remove old test\n\n* First step of memory constraints\n\n* Add public input to AIR. Build auxiliary columns for Cairo.\n\n* Fix bug in build_aux\n\n* Cargo fmt\n\n* fix aux table bug\n\n* fix number_auxiliary_rap_columns\n\n* Remove program_size parameter for CairoAIR\n\n* Refactor merge PublicInputs\n\n* Add transition degree in fibonacci rap\n\n* Tests for build auxiliary trace\n\n* Add auxiliary transition constraints for memory\n\n* Add final permutation argumentation boundary auxiliary constraint\n\n* WIP: boundary memory constraints\n\n* fix memory indexing\n\n* make public input mutable argument of build_main_trace\n\n* add range check column build function\n\n* Add missing values in offsets columns. Concatenate sorted offsets to original trace\n\n* add test to concatenate. Fix offset variables\n\n* add permutation argument column for range checks\n\n* add range_check permutation argument transition constraints\n\n* add range checks continuity transition constraints\n\n* Add debug assertion for verifying divisibility of transition polynomials by their zerofiers\n\n* Add boundary constraint for range checks\n\n* Add boundary constraints range check min and max\n\n* Add dummy AIR and test to debug\n\n* fix evaluate_polynomial_on_lde_domain\n\n* Remove default implementation for number of auxiliary RAP columns\n\n* fix composition poly bound degree\n\n* remove unused public input attribute\n\n* clippy\n\n* fix dummy test\n\n* Remove check polynomial divisibility\n\n* Add test of malicious program\n\n* Add simple_program.json for testing purposes\n\n* Remove some useless commented code\n\n* Fix: malicious test uses also malicious memory\n\n* Cargo\n\n* Add comments\n\n* comment transition exemptions vector\n\n* update stark protocol to include range checks\n\n* use markdown auto numbered lists\n\n* Ignore test_prove_cairo_simple_program for Metal\n\n* move boundary term degree adjustment power out of the closure\n\n* Add test for range checks. Rename malicious test.\n\n* Helper function to load cairo programs and memory\n\n* Return Error when trace length is not enough for the Cairo trace to fit in\n\n* Return an error when trace length is not large enough. Add larger\nfibonacci tests.\n\n* clippy\n\n* ignore large test cases\n\n---------\n\nCo-authored-by: ajgarassino <ajgarassino@gmail.com>\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Sergio Chouhy <schouhy@eryxsoluciones.com.ar>",
          "timestamp": "2023-05-25T09:40:44Z",
          "tree_id": "cd9853079a2588775eb36934c8d10b13ec7c0046",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/943963cdfb977c2f54bf6dca60e5475985c30606"
        },
        "date": 1685007989914,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131503467,
            "range": "± 4314482",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248141614,
            "range": "± 19962184",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 481096083,
            "range": "± 2121285",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 975324958,
            "range": "± 8794766",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34464143,
            "range": "± 250819",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68014087,
            "range": "± 438897",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132877031,
            "range": "± 631354",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 278139822,
            "range": "± 3846727",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31187599,
            "range": "± 345414",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58582782,
            "range": "± 988400",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122859854,
            "range": "± 3210673",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 263854364,
            "range": "± 20041801",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164858428,
            "range": "± 751532",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 331561989,
            "range": "± 1197276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 653177895,
            "range": "± 4825007",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1302930104,
            "range": "± 10709339",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 456422531,
            "range": "± 1391724",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 915871750,
            "range": "± 5743565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1800456646,
            "range": "± 3952873",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3535342542,
            "range": "± 16118794",
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
          "id": "943963cdfb977c2f54bf6dca60e5475985c30606",
          "message": "Add cairo memory and range check constraints (#337)\n\n* add trace commitments to transcript\n\n* move sampling of z to round 3\n\n* add batch_commit function\n\n* add commitments of the composition polynomial to the transcript\n\n* refactor sampling of boundary and transition coefficients\n\n* add ood evaluations to transcript\n\n* minor refactor\n\n* move sample batch to lib\n\n* extract deep composition randomness to round 4\n\n* refactor fri commitment phase\n\n* refactor next_fri_layer\n\n* remove last iteration of fri commit phase\n\n* refactor fri_commit_phase\n\n* move sampling of q_0 to query phase. Rename\n\n* refactor fri decommitment\n\n* add fri last value to proof and remove it from decommitments\n\n* remove layers commitments from decommitment\n\n* remove unused FriQuery struct\n\n* leave only symmetric points in the proof\n\n* remove unnecesary last fri layer\n\n* reuse composition poly evaluations from round_1 in the consistency check\n\n* minor refactor\n\n* fix trace ood commitments and small refactor\n\n* move fri_query_phase to fri mod\n\n* minor refactor in build deeep composition poly function\n\n* refactor deep composition poly related code\n\n* minor modifications to comments\n\n* clippy\n\n* add comments\n\n* move iota sampling to step 0 and rename challenges\n\n* minor refactor and add missing opening checks\n\n* minor refactor\n\n* move transition divisor computation outside of the main loop of the computation of the composition polynomial\n\n* add protocol to docs\n\n* clippy and fmt\n\n* remove old test\n\n* fix typo\n\n* fmt\n\n* remove unnecessary public attribute\n\n* move build_execution trace to round 1\n\n* add rap support to air\n\n* remove commit_original_trace function\n\n* Add auxiliary trace polynomials and commitments in the first round\n\n* Test simple fibonacci\n\n* fix format in starks protocol docs\n\n* fmt, clippy\n\n* remove comments, add sampling of cairo rap\n\n* add commented test\n\n* remove old test\n\n* First step of memory constraints\n\n* Add public input to AIR. Build auxiliary columns for Cairo.\n\n* Fix bug in build_aux\n\n* Cargo fmt\n\n* fix aux table bug\n\n* fix number_auxiliary_rap_columns\n\n* Remove program_size parameter for CairoAIR\n\n* Refactor merge PublicInputs\n\n* Add transition degree in fibonacci rap\n\n* Tests for build auxiliary trace\n\n* Add auxiliary transition constraints for memory\n\n* Add final permutation argumentation boundary auxiliary constraint\n\n* WIP: boundary memory constraints\n\n* fix memory indexing\n\n* make public input mutable argument of build_main_trace\n\n* add range check column build function\n\n* Add missing values in offsets columns. Concatenate sorted offsets to original trace\n\n* add test to concatenate. Fix offset variables\n\n* add permutation argument column for range checks\n\n* add range_check permutation argument transition constraints\n\n* add range checks continuity transition constraints\n\n* Add debug assertion for verifying divisibility of transition polynomials by their zerofiers\n\n* Add boundary constraint for range checks\n\n* Add boundary constraints range check min and max\n\n* Add dummy AIR and test to debug\n\n* fix evaluate_polynomial_on_lde_domain\n\n* Remove default implementation for number of auxiliary RAP columns\n\n* fix composition poly bound degree\n\n* remove unused public input attribute\n\n* clippy\n\n* fix dummy test\n\n* Remove check polynomial divisibility\n\n* Add test of malicious program\n\n* Add simple_program.json for testing purposes\n\n* Remove some useless commented code\n\n* Fix: malicious test uses also malicious memory\n\n* Cargo\n\n* Add comments\n\n* comment transition exemptions vector\n\n* update stark protocol to include range checks\n\n* use markdown auto numbered lists\n\n* Ignore test_prove_cairo_simple_program for Metal\n\n* move boundary term degree adjustment power out of the closure\n\n* Add test for range checks. Rename malicious test.\n\n* Helper function to load cairo programs and memory\n\n* Return Error when trace length is not enough for the Cairo trace to fit in\n\n* Return an error when trace length is not large enough. Add larger\nfibonacci tests.\n\n* clippy\n\n* ignore large test cases\n\n---------\n\nCo-authored-by: ajgarassino <ajgarassino@gmail.com>\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Sergio Chouhy <schouhy@eryxsoluciones.com.ar>",
          "timestamp": "2023-05-25T09:40:44Z",
          "tree_id": "cd9853079a2588775eb36934c8d10b13ec7c0046",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/943963cdfb977c2f54bf6dca60e5475985c30606"
        },
        "date": 1685010429220,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1627068009,
            "range": "± 10868121",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3892612129,
            "range": "± 77041785",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3395123876,
            "range": "± 5676327",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8552195085,
            "range": "± 61548463",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7067305799,
            "range": "± 6287837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 18341367412,
            "range": "± 89841504",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 14729824848,
            "range": "± 15221688",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 39242362801,
            "range": "± 529624859",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1154489496,
            "range": "± 1383378",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2419840374,
            "range": "± 3255042",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5072167223,
            "range": "± 4063114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10599014256,
            "range": "± 8163458",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 82600361,
            "range": "± 3500586",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 164438756,
            "range": "± 2049624",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 336847889,
            "range": "± 6851403",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 668649937,
            "range": "± 10956530",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2801689561,
            "range": "± 6932607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5859908944,
            "range": "± 6062540",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 12263922694,
            "range": "± 13167590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 25435395707,
            "range": "± 17499229",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3012304017,
            "range": "± 3008576",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6302840024,
            "range": "± 2862855",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 13107479122,
            "range": "± 7515206",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 27251055831,
            "range": "± 51458368",
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
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 84,
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
            "value": 156,
            "range": "± 1",
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
            "value": 119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 1383979,
            "range": "± 396",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 26872750,
            "range": "± 6048",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 78151890,
            "range": "± 155968",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 255266107,
            "range": "± 346243",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 910254037,
            "range": "± 2644000",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 949639,
            "range": "± 1163",
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
          "id": "bf93accc0ea0401499c514634dac064dea4f12eb",
          "message": "Remove plonk (#379)\n\n* Remove plonk\r\n\r\n* Update gitignore\r\n\r\n* Remove DS Store",
          "timestamp": "2023-05-25T08:22:13-03:00",
          "tree_id": "87b52a97812cc51b857b130adf551b025d5d176d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/bf93accc0ea0401499c514634dac064dea4f12eb"
        },
        "date": 1685014089841,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128996345,
            "range": "± 2161765",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256197906,
            "range": "± 1387115",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 487212520,
            "range": "± 5320694",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 981927812,
            "range": "± 10392032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34196275,
            "range": "± 174178",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67784818,
            "range": "± 507683",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 144628746,
            "range": "± 5204995",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 297874958,
            "range": "± 6323514",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 36963347,
            "range": "± 1943273",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 60371793,
            "range": "± 2779711",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 118825035,
            "range": "± 4304820",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 237915229,
            "range": "± 3152582",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 233118020,
            "range": "± 24756147",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 447166635,
            "range": "± 40835955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 1136933771,
            "range": "± 20024922",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 2514252854,
            "range": "± 108715456",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 467387260,
            "range": "± 2950213",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 926398229,
            "range": "± 4028138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1925434333,
            "range": "± 118453923",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 4272485833,
            "range": "± 130077728",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}