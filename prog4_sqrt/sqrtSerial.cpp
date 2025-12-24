#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


void sqrtSerial(int N,
                float initialGuess,
                float values[],
                float output[])
{

    static const float kThreshold = 0.00001f;

    for (int i=0; i<N; i++) {

        float x = values[i];
        float guess = initialGuess;

        float error = fabs(guess * guess * x - 1.f);

        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = fabs(guess * guess * x - 1.f);
        }

        output[i] = x * guess;
    }
}

void my_sqrt(int N,
            float initialGuess,
            float values[],
            float output[])
{

    static const float kThreshold = 0.00001f;
    constexpr int VECTOR_LENGTH = 8;
    const __m256 kThreshVec = _mm256_set1_ps(kThreshold);
    const __m256 kOne = _mm256_set1_ps(1.0f);
    const __m256 kHalf = _mm256_set1_ps(0.5f);
    const __m256 kThree = _mm256_set1_ps(3.0f);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    static const __m256i kTailMasks[9] = {
        _mm256_setzero_si256(),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
        _mm256_set1_epi32(-1)
    };

    for (int i=0; i<N; i+=VECTOR_LENGTH) {
        const int remain = N - i;
        const __m256i mask = (remain >= VECTOR_LENGTH) ? kTailMasks[8] : kTailMasks[remain];
        __m256 boundary_mask_ps = _mm256_castsi256_ps(mask);

        __m256 x = _mm256_maskload_ps(values + i, mask);
        __m256 guess = _mm256_set1_ps(initialGuess);

        while (true) {
            __m256 guess_sq = _mm256_mul_ps(guess, guess);
            __m256 diff = _mm256_fnmadd_ps(x, guess_sq, kOne); // 1 - x * guess_sq
            __m256 error = _mm256_andnot_ps(sign_mask, diff);

            __m256 gt_mask = _mm256_cmp_ps(error, kThreshVec, _CMP_GT_OQ);
            __m256 active_mask = _mm256_and_ps(gt_mask, boundary_mask_ps);
            if (_mm256_movemask_ps(active_mask) == 0) {
                break;
            }

            __m256 guess_cube = _mm256_mul_ps(guess_sq, guess);
            __m256 update = _mm256_mul_ps(_mm256_fnmadd_ps(x, guess_cube, _mm256_mul_ps(kThree, guess)), kHalf);
            guess = _mm256_blendv_ps(guess, update, active_mask);
        }

        __m256 result = _mm256_mul_ps(x, guess);
        _mm256_maskstore_ps(output + i, mask, result);
    }
}
