#include <stdio.h>
#include "ggml.h"
#include "llamac.h"

// todo: just a placeholder for now... will be moved to the cuda module
int main() {

    // crude lib test

    const int N = 4;
    float A[N*N], B[N*N], C[N*N];

    for (int i = 0; i < N*N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    matmul_cuda(A, B, C, N);

    printf("Result:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.1f ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}

