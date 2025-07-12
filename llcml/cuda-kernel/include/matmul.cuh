//
// Created by ervin on 7/12/25.
//

#ifndef MATMUL_H
#define MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

    void matmul_cuda(const float* A, const float* B, float* C, int N);

#ifdef __cplusplus
}
#endif


#endif // MATMUL_H


