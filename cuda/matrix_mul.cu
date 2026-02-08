%%writefile fixed_matmul.cu
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matMulKernel(
    float* A, float* B, float* C,
    int M, int N, int P, int K)
{
    int k   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < K && row < M && col < P) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[k*M*N + row*N + j] *
                   B[k*N*P + j*P + col];
        }
        C[k*M*P + row*P + col] = sum;
    }
}

int main() {
    int M = 4, N = 4, P = 4, K = 2;

    int sizeA = M*N*K;
    int sizeB = N*P*K;
    int sizeC = M*P*K;

    float *h_A = new float[sizeA];
    float *h_B = new float[sizeB];
    float *h_C = new float[sizeC];

    // Initialize A and B, initialize C = 0
    for (int i = 0; i < sizeA; i++) h_A[i] = 1;
    for (int i = 0; i < sizeB; i++) h_B[i] = 1;
    for (int i = 0; i < sizeC; i++) h_C[i] = 0;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA*sizeof(float));
    cudaMalloc(&d_B, sizeB*sizeof(float));
    cudaMalloc(&d_C, sizeC*sizeof(float));

    cudaMemcpy(d_A, h_A, sizeA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(16,16);
    dim3 blocks((P+15)/16, (M+15)/16, K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, N, P, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, sizeC*sizeof(float), cudaMemcpyDeviceToHost);

    // Print A[0]
    cout << "\nA[0]:\n";
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++)
            cout << h_A[i*N + j] << " ";
        cout << endl;
    }

    // Print B[0]
    cout << "\nB[0]:\n";
    for(int i=0;i<N;i++){
        for(int j=0;j<P;j++)
            cout << h_B[i*P + j] << " ";
        cout << endl;
    }

    // Print C[0]
    cout << "\nC[0] = A[0] × B[0]:\n";
    for(int i=0;i<M;i++){
        for(int j=0;j<P;j++)
            cout << h_C[i*P + j] << " ";
        cout << endl;
    }

    cout << "\nGPU Time: " << ms << " ms\n";

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
