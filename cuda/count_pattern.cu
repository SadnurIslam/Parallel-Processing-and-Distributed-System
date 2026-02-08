%%writefile cuda_pattern.cu
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

__global__ void countKernel(char* text, char* pat, int* count, int n, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx + m <= n) {
        bool match = true;
        for (int j = 0; j < m; j++) {
            if (text[idx + j] != pat[j]) {
                match = false;
                break;
            }
        }
        if (match) atomicAdd(count, 1);
    }
}

int main() {
    ifstream fin("input.txt");
    string paragraph, line;
    while (getline(fin, line))
        paragraph += line + " ";
    fin.close();

    string pattern;
    cout << "Enter pattern (like %x%): ";
    cin >> pattern;
    pattern = pattern.substr(1, pattern.size() - 2);

    int n = paragraph.size();
    int m = pattern.size();

    char *d_text, *d_pat;
    int *d_count;

    cudaMalloc(&d_text, n);
    cudaMalloc(&d_pat, m);
    cudaMalloc(&d_count, sizeof(int));

    int zero = 0;
    cudaMemcpy(d_text, paragraph.c_str(), n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat, pattern.c_str(), m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &zero, sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    countKernel<<<blocks, threads>>>(d_text, d_pat, d_count, n, m);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    int result;
    cudaMemcpy(&result, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "\nTotal Occurrences: " << result << endl;
    cout << "CUDA Time: " << ms << " ms\n";

    cudaFree(d_text);
    cudaFree(d_pat);
    cudaFree(d_count);
}
