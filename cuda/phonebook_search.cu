%%writefile cuda_phonebook.cu
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define MAX_LINE 100
#define MAX_NAME 50

__global__ void searchKernel(char phonebook[][MAX_LINE],
                             char* search,
                             int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        bool match = true;
        int i = 0;
        while (search[i] != '\0') {
            if (phonebook[idx][i] != search[i]) {
                match = false;
                break;
            }
            i++;
        }
        if (match) {
            printf("Match: %s\n", phonebook[idx]);
        }
    }
}

int main() {
    vector<string> lines;
    string line, search;

    ifstream fin("phonebook.txt");
    while (getline(fin, line))
        lines.push_back(line);
    fin.close();

    cout << "Enter name to search: ";
    cin >> search;

    int n = lines.size();

    char h_phonebook[n][MAX_LINE];
    char h_search[MAX_NAME];

    for (int i = 0; i < n; i++)
        strcpy(h_phonebook[i], lines[i].c_str());
    strcpy(h_search, search.c_str());

    char (*d_phonebook)[MAX_LINE];
    char* d_search;

    cudaMalloc(&d_phonebook, n * MAX_LINE);
    cudaMalloc(&d_search, MAX_NAME);

    cudaMemcpy(d_phonebook, h_phonebook, n * MAX_LINE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_search, h_search, MAX_NAME, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    searchKernel<<<blocks, threads>>>(d_phonebook, d_search, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cout << "\nCUDA Time: " << ms << " ms\n";

    cudaFree(d_phonebook);
    cudaFree(d_search);
}
