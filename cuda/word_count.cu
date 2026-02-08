%%writefile wordcount.cu
#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define MAX_WORDS 50000
#define WORD_LEN 32

__global__ void countWords(char* words, int* freq, int totalWords) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < totalWords) {
        atomicAdd(&freq[id], 1);
    }
}

int main() {
    vector<string> words;
    ifstream fin("input.txt");
    string w;
    while(fin >> w) words.push_back(w);
    fin.close();

    int n = words.size();
    if(n > MAX_WORDS) n = MAX_WORDS;

    char* h_words = new char[n * WORD_LEN];
    int* h_freq = new int[n];

    memset(h_freq, 0, n*sizeof(int));

    for(int i=0;i<n;i++)
        strncpy(&h_words[i*WORD_LEN], words[i].c_str(), WORD_LEN);

    char* d_words;
    int* d_freq;
    cudaMalloc(&d_words, n * WORD_LEN);
    cudaMalloc(&d_freq, n * sizeof(int));

    cudaMemcpy(d_words, h_words, n*WORD_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_freq, h_freq, n*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    countWords<<<blocks,threads>>>(d_words, d_freq, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_freq, d_freq, n*sizeof(int), cudaMemcpyDeviceToHost);

    map<string,int> mp;
    for(int i=0;i<n;i++)
        mp[words[i]]++;

    vector<pair<int,string>> v;
    for(auto &p:mp)
        v.push_back({p.second,p.first});

    sort(v.rbegin(), v.rend());

    cout << "\nTop 10 words:\n";
    for(int i=0;i<min(10,(int)v.size());i++)
        cout << v[i].second << " -> " << v[i].first << endl;

    cout << "\nCUDA Time: " << ms << " ms\n";

    cudaFree(d_words);
    cudaFree(d_freq);
    delete[] h_words;
    delete[] h_freq;
}


/*
nvcc wordcount.cu -o wc_cuda
./wc_cuda

*/