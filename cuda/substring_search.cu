%%writefile substring_search.cu
#include <bits/stdc++.h>
using namespace std;
#include <cuda.h>

#define MAXN 10000
#define MAXLEN 65

struct Contact{
    char name[MAXLEN];
    char phone_number[MAXLEN];
};

/*
    Compute maximum length substring of pat
    that appears in str
*/
__device__ int maxMatchLen(char* str, char* pat){
    int best = 0;

    for(int i = 0; pat[i] != '\0'; i++){
        for(int len = 1; pat[i + len - 1] != '\0'; len++){
            int ok = 1;
            for(int j = 0; j < len; j++){
                if(str[i + j] == '\0' || str[i + j] != pat[i + j]){
                    ok = 0;
                    break;
                }
            }
            if(ok) best = max(best, len);
        }
    }
    return best;
}

/*
    Each thread processes one contact
*/
__global__ void kernel(Contact* phoneBook, char* pat, int* bestLen, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n){
        bestLen[tid] = maxMatchLen(phoneBook[tid].name, pat);
    }
}

int main(int argc, char* argv[])
{
    if(argc < 3){
        cerr << "Usage: " << argv[0] << " <pattern> <threads_per_block>\n";
        return 1;
    }

    string pattern = argv[1];
    int threadsPerBlock = atoi(argv[2]);

    ifstream fin("one.txt");
    if(!fin.is_open()){
        cerr << "Error opening one.txt\n";
        return 1;
    }

    vector<Contact> phoneBook;
    string line;

    while(getline(fin, line) && phoneBook.size() < MAXN){
        if(line.empty()) continue;

        size_t pos = line.find(' ');
        if(pos == string::npos) continue;

        Contact c{};
        strncpy(c.name, line.substr(0,pos).c_str(), MAXLEN-1);
        strncpy(c.phone_number, line.substr(pos+1).c_str(), MAXLEN-1);

        phoneBook.push_back(c);
    }
    fin.close();

    int n = phoneBook.size();

    // ---------------- CUDA MEMORY ----------------
    Contact* d_phoneBook;
    char* d_pat;
    int* d_bestLen;

    cudaMalloc(&d_phoneBook, n * sizeof(Contact));
    cudaMalloc(&d_pat, MAXLEN);
    cudaMalloc(&d_bestLen, n * sizeof(int));

    cudaMemcpy(d_phoneBook, phoneBook.data(),
               n * sizeof(Contact), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat, pattern.c_str(),
               MAXLEN, cudaMemcpyHostToDevice);

    // ---------------- KERNEL LAUNCH ----------------
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<blocks, threadsPerBlock>>>(d_phoneBook, d_pat, d_bestLen, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // ---------------- RESULT COLLECTION ----------------
    vector<int> bestLen(n);
    cudaMemcpy(bestLen.data(), d_bestLen,
               n * sizeof(int), cudaMemcpyDeviceToHost);

    int globalBest = 0;
    int idx = -1;
    for(int i = 0; i < n; i++){
        if(bestLen[i] > globalBest){
            globalBest = bestLen[i];
            idx = i;
        }
    }

    // ---------------- OUTPUT ----------------
    cout << "\nBest Matching Line:\n";
    if(idx != -1){
        cout << phoneBook[idx].name << " "
             << phoneBook[idx].phone_number << "\n";
        cout << "Match Length: " << globalBest << "\n";
    }else{
        cout << "No match found\n";
    }

    cout << "Total GPU time: " << ms/1000.0 << " seconds\n";

    // ---------------- CLEANUP ----------------
    cudaFree(d_phoneBook);
    cudaFree(d_pat);
    cudaFree(d_bestLen);

    return 0;
}



/*

from google.colab import files
files.upload()

!nvcc -arch=sm_75 substring_search.cu -o substring_search
!time ./substring_search Lily 5 > output.txt
*/