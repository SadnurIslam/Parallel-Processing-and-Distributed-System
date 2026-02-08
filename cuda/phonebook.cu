%%writefile phonebook_search.cu
#include <bits/stdc++.h>
using namespace std;
#include <cuda.h>

struct Contact{
    char name[65];
    char phone_number[65];
};

// Removed the old getInput function as it's no longer needed

__device__ bool check(char* str1, char* str2){
    for(int i = 0; str1[i] != '\0'; i++){
        int flag = 1;
        for(int j = 0; str2[j] != '\0' ; j++){
            if(str1[i + j] != str2[j]){
                flag = 0;
                break;
            }
        }
        if(flag == 1) return true;
    }
    return false;
}


__global__ void myKernel(Contact* phoneBook, char* pat, int offset){
    int threadNumber = threadIdx.x + blockDim.x * blockIdx.x + offset; // Corrected thread numbering for multiple blocks
    // Added a bounds check to prevent out-of-bounds access if n is not a multiple of blockSize
    if (threadNumber < 10000) { // Limit to max contacts read
        if(check(phoneBook[threadNumber].name, pat)){
            printf("%s %s\n", phoneBook[threadNumber].name, phoneBook[threadNumber].phone_number);
        }
    }
}



int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <search_name> <thread_limit>\n";
        return 1;
    }
    int threadLimit = atoi(argv[2]);

    ifstream myfile("one.txt");
    if (!myfile.is_open()) {
        cerr << "Error: Could not open one.txt\n";
        return 1;
    }
    vector<Contact> phoneBook;

    string line;
    int count = 0;
    while(getline(myfile, line)){
        if(count >= 10000) break; // Apply the same limit as before
        if(line.empty()) continue;

        size_t space_pos = line.find(' ');
        if(space_pos == string::npos) {
            // Handle lines without a space separator, e.g., print a warning or skip
            cerr << "Warning: Skipping malformed line: " << line << "\n";
            continue;
        }

        string name_str = line.substr(0, space_pos);
        string phone_str = line.substr(space_pos + 1);

        Contact c;
        // Ensure strings are null-terminated and do not overflow buffers
        strncpy(c.name, name_str.c_str(), sizeof(c.name) - 1);
        c.name[sizeof(c.name) - 1] = '\0';
        strncpy(c.phone_number, phone_str.c_str(), sizeof(c.phone_number) - 1);
        c.phone_number[sizeof(c.phone_number) - 1] = '\0';

        phoneBook.push_back(c);
        count++;
    }
    myfile.close();

    string search_name = argv[1];
    char pat[65];
    strncpy(pat, search_name.c_str(), sizeof(pat) - 1);
    pat[sizeof(pat) - 1] = '\0';

    char* d_pat;
    cudaMalloc(&d_pat, 65); //memory allocation
    cudaMemcpy(d_pat, pat, 65, cudaMemcpyHostToDevice); //copying to device

    int n = phoneBook.size();
    Contact* d_phoneBook;
    cudaMalloc(&d_phoneBook, n*sizeof(Contact));
    cudaMemcpy(d_phoneBook, phoneBook.data(), n * sizeof(Contact), cudaMemcpyHostToDevice);

    int bakiAche = n;
    int offset = 0;
    // Determine a reasonable block size for better GPU utilization, e.g., 256 or 512
    // Using threadLimit as blockSize now, as it's the max threads per block.
    int blockSize = threadLimit;

    while(bakiAche > 0){
        int numThreads = min(blockSize, bakiAche);
        int numBlocks = (numThreads + blockSize - 1) / blockSize; // Should always be 1 block if numThreads <= blockSize
        myKernel<<<numBlocks, numThreads>>>(d_phoneBook, d_pat, offset);
        cudaDeviceSynchronize();

        bakiAche -= numThreads;
        offset += numThreads;
    }

    cudaFree(d_pat);
    cudaFree(d_phoneBook);

    return 0;
}


/*
!nvcc -arch=sm_75 phonebook_search.cu -o phonebook_search
!time ./phonebook_search Lily 5 > output.txt
*/