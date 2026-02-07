#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

// ----------------- MPI UTILS -----------------
void broadcast_string(string &text, int root = 0) {
    int length;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == root) length = text.size();
    MPI_Bcast(&length, 1, MPI_INT, root, MPI_COMM_WORLD);
    if(rank != root) text.resize(length);
    MPI_Bcast(&text[0], length, MPI_CHAR, root, MPI_COMM_WORLD);
}

string vector_to_string(const vector<string> &vec, int start = 0, int end = -1, const string &sep = "\n") {
    if(end == -1 || end > vec.size()) end = vec.size();
    string text;
    for(int i=start;i<end;i++) text += vec[i] + sep;
    return text;
}

vector<string> string_to_vector(const string &text, const string &sep = "\n") {
    vector<string> vec;
    stringstream ss(text);
    string temp;
    if(sep=="\n") {
        while(getline(ss,temp)) vec.push_back(temp);
    } else {
        while(getline(ss,temp,sep[0])) vec.push_back(temp);
    }
    return vec;
}

pair<int,int> get_work_range(int total_items, int rank, int size) {
    int per_proc = total_items / size;
    int start = rank * per_proc;
    int end = (rank == size-1) ? total_items : start + per_proc;
    return {start,end};
}

int longestMatch(const string &line, const string &pattern) {
    int maxLen = 0;
    int n = pattern.size();
    for(int i=0;i<n;i++){
        for(int len=1;i+len<=n;len++){
            if(line.find(pattern.substr(i,len)) != string::npos)
                maxLen = max(maxLen,len);
        }
    }
    return maxLen;
}

// ----------------- MAIN -----------------
int main(int argc, char** argv){
    MPI_Init(&argc,&argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime(); // start timer

    vector<string> lines;
    string pattern;

    // ----------------- PROCESS 0 READS FILE -----------------
    if(rank==0){
        ifstream fin("phonebook.txt");
        string line;
        while(getline(fin,line)) lines.push_back(line);
        fin.close();

        cout << "Enter pattern to search: ";
        cin >> pattern;
    }

    // ----------------- BROADCAST PATTERN -----------------
    broadcast_string(pattern);

    // ----------------- BROADCAST LINES -----------------
    string allText = vector_to_string(lines);
    broadcast_string(allText);

    // Reconstruct lines on all processes
    lines = string_to_vector(allText);

    // ----------------- COMPUTE LOCAL RANGE -----------------
    auto [start,end] = get_work_range(lines.size(), rank, size);

    int localBestLen = 0;
    string localBestLine;

    for(int i=start;i<end;i++){
        int matchLen = longestMatch(lines[i], pattern);
        if(matchLen > localBestLen){
            localBestLen = matchLen;
            localBestLine = lines[i];
        }
    }

    // ----------------- REDUCE TO FIND GLOBAL BEST -----------------
    struct { int len; int rank; } localData{localBestLen, rank}, globalData;
    MPI_Reduce(&localData,&globalData,1,MPI_2INT,MPI_MAXLOC,0,MPI_COMM_WORLD);

    if(rank == globalData.rank){
        cout << "\nBest Matching Line:\n" << localBestLine << endl;
        cout << "Match Length: " << localBestLen << endl;
    }

    // ----------------- TIME PRINT -----------------
    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    // Each process prints its own time
    printf("Process %d time: %f seconds\n", rank, local_time);
    MPI_Barrier(MPI_COMM_WORLD);
    // Compute total (max) time across all processes
    double total_time;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank==0) printf("Total parallel time: %f seconds\n", total_time);

    MPI_Finalize();
    return 0;
}
