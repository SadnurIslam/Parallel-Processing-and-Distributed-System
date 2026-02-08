#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();

    vector<string> lines;
    string filename = "input.txt";

    // ---------- Rank 0 reads file ----------
    if(rank == 0) {
        ifstream fin(filename);
        string line;
        while(getline(fin, line)) lines.push_back(line);
        fin.close();
    }

    // ---------- Broadcast number of lines ----------
    int totalLines;
    if(rank == 0) totalLines = lines.size();
    MPI_Bcast(&totalLines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------- Broadcast all text ----------
    string allText;
    if(rank == 0) {
        for(auto &l : lines) allText += l + "\n";
    }

    int textSize;
    if(rank == 0) textSize = allText.size();
    MPI_Bcast(&textSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0) allText.resize(textSize);
    MPI_Bcast(&allText[0], textSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    // ---------- Rebuild lines ----------
    lines.clear();
    stringstream ss(allText);
    string temp;
    while(getline(ss, temp)) lines.push_back(temp);

    // ---------- Divide work ----------
    int perProc = totalLines / size;
    int startLine = rank * perProc;
    int endLine = (rank == size - 1) ? totalLines : startLine + perProc;

    map<string,int> localCount;

    // ---------- Local word count ----------
    for(int i = startLine; i < endLine; i++) {
        stringstream ls(lines[i]);
        string word;
        while(ls >> word) {
            localCount[word]++;
        }
    }

    // ---------- Convert local map to string ----------
    string localData;
    for(auto &p : localCount)
        localData += p.first + " " + to_string(p.second) + "\n";

    int localSize = localData.size();
    MPI_Gather(&localSize, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------- Rank 0 merges ----------
    map<string,int> globalCount;

    if(rank == 0) {
        globalCount = localCount;
        for(int p = 1; p < size; p++) {
            int sz;
            MPI_Recv(&sz, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            string buf(sz, ' ');
            MPI_Recv(&buf[0], sz, MPI_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            string w; int c;
            stringstream s(buf);
            while(s >> w >> c) globalCount[w] += c;
        }
    } else {
        MPI_Send(&localSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&localData[0], localSize, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();
    double localTime = end - start;
    double totalTime;

    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---------- Print result ----------
    if(rank == 0) {
        vector<pair<int,string>> v;
        for(auto &p : globalCount)
            v.push_back({p.second, p.first});

        sort(v.rbegin(), v.rend());

        cout << "\nTop 10 words:\n";
        for(int i = 0; i < min(10,(int)v.size()); i++)
            cout << v[i].second << " -> " << v[i].first << endl;

        cout << "\nTotal MPI Time: " << totalTime << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
