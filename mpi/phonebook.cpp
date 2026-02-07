#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string searchName;
    vector<string> lines;

    // ---------- ROOT READS INPUT ----------
    if (rank == 0) {
        cin >> searchName;

        ifstream file("phonebook.txt");
        string line;
        while (getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
    }

    // ---------- BROADCAST SEARCH NAME ----------
    int nameLen;
    if (rank == 0) nameLen = searchName.size();
    MPI_Bcast(&nameLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

    searchName.resize(nameLen);
    MPI_Bcast(&searchName[0], nameLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    // ---------- BROADCAST TOTAL LINES ----------
    int totalLines;
    if (rank == 0) totalLines = lines.size();
    MPI_Bcast(&totalLines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---------- DIVIDE WORK ----------
    int perProc = totalLines / size;
    int start = rank * perProc;
    int end = (rank == size - 1) ? totalLines : start + perProc;

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // ---------- LOCAL SEARCH ----------
    vector<string> localMatches;
    for (int i = start; i < end; i++) {
        if (rank == 0) {
            if (lines[i].find(searchName) == 0)
                localMatches.push_back(lines[i]);
        }
    }

    // ---------- SEND LINES TO OTHER PROCESSES ----------
    if (rank == 0) {
        for (int p = 1; p < size; p++) {
            int s = p * perProc;
            int e = (p == size - 1) ? totalLines : s + perProc;

            for (int i = s; i < e; i++) {
                int len = lines[i].size();
                MPI_Send(&len, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(lines[i].c_str(), len, MPI_CHAR, p, 0, MPI_COMM_WORLD);
            }
        }
    }

    // ---------- OTHER PROCESSES RECEIVE & SEARCH ----------
    if (rank != 0) {
        for (int i = start; i < end; i++) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            string line(len, ' ');
            MPI_Recv(&line[0], len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (line.find(searchName) == 0)
                localMatches.push_back(line);
        }
    }

    double t2 = MPI_Wtime();
    double localTime = t2 - t1;

    // ---------- SEND MATCHES BACK TO ROOT ----------
    int localCount = localMatches.size();
    MPI_Send(&localCount, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);

    for (auto &s : localMatches) {
        int len = s.size();
        MPI_Send(&len, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(s.c_str(), len, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    // ---------- ROOT COLLECTS ALL MATCHES ----------
    if (rank == 0) {
        vector<string> allMatches = localMatches;

        for (int p = 1; p < size; p++) {
            int cnt;
            MPI_Recv(&cnt, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int i = 0; i < cnt; i++) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                string s(len, ' ');
                MPI_Recv(&s[0], len, MPI_CHAR, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                allMatches.push_back(s);
            }
        }

        cout << "\nMatched Contacts:\n";
        for (auto &s : allMatches)
            cout << s << endl;
    }

    // ---------- TOTAL TIME ----------
    double totalTime;
    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "\nTotal Time Taken: " << totalTime << " seconds\n";

    MPI_Finalize();
    return 0;
}
