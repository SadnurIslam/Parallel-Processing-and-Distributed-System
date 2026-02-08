#include <mpi.h>
#include <bits/stdc++.h>
using namespace std;

// Count substring occurrences in a string
int countOccurrences(const string& text, const string& pat) {
    int count = 0;
    size_t pos = text.find(pat);
    while (pos != string::npos) {
        count++;
        pos = text.find(pat, pos + 1);
    }
    return count;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string paragraph, pattern;

    double start = MPI_Wtime();

    // -------- Rank 0 reads input --------
    if (rank == 0) {
        ifstream fin("input.txt");
        string line;
        while (getline(fin, line))
            paragraph += line + " ";
        fin.close();

        cout << "Enter pattern (like %x%): ";
        cin >> pattern;

        // remove % from %x%
        pattern = pattern.substr(1, pattern.size() - 2);
    }

    // -------- Broadcast pattern --------
    int patLen;
    if (rank == 0) patLen = pattern.size();
    MPI_Bcast(&patLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

    pattern.resize(patLen);
    MPI_Bcast(&pattern[0], patLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    // -------- Broadcast paragraph --------
    int textLen;
    if (rank == 0) textLen = paragraph.size();
    MPI_Bcast(&textLen, 1, MPI_INT, 0, MPI_COMM_WORLD);

    paragraph.resize(textLen);
    MPI_Bcast(&paragraph[0], textLen, MPI_CHAR, 0, MPI_COMM_WORLD);

    // -------- Divide work --------
    int n = paragraph.size();
    int perProc = n / size;
    int startIdx = rank * perProc;
    int endIdx = (rank == size - 1) ? n : startIdx + perProc + pattern.size();

    string localText = paragraph.substr(startIdx, endIdx - startIdx);

    // -------- Local count --------
    int localCount = countOccurrences(localText, pattern);

    // -------- Reduce counts --------
    int totalCount;
    MPI_Reduce(&localCount, &totalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double localTime = end - start;
    double totalTime;

    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "\nTotal Occurrences: " << totalCount << endl;
        cout << "Total MPI Time: " << totalTime << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
