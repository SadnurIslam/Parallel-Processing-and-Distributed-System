#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>  // for rand()
#include <ctime>    // for srand(time(0))
using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes

    int K, M, N, P;

    if (rank == 0) {
        // Root process reads input
        cin >> K >> M >> N >> P;
    }

    // Broadcast input to all processes
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine which matrices this process will compute
    int per_proc = K / size;
    int start = rank * per_proc;
    int end = (rank == size - 1) ? K : start + per_proc;

    // Initialize matrices A, B, C (flattened 1D arrays)
    vector<double> A(M * N), B(N * P), C(M * P);

    // Seed random generator differently for each process
    srand(time(0) + rank);

    // Fill A and B with random numbers
    for (int i = 0; i < M * N; i++)
        A[i] = rand() % 10 + 1; // random 1-10
    for (int i = 0; i < N * P; i++)
        B[i] = rand() % 10 + 1; // random 1-10

    MPI_Barrier(MPI_COMM_WORLD); // synchronize before timing
    double t1 = MPI_Wtime();     // start local timer

    // Multiply matrices for assigned K matrices
    for (int k = start; k < end; k++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                C[i * P + j] = 0;
                for (int x = 0; x < N; x++) {
                    C[i * P + j] += A[i * N + x] * B[x * P + j];
                }
            }
        }
    }

    double t2 = MPI_Wtime();
    double local_time = t2 - t1; // local computation time
    double total_time;

    // Reduce to find max time among all processes (total time)
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print local time for each process
    cout << "Process " << rank << " time = " << local_time << " seconds\n";

    if (rank == 0) {
        cout << "\nMPI Total Time Taken: " << total_time << " seconds\n";

        // Print A[0]
        cout << "\nA[0]:\n";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++)
                cout << A[i * N + j] << " ";
            cout << endl;
        }

        // Print B[0]
        cout << "\nB[0]:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < P; j++)
                cout << B[i * P + j] << " ";
            cout << endl;
        }

        // Print C[0]
        cout << "\nC[0] = A[0] × B[0]:\n";
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++)
                cout << C[i * P + j] << " ";
            cout << endl;
        }
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}
