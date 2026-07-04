#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    std::cout
        << "Hello from rank "
        << rank
        << " of "
        << processCount
        << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout
            << "All MPI processes completed successfully."
            << std::endl;
    }

    MPI_Finalize();
    return 0;
}

