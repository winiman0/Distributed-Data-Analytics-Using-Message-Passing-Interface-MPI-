#include <mpi.h>
#include <iostream>
#include <numeric>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto mark = [rank](const char* message) {
        std::cerr << "[Rank " << rank << "] "
                  << message << std::endl;
    };

    mark("MPI_Init successful");

    int n = 1000000;

    mark("Before MPI_Bcast");
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    mark("After MPI_Bcast");

    int localCount = n / size;

    std::vector<double> completeData;

    if (rank == 0) {
        completeData.assign(n, 1.0);
    }

    std::vector<double> localData(localCount);

    mark("Before MPI_Scatter");

    MPI_Scatter(
        rank == 0 ? completeData.data() : nullptr,
        localCount,
        MPI_DOUBLE,
        localData.data(),
        localCount,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    mark("After MPI_Scatter");

    double localSum =
        std::accumulate(localData.begin(), localData.end(), 0.0);

    double globalSum = 0.0;

    mark("Before MPI_Reduce");

    MPI_Reduce(
        &localSum,
        &globalSum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    mark("After MPI_Reduce");

    if (rank == 0) {
        std::cout << "Global sum: " << globalSum << '\n';
        std::cout << "MPI data communication passed.\n";
    }

    MPI_Finalize();
    return 0;
}