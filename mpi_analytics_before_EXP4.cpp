#include "data_io.h"
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// ============================================================
// Structures
// ============================================================

struct Statistics {
    double mean{};
    double variance{};
    double standardDeviation{};
    double minimum{};
    double maximum{};
};

struct MovingAverageResult {
    long long count{};
    std::vector<double> firstValues;
    double lastValue{};
};

struct TimingRecord {
    std::string task;
    double timeMs{};
    double resultValue{};
};

// ============================================================
// MPI timing
// ============================================================

double start_mpi_timer() {
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime();
}

double stop_mpi_timer(double startTime, int rank) {
    MPI_Barrier(MPI_COMM_WORLD);

    const double localElapsedMs =
        (MPI_Wtime() - startTime) * 1000.0;

    double globalElapsedMs = 0.0;

    // The parallel task finishes when the slowest process finishes.
    MPI_Reduce(
        &localElapsedMs,
        &globalElapsedMs,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    return rank == 0 ? globalElapsedMs : 0.0;
}

// ============================================================
// Save MPI results to CSV
// ============================================================

bool save_mpi_results(
    const std::string& outputFilename,
    const std::string& datasetFilename,
    int processCount,
    long long recordCount,
    const std::vector<TimingRecord>& records
) {
    bool writeHeader = true;

    {
        std::ifstream existingFile(
            outputFilename,
            std::ios::binary | std::ios::ate
        );

        if (
            existingFile.is_open() &&
            existingFile.tellg() > 0
        ) {
            writeHeader = false;
        }
    }

    std::ofstream outputFile(
        outputFilename,
        std::ios::app
    );

    if (!outputFile.is_open()) {
        std::cerr
            << "WARNING: Cannot open CSV output file: "
            << outputFilename
            << '\n';

        return false;
    }

    if (writeHeader) {
        outputFile
            << "mode,processes,dataset,records,"
            << "task,time_ms,result_value\n";
    }

    outputFile
        << std::fixed
        << std::setprecision(6);

    for (const TimingRecord& record : records) {
        outputFile
            << "mpi,"
            << processCount
            << ",\""
            << datasetFilename
            << "\","
            << recordCount
            << ","
            << record.task
            << ","
            << record.timeMs
            << ","
            << record.resultValue
            << '\n';
    }

    return true;
}

// ============================================================
// Calculate MPI partition sizes
// ============================================================

void calculate_counts_and_displacements(
    long long globalN,
    int processCount,
    std::vector<int>& counts,
    std::vector<int>& displacements
) {
    const long long base =
        globalN / processCount;

    const long long remainder =
        globalN % processCount;

    for (
        int process = 0;
        process < processCount;
        ++process
    ) {
        const long long count =
            base +
            (
                process < remainder
                ? 1LL
                : 0LL
            );

        counts[
            static_cast<std::size_t>(process)
        ] = static_cast<int>(count);
    }

    displacements[0] = 0;

    for (
        int process = 1;
        process < processCount;
        ++process
    ) {
        displacements[
            static_cast<std::size_t>(process)
        ] =
            displacements[
                static_cast<std::size_t>(
                    process - 1
                )
            ] +
            counts[
                static_cast<std::size_t>(
                    process - 1
                )
            ];
    }
}

// ============================================================
// Parallel moving average
//
// Window size = 5.
// Each rank receives four ending values from the previous rank.
// ============================================================

MovingAverageResult calculate_parallel_moving_average(
    const std::vector<double>& localData,
    std::size_t windowSize,
    int rank,
    int processCount
) {
    MovingAverageResult result{};

    if (
        windowSize == 0 ||
        localData.empty()
    ) {
        return result;
    }

    const std::size_t haloCount =
        windowSize - 1;

    std::vector<double> halo(
        haloCount,
        0.0
    );

    const int sendTo =
        rank < processCount - 1
        ? rank + 1
        : MPI_PROC_NULL;

    const int receiveFrom =
        rank > 0
        ? rank - 1
        : MPI_PROC_NULL;

    const double* sendBuffer = nullptr;

    if (
        haloCount > 0 &&
        localData.size() >= haloCount
    ) {
        sendBuffer =
            localData.data() +
            (
                localData.size() -
                haloCount
            );
    }

    MPI_Sendrecv(
        sendBuffer,
        static_cast<int>(haloCount),
        MPI_DOUBLE,
        sendTo,
        100,

        halo.data(),
        static_cast<int>(haloCount),
        MPI_DOUBLE,
        receiveFrom,
        100,

        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    const auto store_average =
        [&result](double average) {
            result.count++;
            result.lastValue = average;

            if (
                result.firstValues.size() < 5
            ) {
                result.firstValues.push_back(
                    average
                );
            }
        };

    // Rank 0 does not require a halo.
    if (rank == 0) {
        if (localData.size() < windowSize) {
            return result;
        }

        long double windowSum = 0.0L;

        for (
            std::size_t i = 0;
            i < windowSize;
            ++i
        ) {
            windowSum +=
                static_cast<long double>(
                    localData[i]
                );
        }

        store_average(
            static_cast<double>(
                windowSum /
                static_cast<long double>(
                    windowSize
                )
            )
        );

        for (
            std::size_t i = windowSize;
            i < localData.size();
            ++i
        ) {
            windowSum -=
                static_cast<long double>(
                    localData[
                        i - windowSize
                    ]
                );

            windowSum +=
                static_cast<long double>(
                    localData[i]
                );

            store_average(
                static_cast<double>(
                    windowSum /
                    static_cast<long double>(
                        windowSize
                    )
                )
            );
        }

        return result;
    }

    // Other ranks calculate across their partition boundary.
    const auto combined_value =
        [&halo, &localData, haloCount]
        (std::size_t index) -> double {

            if (index < haloCount) {
                return halo[index];
            }

            return localData[
                index - haloCount
            ];
        };

    const std::size_t combinedSize =
        haloCount +
        localData.size();

    long double windowSum = 0.0L;

    for (
        std::size_t i = 0;
        i < windowSize;
        ++i
    ) {
        windowSum +=
            static_cast<long double>(
                combined_value(i)
            );
    }

    store_average(
        static_cast<double>(
            windowSum /
            static_cast<long double>(
                windowSize
            )
        )
    );

    for (
        std::size_t i = windowSize;
        i < combinedSize;
        ++i
    ) {
        windowSum -=
            static_cast<long double>(
                combined_value(
                    i - windowSize
                )
            );

        windowSum +=
            static_cast<long double>(
                combined_value(i)
            );

        store_average(
            static_cast<double>(
                windowSum /
                static_cast<long double>(
                    windowSize
                )
            )
        );
    }

    return result;
}

// ============================================================
// Distributed sample sort
// ============================================================

bool distributed_sample_sort(
    std::vector<double>& localData,
    int rank,
    int processCount,
    long long globalN,
    long long& globalSortedCount
) {
    // Step 1: Sort each local partition.
    std::sort(
        localData.begin(),
        localData.end()
    );

    if (processCount == 1) {
        globalSortedCount =
            static_cast<long long>(
                localData.size()
            );

        return std::is_sorted(
            localData.begin(),
            localData.end()
        );
    }

    // Step 2: Select regular samples.
    const int samplesPerRank =
        processCount - 1;

    std::vector<double> localSamples(
        static_cast<std::size_t>(
            samplesPerRank
        ),
        0.0
    );

    for (
        int i = 1;
        i < processCount;
        ++i
    ) {
        std::size_t index =
            (
                static_cast<std::size_t>(i) *
                localData.size()
            ) /
            static_cast<std::size_t>(
                processCount
            );

        if (index >= localData.size()) {
            index =
                localData.size() - 1;
        }

        localSamples[
            static_cast<std::size_t>(
                i - 1
            )
        ] = localData[index];
    }

    // Step 3: Gather samples at Rank 0.
    std::vector<double> allSamples;

    if (rank == 0) {
        allSamples.resize(
            static_cast<std::size_t>(
                processCount *
                samplesPerRank
            )
        );
    }

    MPI_Gather(
        localSamples.data(),
        samplesPerRank,
        MPI_DOUBLE,

        rank == 0
            ? allSamples.data()
            : nullptr,

        samplesPerRank,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Step 4: Rank 0 chooses splitters.
    std::vector<double> splitters(
        static_cast<std::size_t>(
            samplesPerRank
        ),
        0.0
    );

    if (rank == 0) {
        std::sort(
            allSamples.begin(),
            allSamples.end()
        );

        for (
            int i = 1;
            i < processCount;
            ++i
        ) {
            std::size_t index =
                (
                    static_cast<std::size_t>(i) *
                    allSamples.size()
                ) /
                static_cast<std::size_t>(
                    processCount
                );

            if (
                index >= allSamples.size()
            ) {
                index =
                    allSamples.size() - 1;
            }

            splitters[
                static_cast<std::size_t>(
                    i - 1
                )
            ] = allSamples[index];
        }
    }

    MPI_Bcast(
        splitters.data(),
        samplesPerRank,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Step 5: Split local data into buckets.
    std::vector<int> sendCounts(
        static_cast<std::size_t>(
            processCount
        ),
        0
    );

    std::vector<int> sendDisplacements(
        static_cast<std::size_t>(
            processCount
        ),
        0
    );

    std::size_t previousBoundary = 0;

    for (
        int bucket = 0;
        bucket < samplesPerRank;
        ++bucket
    ) {
        const auto boundaryIterator =
            std::upper_bound(
                localData.begin() +
                static_cast<std::ptrdiff_t>(
                    previousBoundary
                ),
                localData.end(),
                splitters[
                    static_cast<std::size_t>(
                        bucket
                    )
                ]
            );

        const std::size_t boundary =
            static_cast<std::size_t>(
                std::distance(
                    localData.begin(),
                    boundaryIterator
                )
            );

        sendCounts[
            static_cast<std::size_t>(
                bucket
            )
        ] =
            static_cast<int>(
                boundary -
                previousBoundary
            );

        previousBoundary = boundary;
    }

    sendCounts[
        static_cast<std::size_t>(
            processCount - 1
        )
    ] =
        static_cast<int>(
            localData.size() -
            previousBoundary
        );

    for (
        int i = 1;
        i < processCount;
        ++i
    ) {
        sendDisplacements[
            static_cast<std::size_t>(i)
        ] =
            sendDisplacements[
                static_cast<std::size_t>(
                    i - 1
                )
            ] +
            sendCounts[
                static_cast<std::size_t>(
                    i - 1
                )
            ];
    }

    // Step 6: Exchange bucket sizes.
    std::vector<int> receiveCounts(
        static_cast<std::size_t>(
            processCount
        ),
        0
    );

    MPI_Alltoall(
        sendCounts.data(),
        1,
        MPI_INT,

        receiveCounts.data(),
        1,
        MPI_INT,

        MPI_COMM_WORLD
    );

    std::vector<int> receiveDisplacements(
        static_cast<std::size_t>(
            processCount
        ),
        0
    );

    for (
        int i = 1;
        i < processCount;
        ++i
    ) {
        receiveDisplacements[
            static_cast<std::size_t>(i)
        ] =
            receiveDisplacements[
                static_cast<std::size_t>(
                    i - 1
                )
            ] +
            receiveCounts[
                static_cast<std::size_t>(
                    i - 1
                )
            ];
    }

    const int totalReceiveCount =
        std::accumulate(
            receiveCounts.begin(),
            receiveCounts.end(),
            0
        );

    std::vector<double> receivedData(
        static_cast<std::size_t>(
            totalReceiveCount
        )
    );

    // Step 7: Exchange buckets.
    MPI_Alltoallv(
        localData.data(),
        sendCounts.data(),
        sendDisplacements.data(),
        MPI_DOUBLE,

        receivedData.data(),
        receiveCounts.data(),
        receiveDisplacements.data(),
        MPI_DOUBLE,

        MPI_COMM_WORLD
    );

    // Step 8: Sort the received partition.
    std::sort(
        receivedData.begin(),
        receivedData.end()
    );

    localData.swap(receivedData);

    // Validate local sorting.
    const int localSorted =
        std::is_sorted(
            localData.begin(),
            localData.end()
        )
        ? 1
        : 0;

    int allLocallySorted = 0;

    MPI_Allreduce(
        &localSorted,
        &allLocallySorted,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    const long long localSortedCount =
        static_cast<long long>(
            localData.size()
        );

    MPI_Allreduce(
        &localSortedCount,
        &globalSortedCount,
        1,
        MPI_LONG_LONG_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );

    // Validate the ordering between ranks.
    const double localMinimum =
        localData.empty()
        ? std::numeric_limits<double>::infinity()
        : localData.front();

    const double localMaximum =
        localData.empty()
        ? -std::numeric_limits<double>::infinity()
        : localData.back();

    std::vector<double> rankMinimums;
    std::vector<double> rankMaximums;
    std::vector<long long> rankCounts;

    if (rank == 0) {
        rankMinimums.resize(
            static_cast<std::size_t>(
                processCount
            )
        );

        rankMaximums.resize(
            static_cast<std::size_t>(
                processCount
            )
        );

        rankCounts.resize(
            static_cast<std::size_t>(
                processCount
            )
        );
    }

    MPI_Gather(
        &localMinimum,
        1,
        MPI_DOUBLE,

        rank == 0
            ? rankMinimums.data()
            : nullptr,

        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &localMaximum,
        1,
        MPI_DOUBLE,

        rank == 0
            ? rankMaximums.data()
            : nullptr,

        1,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Gather(
        &localSortedCount,
        1,
        MPI_LONG_LONG_INT,

        rank == 0
            ? rankCounts.data()
            : nullptr,

        1,
        MPI_LONG_LONG_INT,
        0,
        MPI_COMM_WORLD
    );

    int globallyOrdered = 1;

    if (rank == 0) {
        int previousNonEmptyRank = -1;

        for (
            int currentRank = 0;
            currentRank < processCount;
            ++currentRank
        ) {
            if (
                rankCounts[
                    static_cast<std::size_t>(
                        currentRank
                    )
                ] == 0
            ) {
                continue;
            }

            if (
                previousNonEmptyRank >= 0
            ) {
                const double previousMaximum =
                    rankMaximums[
                        static_cast<std::size_t>(
                            previousNonEmptyRank
                        )
                    ];

                const double currentMinimum =
                    rankMinimums[
                        static_cast<std::size_t>(
                            currentRank
                        )
                    ];

                if (
                    previousMaximum >
                    currentMinimum
                ) {
                    globallyOrdered = 0;
                    break;
                }
            }

            previousNonEmptyRank =
                currentRank;
        }

        if (
            globalSortedCount != globalN
        ) {
            globallyOrdered = 0;
        }
    }

    MPI_Bcast(
        &globallyOrdered,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    return
        allLocallySorted == 1 &&
        globallyOrdered == 1;
}

// ============================================================
// Main program
// ============================================================

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 0;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &rank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &processCount
    );

    std::string filename =
        "data/dataset_1M.bin";

    if (argc >= 2) {
        filename = argv[1];
    }

    const std::string csvFilename =
        "results/mpi_results.csv";

    std::vector<TimingRecord>
        timingRecords;

    Dataset dataset;
    dataset.N = 0;

    long long globalN = 0;
    int loadSuccessful = 1;
    double loadingTimeMs = 0.0;

    // ========================================================
    // Rank 0 loads the complete dataset
    // ========================================================

    if (rank == 0) {
        std::cout
            << "=====================================\n"
            << " MPI Analytics Program\n"
            << "=====================================\n"
            << "MPI processes: "
            << processCount
            << '\n'
            << "Dataset file : "
            << filename
            << "\n\n";

        const double loadStart =
            MPI_Wtime();

        dataset =
            load_dataset(filename);

        loadingTimeMs =
            (
                MPI_Wtime() -
                loadStart
            ) *
            1000.0;

        if (
            dataset.N <= 0 ||
            dataset.col1.size() !=
                static_cast<std::size_t>(
                    dataset.N
                ) ||
            dataset.col2.size() !=
                static_cast<std::size_t>(
                    dataset.N
                )
        ) {
            loadSuccessful = 0;
        }
        else {
            globalN =
                dataset.N;
        }
    }

    MPI_Bcast(
        &loadSuccessful,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (loadSuccessful == 0) {
        if (rank == 0) {
            std::cerr
                << "ERROR: Dataset could not "
                << "be loaded.\n";
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    MPI_Bcast(
        &globalN,
        1,
        MPI_LONG_LONG_INT,
        0,
        MPI_COMM_WORLD
    );

    if (
        globalN >
        std::numeric_limits<int>::max()
    ) {
        if (rank == 0) {
            std::cerr
                << "ERROR: Dataset is too large "
                << "for the current MPI_Scatterv "
                << "integer displacement format.\n";
        }

        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // ========================================================
    // Partition dataset
    // ========================================================

    std::vector<int> counts(
        static_cast<std::size_t>(
            processCount
        ),
        0
    );

    std::vector<int> displacements(
        static_cast<std::size_t>(
            processCount
        ),
        0
    );

    calculate_counts_and_displacements(
        globalN,
        processCount,
        counts,
        displacements
    );

    const int localCount =
        counts[
            static_cast<std::size_t>(
                rank
            )
        ];

    std::vector<double> localCol1(
        static_cast<std::size_t>(
            localCount
        )
    );

    std::vector<double> localCol2(
        static_cast<std::size_t>(
            localCount
        )
    );

    // ========================================================
    // Scatter both columns
    // ========================================================

    const double distributionStart =
        start_mpi_timer();

    MPI_Scatterv(
        rank == 0
            ? dataset.col1.data()
            : nullptr,

        counts.data(),
        displacements.data(),
        MPI_DOUBLE,

        localCol1.data(),
        localCount,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD
    );

    MPI_Scatterv(
        rank == 0
            ? dataset.col2.data()
            : nullptr,

        counts.data(),
        displacements.data(),
        MPI_DOUBLE,

        localCol2.data(),
        localCount,
        MPI_DOUBLE,

        0,
        MPI_COMM_WORLD
    );

    const double distributionTimeMs =
        stop_mpi_timer(
            distributionStart,
            rank
        );

    if (rank == 0) {
        // Release the complete dataset after distribution.
        std::vector<double>().swap(
            dataset.col1
        );

        std::vector<double>().swap(
            dataset.col2
        );

        timingRecords.push_back({
            "data_loading",
            loadingTimeMs,
            static_cast<double>(
                globalN
            )
        });

        timingRecords.push_back({
            "data_distribution",
            distributionTimeMs,
            static_cast<double>(
                globalN
            )
        });

        std::cout
            << "Records loaded    : "
            << globalN
            << '\n'
            << "Loading time     : "
            << loadingTimeMs
            << " ms\n"
            << "Distribution time: "
            << distributionTimeMs
            << " ms\n\n"
            << "Partition sizes\n"
            << "---------------\n";

        for (
            int process = 0;
            process < processCount;
            ++process
        ) {
            std::cout
                << "Rank "
                << process
                << " : "
                << counts[
                    static_cast<std::size_t>(
                        process
                    )
                ]
                << " records\n";
        }

        std::cout << '\n';
    }

    // ========================================================
    // Task 1: Parallel basic statistics
    // ========================================================

    const double statisticsStart =
        start_mpi_timer();

    long double localSum = 0.0L;
    long double localSumSquares = 0.0L;

    double localMinimum =
        std::numeric_limits<double>::max();

    double localMaximum =
        std::numeric_limits<double>::lowest();

    for (double value : localCol1) {
        const long double longValue =
            static_cast<long double>(
                value
            );

        localSum += longValue;

        localSumSquares +=
            longValue *
            longValue;

        localMinimum =
            std::min(
                localMinimum,
                value
            );

        localMaximum =
            std::max(
                localMaximum,
                value
            );
    }

    long double globalSum = 0.0L;
    long double globalSumSquares = 0.0L;

    double globalMinimum = 0.0;
    double globalMaximum = 0.0;

    MPI_Reduce(
        &localSum,
        &globalSum,
        1,
        MPI_LONG_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localSumSquares,
        &globalSumSquares,
        1,
        MPI_LONG_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localMinimum,
        &globalMinimum,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &localMaximum,
        &globalMaximum,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    Statistics statistics{};

    if (rank == 0) {
        const long double count =
            static_cast<long double>(
                globalN
            );

        const long double mean =
            globalSum / count;

        long double variance =
            (
                globalSumSquares /
                count
            ) -
            (
                mean *
                mean
            );

        if (variance < 0.0L) {
            variance = 0.0L;
        }

        statistics.mean =
            static_cast<double>(
                mean
            );

        statistics.variance =
            static_cast<double>(
                variance
            );

        statistics.standardDeviation =
            std::sqrt(
                statistics.variance
            );

        statistics.minimum =
            globalMinimum;

        statistics.maximum =
            globalMaximum;
    }

    double statisticsValues[5] = {};

    if (rank == 0) {
        statisticsValues[0] =
            statistics.mean;

        statisticsValues[1] =
            statistics.variance;

        statisticsValues[2] =
            statistics.standardDeviation;

        statisticsValues[3] =
            statistics.minimum;

        statisticsValues[4] =
            statistics.maximum;
    }

    MPI_Bcast(
        statisticsValues,
        5,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    statistics.mean =
        statisticsValues[0];

    statistics.variance =
        statisticsValues[1];

    statistics.standardDeviation =
        statisticsValues[2];

    statistics.minimum =
        statisticsValues[3];

    statistics.maximum =
        statisticsValues[4];

    const double statisticsTimeMs =
        stop_mpi_timer(
            statisticsStart,
            rank
        );

    // ========================================================
    // Task 2: Parallel histogram
    // ========================================================

    const int numberOfBins = 10;

    const double binWidth =
        (
            statistics.maximum -
            statistics.minimum
        ) /
        static_cast<double>(
            numberOfBins
        );

    std::vector<long long> localHistogram(
        static_cast<std::size_t>(
            numberOfBins
        ),
        0
    );

    std::vector<long long> globalHistogram(
        static_cast<std::size_t>(
            numberOfBins
        ),
        0
    );

    const double histogramStart =
        start_mpi_timer();

    for (double value : localCol1) {
        int binIndex =
            static_cast<int>(
                (
                    value -
                    statistics.minimum
                ) /
                binWidth
            );

        if (
            binIndex >= numberOfBins
        ) {
            binIndex =
                numberOfBins - 1;
        }

        if (binIndex < 0) {
            binIndex = 0;
        }

        localHistogram[
            static_cast<std::size_t>(
                binIndex
            )
        ]++;
    }

    MPI_Reduce(
        localHistogram.data(),
        globalHistogram.data(),
        numberOfBins,
        MPI_LONG_LONG_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    const double histogramTimeMs =
        stop_mpi_timer(
            histogramStart,
            rank
        );

    // ========================================================
    // Task 3: Parallel Pearson correlation
    // ========================================================

    const double correlationStart =
        start_mpi_timer();

    long double localCorrelationValues[5] = {};

    for (
        int i = 0;
        i < localCount;
        ++i
    ) {
        const long double x =
            static_cast<long double>(
                localCol1[
                    static_cast<std::size_t>(
                        i
                    )
                ]
            );

        const long double y =
            static_cast<long double>(
                localCol2[
                    static_cast<std::size_t>(
                        i
                    )
                ]
            );

        localCorrelationValues[0] += x;
        localCorrelationValues[1] += y;
        localCorrelationValues[2] += x * x;
        localCorrelationValues[3] += y * y;
        localCorrelationValues[4] += x * y;
    }

    long double globalCorrelationValues[5] = {};

    MPI_Reduce(
        localCorrelationValues,
        globalCorrelationValues,
        5,
        MPI_LONG_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    double correlation = 0.0;

    if (rank == 0) {
        const long double n =
            static_cast<long double>(
                globalN
            );

        const long double sumX =
            globalCorrelationValues[0];

        const long double sumY =
            globalCorrelationValues[1];

        const long double sumXSquare =
            globalCorrelationValues[2];

        const long double sumYSquare =
            globalCorrelationValues[3];

        const long double sumXY =
            globalCorrelationValues[4];

        const long double numerator =
            (
                n *
                sumXY
            ) -
            (
                sumX *
                sumY
            );

        const long double denominatorX =
            (
                n *
                sumXSquare
            ) -
            (
                sumX *
                sumX
            );

        const long double denominatorY =
            (
                n *
                sumYSquare
            ) -
            (
                sumY *
                sumY
            );

        if (
            denominatorX > 0.0L &&
            denominatorY > 0.0L
        ) {
            correlation =
                static_cast<double>(
                    numerator /
                    std::sqrt(
                        denominatorX *
                        denominatorY
                    )
                );
        }
    }

    const double correlationTimeMs =
        stop_mpi_timer(
            correlationStart,
            rank
        );

    // col2 is no longer required.
    std::vector<double>().swap(
        localCol2
    );

    // ========================================================
    // Task 4: Parallel moving average
    // ========================================================

    const std::size_t movingAverageWindow = 5;

    const double movingAverageStart =
        start_mpi_timer();

    const MovingAverageResult
        localMovingAverage =
            calculate_parallel_moving_average(
                localCol1,
                movingAverageWindow,
                rank,
                processCount
            );

    long long globalMovingAverageCount = 0;

    MPI_Reduce(
        &localMovingAverage.count,
        &globalMovingAverageCount,
        1,
        MPI_LONG_LONG_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // Last rank owns the final moving average.
    double globalLastMovingAverage =
        localMovingAverage.lastValue;

    MPI_Bcast(
        &globalLastMovingAverage,
        1,
        MPI_DOUBLE,
        processCount - 1,
        MPI_COMM_WORLD
    );

    const double movingAverageTimeMs =
        stop_mpi_timer(
            movingAverageStart,
            rank
        );

    // ========================================================
    // Task 5: Parallel Z-score detection
    // ========================================================

    const double zScoreThreshold = 3.0;

    const double outlierStart =
        start_mpi_timer();

    long long localOutlierCount = 0;

    if (
        statistics.standardDeviation > 0.0
    ) {
        for (double value : localCol1) {
            const double zScore =
                (
                    value -
                    statistics.mean
                ) /
                statistics.standardDeviation;

            if (
                std::fabs(zScore) >
                zScoreThreshold
            ) {
                localOutlierCount++;
            }
        }
    }

    long long globalOutlierCount = 0;

    MPI_Reduce(
        &localOutlierCount,
        &globalOutlierCount,
        1,
        MPI_LONG_LONG_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    const double outlierTimeMs =
        stop_mpi_timer(
            outlierStart,
            rank
        );

    // ========================================================
    // Task 6: Distributed sample sort
    // ========================================================

    const double sortingStart =
        start_mpi_timer();

    long long globalSortedCount = 0;

    const bool sortedCorrectly =
        distributed_sample_sort(
            localCol1,
            rank,
            processCount,
            globalN,
            globalSortedCount
        );

    const double sortingTimeMs =
        stop_mpi_timer(
            sortingStart,
            rank
        );

    // ========================================================
    // Rank 0 prints and saves final results
    // ========================================================

    if (rank == 0) {
        long long histogramTotal = 0;

        for (
            long long count :
            globalHistogram
        ) {
            histogramTotal += count;
        }

        const long long
            expectedMovingAverageCount =
                globalN -
                static_cast<long long>(
                    movingAverageWindow
                ) +
                1;

        const double totalSixTasksMs =
            statisticsTimeMs +
            histogramTimeMs +
            correlationTimeMs +
            movingAverageTimeMs +
            outlierTimeMs +
            sortingTimeMs;

        const double
            totalWithLoadingDistributionMs =
                loadingTimeMs +
                distributionTimeMs +
                totalSixTasksMs;

        timingRecords.push_back({
            "basic_statistics",
            statisticsTimeMs,
            statistics.mean
        });

        timingRecords.push_back({
            "histogram",
            histogramTimeMs,
            static_cast<double>(
                histogramTotal
            )
        });

        timingRecords.push_back({
            "pearson_correlation",
            correlationTimeMs,
            correlation
        });

        timingRecords.push_back({
            "moving_average",
            movingAverageTimeMs,
            static_cast<double>(
                globalMovingAverageCount
            )
        });

        timingRecords.push_back({
            "zscore_outlier_detection",
            outlierTimeMs,
            static_cast<double>(
                globalOutlierCount
            )
        });

        timingRecords.push_back({
            "sorting",
            sortingTimeMs,
            sortedCorrectly
                ? 1.0
                : 0.0
        });

        timingRecords.push_back({
            "total_six_tasks",
            totalSixTasksMs,
            0.0
        });

        timingRecords.push_back({
            "total_with_loading_distribution",
            totalWithLoadingDistributionMs,
            0.0
        });

        std::cout
            << std::fixed
            << std::setprecision(6);

        std::cout
            << "Parallel Basic Statistics\n"
            << "-------------------------\n"
            << "Mean               : "
            << statistics.mean
            << '\n'
            << "Variance           : "
            << statistics.variance
            << '\n'
            << "Standard deviation : "
            << statistics.standardDeviation
            << '\n'
            << "Minimum            : "
            << statistics.minimum
            << '\n'
            << "Maximum            : "
            << statistics.maximum
            << '\n'
            << "Execution time     : "
            << statisticsTimeMs
            << " ms\n\n";

        std::cout
            << "Parallel Histogram\n"
            << "------------------\n";

        for (
            int i = 0;
            i < numberOfBins;
            ++i
        ) {
            std::cout
                << "Bin "
                << (i + 1)
                << " : "
                << globalHistogram[
                    static_cast<std::size_t>(
                        i
                    )
                ]
                << '\n';
        }

        std::cout
            << "Histogram total    : "
            << histogramTotal
            << '\n'
            << "Expected total     : "
            << globalN
            << '\n'
            << "Execution time     : "
            << histogramTimeMs
            << " ms\n\n";

        std::cout
            << "Parallel Pearson Correlation\n"
            << "----------------------------\n"
            << "Correlation        : "
            << correlation
            << '\n'
            << "Execution time     : "
            << correlationTimeMs
            << " ms\n\n";

        std::cout
            << "Parallel Moving Average\n"
            << "-----------------------\n"
            << "Window size        : "
            << movingAverageWindow
            << '\n'
            << "Output count       : "
            << globalMovingAverageCount
            << '\n'
            << "Expected count     : "
            << expectedMovingAverageCount
            << '\n'
            << "First averages     : ";

        for (
            std::size_t i = 0;
            i <
            localMovingAverage.firstValues.size();
            ++i
        ) {
            std::cout
                << localMovingAverage.firstValues[i];

            if (
                i + 1 <
                localMovingAverage.firstValues.size()
            ) {
                std::cout << ", ";
            }
        }

        std::cout
            << '\n'
            << "Last average       : "
            << globalLastMovingAverage
            << '\n'
            << "Execution time     : "
            << movingAverageTimeMs
            << " ms\n\n";

        std::cout
            << "Parallel Z-score Outlier Detection\n"
            << "----------------------------------\n"
            << "Threshold          : "
            << zScoreThreshold
            << '\n'
            << "Outlier count      : "
            << globalOutlierCount
            << '\n'
            << "Execution time     : "
            << outlierTimeMs
            << " ms\n\n";

        std::cout
            << "Distributed Sample Sort\n"
            << "-----------------------\n"
            << "Sorted correctly   : "
            << (
                sortedCorrectly
                ? "Yes"
                : "No"
            )
            << '\n'
            << "Sorted record count: "
            << globalSortedCount
            << '\n'
            << "Expected count     : "
            << globalN
            << '\n'
            << "Execution time     : "
            << sortingTimeMs
            << " ms\n\n";

        std::cout
            << "Timing Summary\n"
            << "--------------\n"
            << "Six-task total     : "
            << totalSixTasksMs
            << " ms\n"
            << "Including loading and distribution: "
            << totalWithLoadingDistributionMs
            << " ms\n\n";

        if (
            save_mpi_results(
                csvFilename,
                filename,
                processCount,
                globalN,
                timingRecords
            )
        ) {
            std::cout
                << "Results saved to   : "
                << csvFilename
                << '\n';
        }

        std::cout
            << "\nMPI processing completed "
            << "successfully.\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

