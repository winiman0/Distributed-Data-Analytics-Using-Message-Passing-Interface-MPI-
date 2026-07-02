#include "data_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ============================================================
// Result structures
// ============================================================

struct Statistics {
    double mean;
    double variance;
    double standardDeviation;
    double minimum;
    double maximum;
};

struct MovingAverageResult {
    std::size_t count;
    std::vector<double> firstValues;
    double lastValue;
};

struct OutlierResult {
    long long count;
    std::vector<std::size_t> sampleIndexes;
};

struct TimingRecord {
    std::string task;
    double timeMs;
    double resultValue;
};

// ============================================================
// Task 1: Basic statistics
// Mean, variance, standard deviation, minimum and maximum
// ============================================================

Statistics calculate_statistics(
    const std::vector<double>& data
) {
    Statistics result{};

    if (data.empty()) {
        return result;
    }

    long double sum = 0.0L;
    long double sumSquares = 0.0L;

    result.minimum =
        std::numeric_limits<double>::max();

    result.maximum =
        std::numeric_limits<double>::lowest();

    for (double value : data) {
        const long double longValue =
            static_cast<long double>(value);

        sum += longValue;
        sumSquares += longValue * longValue;

        if (value < result.minimum) {
            result.minimum = value;
        }

        if (value > result.maximum) {
            result.maximum = value;
        }
    }

    const long double count =
        static_cast<long double>(data.size());

    const long double mean =
        sum / count;

    long double variance =
        (sumSquares / count) -
        (mean * mean);

    // Prevent negative variance caused by floating-point rounding.
    if (variance < 0.0L) {
        variance = 0.0L;
    }

    result.mean =
        static_cast<double>(mean);

    result.variance =
        static_cast<double>(variance);

    result.standardDeviation =
        std::sqrt(result.variance);

    return result;
}

// ============================================================
// Task 2: Histogram
// ============================================================

std::vector<long long> calculate_histogram(
    const std::vector<double>& data,
    int numberOfBins,
    double minimum,
    double maximum
) {
    if (numberOfBins <= 0) {
        return {};
    }

    std::vector<long long> histogram(
        static_cast<std::size_t>(numberOfBins),
        0
    );

    if (
        data.empty() ||
        maximum <= minimum
    ) {
        return histogram;
    }

    const double binWidth =
        (maximum - minimum) /
        static_cast<double>(numberOfBins);

    for (double value : data) {
        int binIndex =
            static_cast<int>(
                (value - minimum) / binWidth
            );

        // The maximum value may produce index = numberOfBins.
        if (binIndex >= numberOfBins) {
            binIndex = numberOfBins - 1;
        }

        if (binIndex < 0) {
            binIndex = 0;
        }

        histogram[
            static_cast<std::size_t>(binIndex)
        ]++;
    }

    return histogram;
}

// ============================================================
// Task 3: Pearson correlation
// ============================================================

double calculate_pearson_correlation(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    if (
        x.empty() ||
        x.size() != y.size()
    ) {
        return 0.0;
    }

    long double sumX = 0.0L;
    long double sumY = 0.0L;
    long double sumXSquare = 0.0L;
    long double sumYSquare = 0.0L;
    long double sumXY = 0.0L;

    for (std::size_t i = 0; i < x.size(); ++i) {
        const long double xValue =
            static_cast<long double>(x[i]);

        const long double yValue =
            static_cast<long double>(y[i]);

        sumX += xValue;
        sumY += yValue;

        sumXSquare += xValue * xValue;
        sumYSquare += yValue * yValue;
        sumXY += xValue * yValue;
    }

    const long double n =
        static_cast<long double>(x.size());

    const long double numerator =
        (n * sumXY) -
        (sumX * sumY);

    const long double denominatorX =
        (n * sumXSquare) -
        (sumX * sumX);

    const long double denominatorY =
        (n * sumYSquare) -
        (sumY * sumY);

    if (
        denominatorX <= 0.0L ||
        denominatorY <= 0.0L
    ) {
        return 0.0;
    }

    const long double denominator =
        std::sqrt(
            denominatorX * denominatorY
        );

    if (denominator == 0.0L) {
        return 0.0;
    }

    return static_cast<double>(
        numerator / denominator
    );
}

// ============================================================
// Task 4: Moving average
//
// This implementation does not store every moving average.
// It stores only the first five and final value to save memory,
// especially for the 100-million-record dataset.
// ============================================================

MovingAverageResult calculate_moving_average(
    const std::vector<double>& data,
    std::size_t windowSize
) {
    MovingAverageResult result{};
    result.count = 0;
    result.lastValue = 0.0;

    if (
        windowSize == 0 ||
        data.size() < windowSize
    ) {
        return result;
    }

    long double windowSum = 0.0L;

    for (
        std::size_t i = 0;
        i < windowSize;
        ++i
    ) {
        windowSum +=
            static_cast<long double>(data[i]);
    }

    double currentAverage =
        static_cast<double>(
            windowSum /
            static_cast<long double>(windowSize)
        );

    result.count = 1;
    result.lastValue = currentAverage;
    result.firstValues.push_back(currentAverage);

    for (
        std::size_t i = windowSize;
        i < data.size();
        ++i
    ) {
        windowSum -=
            static_cast<long double>(
                data[i - windowSize]
            );

        windowSum +=
            static_cast<long double>(data[i]);

        currentAverage =
            static_cast<double>(
                windowSum /
                static_cast<long double>(windowSize)
            );

        result.count++;
        result.lastValue = currentAverage;

        if (result.firstValues.size() < 5) {
            result.firstValues.push_back(
                currentAverage
            );
        }
    }

    return result;
}

// ============================================================
// Task 5: Z-score outlier detection
// A value is considered an outlier when |z| > threshold.
// ============================================================

OutlierResult detect_zscore_outliers(
    const std::vector<double>& data,
    double mean,
    double standardDeviation,
    double threshold
) {
    OutlierResult result{};
    result.count = 0;

    if (
        data.empty() ||
        standardDeviation <= 0.0 ||
        threshold <= 0.0
    ) {
        return result;
    }

    const std::size_t maximumSamples = 5;

    for (
        std::size_t i = 0;
        i < data.size();
        ++i
    ) {
        const double zScore =
            (data[i] - mean) /
            standardDeviation;

        if (std::fabs(zScore) > threshold) {
            result.count++;

            if (
                result.sampleIndexes.size() <
                maximumSamples
            ) {
                result.sampleIndexes.push_back(i);
            }
        }
    }

    return result;
}

// ============================================================
// Save timing results to CSV
// ============================================================

bool save_timing_results(
    const std::string& outputFilename,
    const std::string& datasetFilename,
    long long recordCount,
    const std::vector<TimingRecord>& records
) {
    bool writeHeader = true;

    {
        std::ifstream existingFile(
            outputFilename,
            std::ios::binary |
            std::ios::ate
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
            << "sequential,"
            << 1 << ","
            << "\"" << datasetFilename << "\","
            << recordCount << ","
            << record.task << ","
            << record.timeMs << ","
            << record.resultValue
            << '\n';
    }

    return true;
}

// ============================================================
// Main program
// ============================================================

int main(int argc, char* argv[]) {
    // Default dataset if no argument is provided.
    std::string filename =
        "data/dataset_1M.bin";

    if (argc >= 2) {
        filename = argv[1];
    }

    const std::string csvFilename =
        "results/sequential_results.csv";

    std::vector<TimingRecord> timingRecords;

    std::cout
        << "=====================================\n"
        << " Sequential Analytics Program\n"
        << "=====================================\n"
        << "Dataset file: "
        << filename
        << "\n\n";

    // ========================================================
    // Load dataset
    // ========================================================

    const auto loadStart =
        std::chrono::high_resolution_clock::now();

    Dataset dataset =
        load_dataset(filename);

    const auto loadEnd =
        std::chrono::high_resolution_clock::now();

    if (dataset.N <= 0) {
        std::cerr
            << "ERROR: Dataset could not be loaded.\n";

        return EXIT_FAILURE;
    }

    if (
        dataset.col1.size() !=
            static_cast<std::size_t>(dataset.N) ||
        dataset.col2.size() !=
            static_cast<std::size_t>(dataset.N)
    ) {
        std::cerr
            << "ERROR: Dataset column sizes are invalid.\n";

        return EXIT_FAILURE;
    }

    const double loadTimeMs =
        std::chrono::duration<double, std::milli>(
            loadEnd - loadStart
        ).count();

    timingRecords.push_back({
        "data_loading",
        loadTimeMs,
        static_cast<double>(dataset.N)
    });

    std::cout
        << "Records loaded : "
        << dataset.N
        << '\n'
        << "Loading time   : "
        << loadTimeMs
        << " ms\n\n";

    // ========================================================
    // Preview first three records
    // ========================================================

    std::cout
        << "First three records\n"
        << "-------------------\n";

    const std::size_t previewCount =
        std::min<std::size_t>(
            3,
            dataset.col1.size()
        );

    for (
        std::size_t i = 0;
        i < previewCount;
        ++i
    ) {
        std::cout
            << "Record "
            << i
            << " | col1 = "
            << dataset.col1[i]
            << " | col2 = "
            << dataset.col2[i]
            << '\n';
    }

    std::cout << '\n';

    std::cout
        << std::fixed
        << std::setprecision(6);

    // ========================================================
    // Task 1: Basic statistics
    // ========================================================

    const auto statisticsStart =
        std::chrono::high_resolution_clock::now();

    const Statistics statistics =
        calculate_statistics(
            dataset.col1
        );

    const auto statisticsEnd =
        std::chrono::high_resolution_clock::now();

    const double statisticsTimeMs =
        std::chrono::duration<double, std::milli>(
            statisticsEnd - statisticsStart
        ).count();

    timingRecords.push_back({
        "basic_statistics",
        statisticsTimeMs,
        statistics.mean
    });

    std::cout
        << "Basic Statistics for col1\n"
        << "-------------------------\n"
        << "Mean               : "
        << statistics.mean << '\n'
        << "Variance           : "
        << statistics.variance << '\n'
        << "Standard deviation : "
        << statistics.standardDeviation << '\n'
        << "Minimum            : "
        << statistics.minimum << '\n'
        << "Maximum            : "
        << statistics.maximum << '\n'
        << "Execution time     : "
        << statisticsTimeMs
        << " ms\n\n";

    // ========================================================
    // Task 2: Histogram
    // ========================================================

    const int numberOfBins = 10;

    const auto histogramStart =
        std::chrono::high_resolution_clock::now();

    const std::vector<long long> histogram =
        calculate_histogram(
            dataset.col1,
            numberOfBins,
            statistics.minimum,
            statistics.maximum
        );

    const auto histogramEnd =
        std::chrono::high_resolution_clock::now();

    const double histogramTimeMs =
        std::chrono::duration<double, std::milli>(
            histogramEnd - histogramStart
        ).count();

    const double binWidth =
        (statistics.maximum - statistics.minimum) /
        static_cast<double>(numberOfBins);

    long long histogramTotal = 0;

    std::cout
        << "Histogram for col1\n"
        << "------------------\n";

    for (
        int i = 0;
        i < numberOfBins;
        ++i
    ) {
        const double lowerBound =
            statistics.minimum +
            (
                static_cast<double>(i) *
                binWidth
            );

        const double upperBound =
            lowerBound + binWidth;

        const long long binCount =
            histogram[
                static_cast<std::size_t>(i)
            ];

        std::cout
            << "Bin "
            << (i + 1)
            << " ["
            << lowerBound
            << ", "
            << upperBound;

        if (i == numberOfBins - 1) {
            std::cout << "]";
        }
        else {
            std::cout << ")";
        }

        std::cout
            << " : "
            << binCount
            << '\n';

        histogramTotal += binCount;
    }

    timingRecords.push_back({
        "histogram",
        histogramTimeMs,
        static_cast<double>(histogramTotal)
    });

    std::cout
        << "Histogram total    : "
        << histogramTotal
        << '\n'
        << "Expected total     : "
        << dataset.N
        << '\n'
        << "Execution time     : "
        << histogramTimeMs
        << " ms\n\n";

    if (histogramTotal != dataset.N) {
        std::cerr
            << "WARNING: Histogram total does not "
            << "match the dataset size.\n\n";
    }

    // ========================================================
    // Task 3: Pearson correlation
    // ========================================================

    const auto correlationStart =
        std::chrono::high_resolution_clock::now();

    const double correlation =
        calculate_pearson_correlation(
            dataset.col1,
            dataset.col2
        );

    const auto correlationEnd =
        std::chrono::high_resolution_clock::now();

    const double correlationTimeMs =
        std::chrono::duration<double, std::milli>(
            correlationEnd - correlationStart
        ).count();

    timingRecords.push_back({
        "pearson_correlation",
        correlationTimeMs,
        correlation
    });

    std::cout
        << "Pearson Correlation\n"
        << "-------------------\n"
        << "Correlation        : "
        << correlation
        << '\n'
        << "Execution time     : "
        << correlationTimeMs
        << " ms\n\n";

    // ========================================================
    // Task 4: Moving average
    // ========================================================

    const std::size_t movingAverageWindow = 5;

    const auto movingAverageStart =
        std::chrono::high_resolution_clock::now();

    const MovingAverageResult movingAverage =
        calculate_moving_average(
            dataset.col1,
            movingAverageWindow
        );

    const auto movingAverageEnd =
        std::chrono::high_resolution_clock::now();

    const double movingAverageTimeMs =
        std::chrono::duration<double, std::milli>(
            movingAverageEnd -
            movingAverageStart
        ).count();

    timingRecords.push_back({
        "moving_average",
        movingAverageTimeMs,
        static_cast<double>(movingAverage.count)
    });

    std::cout
        << "Moving Average\n"
        << "--------------\n"
        << "Window size        : "
        << movingAverageWindow
        << '\n'
        << "Output count       : "
        << movingAverage.count
        << '\n'
        << "First averages     : ";

    for (
        std::size_t i = 0;
        i < movingAverage.firstValues.size();
        ++i
    ) {
        std::cout
            << movingAverage.firstValues[i];

        if (
            i + 1 <
            movingAverage.firstValues.size()
        ) {
            std::cout << ", ";
        }
    }

    std::cout
        << '\n'
        << "Last average       : "
        << movingAverage.lastValue
        << '\n'
        << "Execution time     : "
        << movingAverageTimeMs
        << " ms\n\n";

    // ========================================================
    // Task 5: Z-score outlier detection
    // ========================================================

    const double zScoreThreshold = 3.0;

    const auto outlierStart =
        std::chrono::high_resolution_clock::now();

    const OutlierResult outlierResult =
        detect_zscore_outliers(
            dataset.col1,
            statistics.mean,
            statistics.standardDeviation,
            zScoreThreshold
        );

    const auto outlierEnd =
        std::chrono::high_resolution_clock::now();

    const double outlierTimeMs =
        std::chrono::duration<double, std::milli>(
            outlierEnd - outlierStart
        ).count();

    timingRecords.push_back({
        "zscore_outlier_detection",
        outlierTimeMs,
        static_cast<double>(outlierResult.count)
    });

    std::cout
        << "Z-score Outlier Detection\n"
        << "-------------------------\n"
        << "Threshold          : "
        << zScoreThreshold
        << '\n'
        << "Outlier count      : "
        << outlierResult.count
        << '\n';

    if (!outlierResult.sampleIndexes.empty()) {
        std::cout
            << "Sample outliers    :\n";

        for (
            std::size_t index :
            outlierResult.sampleIndexes
        ) {
            const double zScore =
                (
                    dataset.col1[index] -
                    statistics.mean
                ) /
                statistics.standardDeviation;

            std::cout
                << "  Index "
                << index
                << " | Value = "
                << dataset.col1[index]
                << " | Z-score = "
                << zScore
                << '\n';
        }
    }
    else {
        std::cout
            << "Sample outliers    : "
            << "None detected\n";
    }

    std::cout
        << "Execution time     : "
        << outlierTimeMs
        << " ms\n\n";

    // ========================================================
    // Task 6: Sorting
    //
    // Sorting is performed last because it changes col1 order.
    // This avoids making another full copy of the dataset.
    // ========================================================

    const auto sortingStart =
        std::chrono::high_resolution_clock::now();

    std::sort(
        dataset.col1.begin(),
        dataset.col1.end()
    );

    const auto sortingEnd =
        std::chrono::high_resolution_clock::now();

    const double sortingTimeMs =
        std::chrono::duration<double, std::milli>(
            sortingEnd - sortingStart
        ).count();

    const bool sortedCorrectly =
        std::is_sorted(
            dataset.col1.begin(),
            dataset.col1.end()
        );

    timingRecords.push_back({
        "sorting",
        sortingTimeMs,
        sortedCorrectly ? 1.0 : 0.0
    });

    std::cout
        << "Sorting\n"
        << "-------\n"
        << "Sorted correctly   : "
        << (
            sortedCorrectly
            ? "Yes"
            : "No"
        )
        << '\n'
        << "First values       : ";

    const std::size_t sortedPreviewCount =
        std::min<std::size_t>(
            3,
            dataset.col1.size()
        );

    for (
        std::size_t i = 0;
        i < sortedPreviewCount;
        ++i
    ) {
        std::cout
            << dataset.col1[i];

        if (i + 1 < sortedPreviewCount) {
            std::cout << ", ";
        }
    }

    std::cout
        << '\n'
        << "Last values        : ";

    for (
        std::size_t i = 0;
        i < sortedPreviewCount;
        ++i
    ) {
        const std::size_t index =
            dataset.col1.size() -
            sortedPreviewCount +
            i;

        std::cout
            << dataset.col1[index];

        if (i + 1 < sortedPreviewCount) {
            std::cout << ", ";
        }
    }

    std::cout
        << '\n'
        << "Execution time     : "
        << sortingTimeMs
        << " ms\n\n";

    // ========================================================
    // Total task time
    // ========================================================

    const double totalTaskTimeMs =
        statisticsTimeMs +
        histogramTimeMs +
        correlationTimeMs +
        movingAverageTimeMs +
        outlierTimeMs +
        sortingTimeMs;

    const double totalWithLoadingMs =
        loadTimeMs +
        totalTaskTimeMs;

    timingRecords.push_back({
        "total_six_tasks",
        totalTaskTimeMs,
        0.0
    });

    timingRecords.push_back({
        "total_with_loading",
        totalWithLoadingMs,
        0.0
    });

    std::cout
        << "Timing Summary\n"
        << "--------------\n"
        << "Six-task total     : "
        << totalTaskTimeMs
        << " ms\n"
        << "Including loading  : "
        << totalWithLoadingMs
        << " ms\n\n";

    // ========================================================
    // Save CSV
    // ========================================================

    const bool csvSaved =
        save_timing_results(
            csvFilename,
            filename,
            dataset.N,
            timingRecords
        );

    if (csvSaved) {
        std::cout
            << "Results saved to   : "
            << csvFilename
            << '\n';
    }

    std::cout
        << "\nSequential processing completed "
        << "successfully.\n";

    return EXIT_SUCCESS;
}

