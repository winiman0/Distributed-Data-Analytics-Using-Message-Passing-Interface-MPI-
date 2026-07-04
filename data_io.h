// data_io.h
// Member 3 - Data Engineer
// Shared header: used by both sequential_analytics.cpp and mpi_analytics.cpp

#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

struct Dataset {
    long long N;
    std::vector<double> col1;
    std::vector<double> col2;
};

Dataset load_dataset(const std::string& filename) {
    Dataset ds;

    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "ERROR: Cannot open dataset: " << filename << "\n";
        ds.N = 0;
        return ds;
    }

    infile.read(reinterpret_cast<char*>(&ds.N), sizeof(long long));

    ds.col1.resize(ds.N);
    ds.col2.resize(ds.N);

    infile.read(reinterpret_cast<char*>(ds.col1.data()), ds.N * sizeof(double));
    infile.read(reinterpret_cast<char*>(ds.col2.data()), ds.N * sizeof(double));

    infile.close();
    std::cout << "Loaded " << ds.N << " records from " << filename << "\n";
    return ds;
}