//
//  closest-points.cpp
//
//  Created by Wayne Cochran on 10/25/22.
//  Copyright © 2022 Quintar. All rights reserved.
//

#include "closest-pairs.h"
#include <iostream>
#include <random>
#include <chrono>

class Timer {
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

#ifdef TEST_CLOSEST_PAIR_SINGLE_SET

int main(int argc, char *argv[]) {
    size_t N = 2000;
    size_t K = 20;

    if (argc == 3) {
        N = atoi(argv[1]);
        K = atoi(argv[1]);
    }

    if (N < 2 || K < 1) {
        std::cerr << "Don't do drugs!\n";
        exit(-1);
    }

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(-100, 100);

    std::vector<Eigen::Vector2f> points;
    for (size_t i = 0; i < N; i++)
        points.emplace_back(Eigen::Vector2f(dist(e2), dist(e2)));

    constexpr bool sortPoints = false;
    if (sortPoints) {
        std::sort(points.begin(), points.end(),
                  [](const Eigen::Vector2f& A, const Eigen::Vector2f& B) -> bool {
            return A.x() < B.x();
        });
    }

    constexpr bool printAllPoints = false;
    if (printAllPoints) {
        for (size_t i = 0; i < N; i++) {
            std::cout << i << " : (" << points[i].x() << "," << points[i].y() << "\n";
        }
    }

    std::vector<std::pair<size_t,size_t>> allPairs;
    std::vector<float> allDistances;
    for (size_t i = 0; i < N-1; i++) {
        const auto& A = points[i];
        for (size_t j = i+1; j < N; j++) {
            const auto& B = points[j];
            allDistances.push_back((B - A).squaredNorm());
            allPairs.emplace_back(std::make_pair(i,j));
        }
    }

    constexpr bool printAllPairs = false;
    if (printAllPairs) {
        std::cout << "all pairs:\n";
        for (size_t n = 0; n < allPairs.size(); n++) {
            std::cout << n << " : " << allPairs[n].first << "," << allPairs[n].second
                      <<  " : "  << std::sqrt(allDistances[n]) << "\n";
        }
    }

    std::vector<std::pair<size_t,size_t>> truthNearestPairs;
    std::vector<float> truthSquaredDistances;
    Timer timer;
    closestPairsBruteForce(points, K, truthNearestPairs, truthSquaredDistances);
    const double timeTruth = timer.elapsed();
    std::cout << "brute force time = " << timeTruth << " seconds" << std::endl;

    constexpr bool printClosestPairs = true;
    if (printClosestPairs) {
        std::cout << "truth closest pairs (via brute force):\n";
        for (int n = 0; n < truthNearestPairs.size(); n++) {
            std::cout << n << " : " << truthNearestPairs[n].first << "," << truthNearestPairs[n].second
                      <<  " : "  << std::sqrt(truthSquaredDistances[n]) << "\n";
        }
    }

    std::vector<std::pair<size_t,size_t>> nearestPairs;
    std::vector<float> squaredDistances;
    timer.reset();
    closestPairs(points, K, nearestPairs, squaredDistances);
    const double time = timer.elapsed();
    std::cout << "time = " << time << " seconds" << std::endl;

    if (printClosestPairs) {
        std::cout << "closest pairs\n";
        for (int n = 0; n < nearestPairs.size(); n++) {
            std::cout << n << " : " << nearestPairs[n].first << "," << nearestPairs[n].second
                      <<  " : "  << std::sqrt(squaredDistances[n]) << "\n";
        }
    }

    bool match = true;
    for (int n = 0; n < K; n++) {
        const size_t i = truthNearestPairs[n].first;
        const size_t j = truthNearestPairs[n].second;
        const auto swapped = std::make_pair(j,i);
        if (nearestPairs[n] != truthNearestPairs[n] && nearestPairs[n] != swapped) {
            if (truthSquaredDistances[n] != squaredDistances[n]) {
                std::cerr << "mismatch at index " << n << "\n";
                std::cerr << "    truth (" << i << "," << j << ") d = "
                << std::sqrt(truthSquaredDistances[n]) << "\n";
                std::cerr << "    fast  (" << nearestPairs[n].first << "," << nearestPairs[n].second << ") d = "
                << std::sqrt(squaredDistances[n]) << "\n";
                match = false;
            }
        }
    }

    if (match)
        std::cerr << "SUCCESS: matches ground truth!\n";
    else
        std::cerr << "FAIL: mismatches with ground truth!\n";

    return 0;
}

#else

int main(int argc, char *argv[]) {
    size_t N = 2000;
    size_t M = 1000;
    size_t K = 70;

    if (argc == 4) {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    if (N < 1 || M < 1 || K < 1) {
        std::cerr << "Don't do drugs!\n";
        exit(-1);
    }

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<float> dist(-100, 100);

    std::vector<Eigen::Vector2f> Apoints, Bpoints;
    for (size_t i = 0; i < N; i++)
        Apoints.emplace_back(Eigen::Vector2f(dist(e2), dist(e2)));
    for (size_t i = 0; i < M; i++)
        Bpoints.emplace_back(Eigen::Vector2f(dist(e2), dist(e2)));

    constexpr bool printAllPoints = false;
    if (printAllPoints) {
        std::cout << "A\n";
        for (size_t i = 0; i < N; i++) {
            std::cout << i << " : (" << Apoints[i].x() << "," << Apoints[i].y() << "\n";
        }
        std::cout << "B\n";
        for (size_t i = 0; i < M; i++) {
            std::cout << i << " : (" << Bpoints[i].x() << "," << Bpoints[i].y() << "\n";
        }
    }

    std::vector<std::pair<size_t,size_t>> allPairs;
    std::vector<float> allDistances;
    for (size_t i = 0; i < N; i++) {
        const auto& A = Apoints[i];
        for (size_t j = 0; j < M; j++) {
            const auto& B = Bpoints[j];
            allDistances.push_back((B - A).squaredNorm());
            allPairs.emplace_back(std::make_pair(i,j));
        }
    }

    constexpr bool printAllPairs = false;
    if (printAllPairs) {
        std::cout << "all pairs:\n";
        for (size_t n = 0; n < allPairs.size(); n++) {
            std::cout << allPairs[n].first << "," << allPairs[n].second
                      <<  " : "  << std::sqrt(allDistances[n]) << "\n";
        }
    }

    std::vector<std::pair<size_t,size_t>> truthNearestPairs;
    std::vector<float> truthSquaredDistances;
    Timer timer;
    closestPairsBruteForce(Apoints, Bpoints, K, truthNearestPairs, truthSquaredDistances);
    const double timeTruth = timer.elapsed();
    std::cout << "brute force time = " << timeTruth << " seconds" << std::endl;

    constexpr bool printClosestPairs = true;
    if (printClosestPairs) {
        std::cout << "truth closest pairs (via brute force):\n";
        for (int n = 0; n < truthNearestPairs.size(); n++) {
            std::cout << n << " : " << truthNearestPairs[n].first << "," << truthNearestPairs[n].second
                      <<  " : "  << std::sqrt(truthSquaredDistances[n]) << "\n";
        }
    }

    std::vector<std::pair<size_t,size_t>> nearestPairs;
    std::vector<float> squaredDistances;
    timer.reset();
    closestPairs(Apoints, Bpoints, K, nearestPairs, squaredDistances);
    const double time = timer.elapsed();
    std::cout << "time = " << time << " seconds" << std::endl;

    if (printClosestPairs) {
        std::cout << "closest pairs\n";
        for (int n = 0; n < nearestPairs.size(); n++) {
            std::cout << n << " : " << nearestPairs[n].first << "," << nearestPairs[n].second
                      <<  " : "  << std::sqrt(squaredDistances[n]) << "\n";
        }
    }

    bool match = true;
    for (int n = 0; n < K; n++) {
        if (nearestPairs[n] != truthNearestPairs[n] || truthSquaredDistances[n] != squaredDistances[n]) {
            std::cerr << "mismatch at index " << n << "\n";
            std::cerr << "    truth (" << truthNearestPairs[n].first << "," << truthNearestPairs[n].second << ") d = "
            << std::sqrt(truthSquaredDistances[n]) << "\n";
            std::cerr << "    fast  (" << nearestPairs[n].first << "," << nearestPairs[n].second << ") d = "
            << std::sqrt(squaredDistances[n]) << "\n";
            match = false;
        }
    }

    if (match)
        std::cerr << "SUCCESS: matches ground truth!\n";
    else
        std::cerr << "FAIL: mismatches with ground truth!\n";

    return 0;
}

#endif
