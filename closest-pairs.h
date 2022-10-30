//
//  closest-points.h
//
//  Created by Wayne Cochran on 10/25/22.
//  Copyright © 2022 Quintar. All rights reserved.
//

#ifndef CLOSEST_PAIRS_H
#define CLOSEST_PAIRS_H

#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>

//
// Header only library for computing the closest K pairs of points.
//
// Functions for computing the closest K pairs from a single
// set of N points; the indices for the pairs are returned via
// the pairIndices array along with the assiacated squared distances
// stored in increasing magnitude:
//
//    closestPairsBruteForce(points, K, pairIndices, distances);
//       Performs exhaustive search over all N*(N-1)/2 pairs of points
//       and thus has O(N^2) time complexity.
//
//    closestPairs(points, K, pairIndices, distances);
//       Uses a divide and conquer approach following
//       https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/closepoints.pdf
//       If K is O(N), then closestPairs() runs in O(N*log(N)) time.
//
// Functions for computing the closest pairs between two sets of points of size
// N and M respectively:
//
//     closestPairsBruteForce(pointsA, pointsB, K, pairIndices, distances);
//       Performs exhaustive search over all N*M pairs of points
//       and thus has O(N*M) time complexity.
//
//     closestPairs(pointsA, pointsB, K, pairIndices, distances);
//       Uses similar divide and conquer approach as above.
//       If K is O(max(N,M)), then this runs in O(max(N,M)*log(max(N,M)).
//
//
// Template parameter P represents a point type of dimension 2
// or higher. Typical types are Eigen::Vector2f or Eigen::Vector3f.
// P must support
//    P.x()  returns x-coodinate
//    P.y()  returns y-coordinate
//    P.squaredNorm() returns magnitude of number
//    A - B : returns difference of two points
//    F=typename P::Scalar : scalar type of P (float or double)
//

//
// Examine all N*(N-1)/2 pairs of points and record the indices of the
// closest K pairs. Use when N is small or K >> N.
//
template <typename P, typename F=typename P::Scalar>
void closestPairsBruteForce(const std::vector<P>& points,
                            size_t K,
                            std::vector<std::pair<size_t,size_t>>& closestPairIndices,
                            std::vector<F>& squaredDistances) {
    const size_t N = points.size();
    if (N <= 2) return; // nada
    assert(K >= 1);
    std::vector<std::pair<size_t,size_t>> pairs;
    std::vector<F> distances;
    for (size_t i = 0; i < N-1; i++)
        for (size_t j = i+1; j < N; j++) {
            const auto d2 = (points[i] - points[j]).squaredNorm();
            pairs.emplace_back(std::make_pair(i,j));
            distances.push_back(d2);
        }
    std::vector<size_t> indices(pairs.size());
    std::iota(indices.begin(), indices.end(), 0);
    const size_t M = std::min(indices.size(),K);
    std::partial_sort(indices.begin(), indices.begin() + M, indices.end(),
                      [&](size_t i, size_t j) -> bool {
                          return distances[i] < distances[j];
                      });
    indices.resize(M);
    for (size_t i : indices) {
        closestPairIndices.emplace_back(pairs[i]);
        squaredDistances.push_back(distances[i]);
    }
}

namespace ClosestPairsAux { // helper functions

//
// Return median x-values of points which is assumed to
// be sorted on x.
//
template <typename P, typename F=typename P::Scalar>
F xmedian(const std::vector<P>& points) {
    const size_t N = points.size();
    assert(N >= 1);
    const size_t m = N/2;
    return (N % 2 == 1) ? points[m].x() : (points[m-1].x() + points[m].x())/2;
}

//
// Find median x-value between combined set A and B.
// Precondition is that A and B are sorted on x.
//
template <typename P, typename F=typename P::Scalar>
F xmedian(const std::vector<P>& A, const std::vector<P>& B) {
    const size_t N = A.size();
    const size_t M = B.size();
    assert(N + M >= 2);
    std::vector<F> xvals(N + M);
    size_t i = 0, j = 0, n = 0;
    while (i < N && j < M) {
        if (A[i].x() < B[j].x())
            xvals[n++] = A[i++].x();
        else
            xvals[n++] = B[j++].x();
    }
    while (i < N)
        xvals[n++] = A[i++].x();
    while (j < M)
        xvals[n++] = B[j++].x();
    assert(n == xvals.size());
    const size_t m = n/2;
    return (n % 2 == 1) ? xvals[m] : (xvals[m-1] + xvals[m])/2;
}

//
// Split A (assumed to be sorted on x) into two pieces divided
// by their x-value relative to xmid.
// Any point with x == xmid will be placed in Aleft.
//
template <typename P>
void splitx(float xmid, const std::vector<P>& A,
            std::vector<P>& Aleft, std::vector<P>& Aright) {
    auto iter = std::lower_bound(A.begin(), A.end(), xmid,
                                 [](const P& p, float val) -> bool {
                                     return p.x() <= val;
                                 });
    std::copy(A.begin(), iter, std::back_inserter(Aleft));
    std::copy(iter, A.end(), std::back_inserter(Aright));
}

//
// Function that determines if points p and q straddle the x = xmid
// dividing line. Points with x-values == xmid are considered to
// to be on the left side (consistent with splitx function above).
//
template <typename P, typename F=typename P::Scalar>
inline bool straddleMiddle(F xmid, const P& p, const P& q) {
    return (p.x() <= xmid && q.x() > xmid) || (q.x() <= xmid && p.x() > xmid);
};

//
// Function to merge two index and distances arrays (sorted on distance)
// into a single index and distances array (also sorted) truncated to length K.
//
template <typename F>
void merge(size_t K,
           const std::vector<std::pair<size_t,size_t>>& indicesA,
           const std::vector<F>& distancesA,
           const std::vector<std::pair<size_t,size_t>>& indicesB,
           const std::vector<F>& distancesB,
           std::vector<std::pair<size_t,size_t>>& indices,
           std::vector<F>& distances) {
    size_t i = 0, j = 0;
    while (i < indicesA.size() && j < indicesB.size()) {
        if (distancesA[i] < distancesB[j]) {
            distances.push_back(distancesA[i]);
            indices.emplace_back(indicesA[i]);
            ++i;
        } else {
            distances.push_back(distancesB[j]);
            indices.emplace_back(indicesB[j]);
            ++j;
        }
        if (distances.size() >= K)
            break;
    }
    while (distances.size() < K && i < indicesA.size()) {
        distances.push_back(distancesA[i]);
        indices.emplace_back(indicesA[i]);
        ++i;
    }
    while (distances.size() < K && j < indicesB.size()) {
        distances.push_back(distancesB[j]);
        indices.emplace_back(indicesB[j]);
        ++j;
    }
}

} // end of ClosestPairsAux namespace


//
// Find K closest points via recursive divide and conquer.
// Precondition is that points are sorted on x.
// The run time for N nodes is
//    T(N) = 2T(N/2) + O(K) + O(N)
// which is O(N*log(N)) if K is O(N).
// The total number of pairs is N*(N-1)/2 which is O(N^2).
// If K is O(N^2) then T(N) is O(N^2*log(N)) and you are
// better of just using exhaustive brute force (see function
// closestPairsBruteForce() above).
//
template <typename P, typename F=typename P::Scalar>
void closestPairsXSorted(const std::vector<P>& points,
                         size_t K,
                         std::vector<std::pair<size_t,size_t>>& closestPairIndices,
                         std::vector<F>& squaredDistances) {
    using namespace ClosestPairsAux;

    const size_t N = points.size();
    if (N <= 0) return; // nada
    assert(K >= 1);

    //
    // Base case.
    //
    const size_t Khi = N * size_t(std::ceil(std::log2f(N)));
    if (N <= 10 || K >= Khi) {
        const size_t totalPairs = N*(N-1)/2;
        closestPairsBruteForce(points, std::min(K,totalPairs),
                               closestPairIndices,
                               squaredDistances);
        return;
    }

    //
    // Split points on x midpoint.
    // Here is we assume the points are sorted on x.
    // Points with x-values == xmid will be in left half.
    //
    const auto xmid = xmedian(points);
    std::vector<P> leftPoints, rightPoints;
    splitx(xmid, points, leftPoints, rightPoints);

    //
    // Recurse on left points : find closest pairs for x <= midpoint
    //
    std::vector<std::pair<size_t,size_t>> leftPairIndices;
    std::vector<F> leftSquaredDistances;
    assert(!leftPoints.empty());
    if (rightPoints.empty()) { // degenerate case where all points on x == xmid line
        // We'll just punt and use brute force for now hoping this is a rare case.
        // We could handle this quite efficiently, but not going to bother.
        closestPairsBruteForce(points, K, closestPairIndices, squaredDistances);
        return;
    }
    closestPairsXSorted(leftPoints, K, leftPairIndices, leftSquaredDistances);

    //
    // Recurse on right points : find closest pairs for x > midpoint
    //
    std::vector<std::pair<size_t,size_t>> rightPairIndices;
    std::vector<F> rightSquaredDistances;
    closestPairsXSorted(rightPoints, K, rightPairIndices, rightSquaredDistances);
    const size_t Nleft = leftPoints.size();
    for (auto& indices : rightPairIndices) {
        indices.first += Nleft;  // adjust right indices
        indices.second += Nleft;
    }

    //
    // Merge closest points pairs from left and right pieces.
    //
    std::vector<F> distances;
    std::vector<std::pair<size_t,size_t>> pairIndices;
    merge(K,
          leftPairIndices, leftSquaredDistances,
          rightPairIndices, rightSquaredDistances,
          pairIndices, distances);

    //
    // d = largest distance of closest pair from each half;
    //
    const float d2 = distances.back();
    const float d = std::sqrt(d2);

    //
    // Harvest all the points in the middle strip, i.e. points
    // with x values within d units of the midpoint line and sort on y.
    //
    std::vector<size_t> stripIndices;
    for (size_t i = 0; i < N; i++)
        if (std::fabs(points[i].x() - xmid) <= d)
            stripIndices.push_back(i);
    std::sort(stripIndices.begin(), stripIndices.end(),
              [&](size_t i, size_t j) -> bool {
                  return points[i].y() < points[j].y();
              });

    //
    // Find the distances between pairs in the "center-x strip"
    // that straddle the middle.
    // We store the pairs in a multimap keyed on distance squared
    // so we can maintain their order sorted on distance.
    //
    std::multimap<F,std::pair<size_t,size_t>> stripPairs;
    for (size_t i = 0; i < stripIndices.size(); i++) {
        const auto& A = points[stripIndices[i]];
        for (size_t j = i+1; j < stripIndices.size(); j++) {
            const auto& B = points[stripIndices[j]];
            if (B.y() - A.y() > d) break;   // key to be linear time!
            if (straddleMiddle(xmid, A, B)) {
                const auto dist2 = (B - A).squaredNorm();
                if (dist2 < d2)
                    stripPairs.insert({dist2,std::make_pair(stripIndices[i], stripIndices[j])});
            }
        }
    }

    //
    // Walk ordered multimap in order of distances and
    // harvest all the pairs in the strip.
    //
    std::vector<F> stripDistances;
    std::vector<std::pair<size_t,size_t>> stripPairIndices;
    for (auto&& stripPair : stripPairs) {
        stripDistances.push_back(stripPair.first);
        stripPairIndices.emplace_back(stripPair.second);
    }

    //
    // Merge left/right results with strip results.
    //
    merge(K,
          pairIndices, distances,
          stripPairIndices, stripDistances,
          closestPairIndices, squaredDistances);
}

//
// Find K closest points via recursive divide and conquer.
//
template <typename P, typename F=typename P::Scalar>
void closestPairs(const std::vector<P>& points,
                  size_t K,
                  std::vector<std::pair<size_t,size_t>>& closestPairIndices,
                  std::vector<F>& squaredDistances) {
    const size_t N = points.size();
    if (N < 2) return; // no pairs
    assert(K >= 1);

    //
    // Create index list that orders points sorted on x.
    //
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t A, size_t B) -> bool {
                  return points[A].x() < points[B].x();
              });

    //
    // Create array of points sorted on x and call
    // closestPairsXSorted() to obtain to closest pairs.
    //
    std::vector<P> sortedXPoints(N);
    for (size_t i = 0; i < N; i++)
        sortedXPoints[i] = points[indices[i]];
    closestPairsXSorted(sortedXPoints, K, closestPairIndices, squaredDistances);

    //
    // Map the returned index pairs back to the original indices
    // of the input array of points.
    //
    const size_t M = closestPairIndices.size();  // may be less than K
    for (size_t n = 0; n < M; n++) {
        auto& p = closestPairIndices[n];
        const size_t i = p.first;
        const size_t j = p.second;
        p.first = indices[i];  // points[indices[i]] <-- sortedXPoints[i]
        p.second = indices[j];
    }
}

//
// Examine all N*M pairs of points between sets A and B and record the
// indices of the K closest pairs. Use when min(N,M) is small
// or K >> min(N,M).
//
template <typename P, typename F=typename P::Scalar>
void closestPairsBruteForce(const std::vector<P>& A,
                            const std::vector<P>& B,
                            size_t K,
                            std::vector<std::pair<size_t,size_t>>& closestPairIndices,
                            std::vector<F>& squaredDistances) {
    const size_t N = A.size();
    const size_t M = B.size();
    if (N <= 0 || M <= 0) return;  // nada
    assert(K >= 1);
    std::vector<std::pair<size_t,size_t>> pairs;
    std::vector<F> distances;
    for (size_t i = 0; i < N; i++)
        for (size_t j = 0; j < M; j++) {
            const auto d2 = (A[i] - B[j]).squaredNorm();
            pairs.emplace_back(std::make_pair(i,j));
            distances.push_back(d2);
        }
    std::vector<size_t> indices(pairs.size());
    std::iota(indices.begin(), indices.end(), 0);
    const size_t L = std::min(indices.size(),K);
    std::partial_sort(indices.begin(), indices.begin() + L, indices.end(),
                      [&](size_t i, size_t j) -> bool {
                          return distances[i] < distances[j];
                      });
    indices.resize(L);
    for (size_t i : indices) {
        closestPairIndices.emplace_back(pairs[i]);
        squaredDistances.push_back(distances[i]);
    }
}

//
// Find K closest points between points in sets A and B
// via recursive divide and conquer.
// Precondition is that points in A and B are sorted on x.
//
template <typename P, typename F=typename P::Scalar>
void closestPairsXSorted(const std::vector<P>& A,
                         const std::vector<P>& B,
                         size_t K,
                         std::vector<std::pair<size_t,size_t>>& closestPairIndices,
                         std::vector<F>& squaredDistances) {
    using namespace ClosestPairsAux;

    const size_t N = A.size();
    const size_t M = B.size();
    if (N <= 0 || M <= 0) return; // nada
    assert(K >= 1);

    //
    // Base case.
    //
    const size_t NM = N * M;
    const size_t Khi = std::max(N,M) * size_t(std::ceil(std::log2f(std::min(N,M))));
    if (NM <= 1000 || K >= Khi) {
        closestPairsBruteForce(A, B, std::min(K,NM),
                               closestPairIndices,
                               squaredDistances);
        return;
    }

    //
    // Split A and B based on the median x-value.
    // Points with x-values == xmid will be in left half.
    //
    const auto xmid = xmedian(A, B);
    std::vector<P> Aleft, Aright;
    splitx(xmid, A, Aleft, Aright);
    std::vector<P> Bleft, Bright;
    splitx(xmid, B, Bleft, Bright);

    //
    // Recurse on left points if the left point sets are not empty.
    //
    std::vector<std::pair<size_t,size_t>> leftPairIndices;
    std::vector<F> leftSquaredDistances;
    // XXXX assert(!Aleft.empty() && !Bleft.empty());
    if (Aright.empty() && Bright.empty()) {
        // degenerate case when all points are on the x == xmid line
        // We will just use brute for now (even though we can do this efficiently).
        closestPairsBruteForce(A, B, K, closestPairIndices, squaredDistances);
        return;
    }
    closestPairsXSorted(Aleft, Bleft, K, leftPairIndices, leftSquaredDistances);

    //
    // Recurse on right points if the right point sets are not empty.
    //
    std::vector<std::pair<size_t,size_t>> rightPairIndices;
    std::vector<F> rightSquaredDistances;
    if (!Aright.empty() || !Bright.empty()) {
        closestPairsXSorted(Aright, Bright, K, rightPairIndices, rightSquaredDistances);
    }
    const size_t Nleft = Aleft.size();
    const size_t Mleft = Bleft.size();
    for (auto& indices : rightPairIndices) {
        indices.first += Nleft; // adjust right indices
        indices.second += Mleft;
    }

    //
    // Merge closest points pairs from left and right pieces.
    //
    std::vector<F> distances;
    std::vector<std::pair<size_t,size_t>> pairIndices;
    merge(K,
          leftPairIndices, leftSquaredDistances,
          rightPairIndices, rightSquaredDistances,
          pairIndices, distances);

    //
    // d = largest distance of closest pair from each half;
    //
    const float d2 = distances.back();
    const float d = std::sqrt(d2);

    //
    // Harvest all the points in the middle strip, i.e. points
    // with x values within d units of the midpoint line and sort on y.
    //
    auto ySortedStripIndices = [d,xmid](const std::vector<P>& pts) -> std::vector<size_t> {
        std::vector<size_t> indices;
        for (size_t i = 0; i < pts.size(); i++)
            if (std::fabs(pts[i].x() - xmid) <= d)
                indices.push_back(i);
        std::sort(indices.begin(), indices.end(),
                  [&](size_t i, size_t j) -> bool {
                      return pts[i].y() < pts[j].y();
                  });
        return indices;
    };
    const std::vector<size_t> AStripIndices = ySortedStripIndices(A);
    const std::vector<size_t> BStripIndices = ySortedStripIndices(B);

    //
    // Find the distances between pairs in the "center-x strip"
    // that straddle the middle.
    // We store the pairs in a multimap keyed on distance squared
    // so we can maintain their order sorted on distance.
    //
    std::multimap<F,std::pair<size_t,size_t>> stripPairs;
    size_t i = 0, j = 0;
    while (i < AStripIndices.size() && j < BStripIndices.size()) {
        const auto& Amin = A[AStripIndices[i]];
        const auto& Bmin = B[BStripIndices[j]];
        if (Amin.y() < Bmin.y()) {
            for (size_t k = j; k < BStripIndices.size(); k++) {
                const auto& pB = B[BStripIndices[k]];
                if (pB.y() - Amin.y() > d) break; // short-circuit
                if (straddleMiddle(xmid, Amin, pB)) {
                    const auto dist2 = (pB - Amin).squaredNorm();
                    if (dist2 < d2) {
                        stripPairs.insert({dist2,std::make_pair(AStripIndices[i], BStripIndices[k])});
                    }
                }
            }
            ++i;
        } else {
            for (size_t k = i; k < AStripIndices.size(); k++) {
                const auto& pA = A[AStripIndices[k]];
                if (pA.y() - Bmin.y() > d) break;  // short circuit
                if (straddleMiddle(xmid, pA, Bmin)) {
                    const auto dist2 = (pA - Bmin).squaredNorm();
                    if (dist2 < d2) {
                        stripPairs.insert({dist2,std::make_pair(AStripIndices[k], BStripIndices[j])});
                    }
                }
            }
            ++j;
        }
    }

    //
    // Walk ordered multimap in order of distances and
    // harvest all the pairs in the strip.
    //
    std::vector<F> stripDistances;
    std::vector<std::pair<size_t,size_t>> stripPairIndices;
    for (auto&& stripPair : stripPairs) {
        stripDistances.push_back(stripPair.first);
        stripPairIndices.emplace_back(stripPair.second);
    }

    //
    // Merge left/right results with strip results.
    //
    merge(K,
          pairIndices, distances,
          stripPairIndices, stripDistances,
          closestPairIndices, squaredDistances);
}

//
// Find K closest points between sets A and B
// via recursive divide and conquer.
//
template <typename P, typename F=typename P::Scalar>
void closestPairs(const std::vector<P>& A,
                  const std::vector<P>& B,
                  size_t K,
                  std::vector<std::pair<size_t,size_t>>& closestPairIndices,
                  std::vector<F>& squaredDistances) {
    const size_t N = A.size();
    const size_t M = B.size();
    if (N <= 0 || M <= 0) return; // nothing to do
    assert(K >= 1);

    //
    // Create index lists that orders points sorted on x.
    //
    auto sortedXIndices = [](const std::vector<P>& pts) -> std::vector<size_t> {
        std::vector<size_t> indices(pts.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](size_t i, size_t j) -> bool {
                      return pts[i].x() < pts[j].x();
                  });
        return indices;
    };
    std::vector<size_t> Aindices = sortedXIndices(A);
    std::vector<size_t> Bindices = sortedXIndices(B);

    //
    // Create arrays of points sorted on x and call
    // closestPairsXSorted() to obtain to closest pairs.
    //
    auto permutedPoints = [](std::vector<size_t>& indices,
                            const std::vector<P>& pts) ->  std::vector<P> {
        assert(indices.size() == pts.size());
        std::vector<P> permuted(pts.size());
        for (size_t i = 0; i < pts.size(); i++)
            permuted[i] = pts[indices[i]];
        return permuted;
    };
    std::vector<P> sortedXA = permutedPoints(Aindices, A);
    std::vector<P> sortedXB = permutedPoints(Bindices, B);

    closestPairsXSorted(sortedXA, sortedXB, K,
                        closestPairIndices, squaredDistances);

    //
    // Map the returned index pairs back to the original indices
    // of the input arrays of points.
    //
    const size_t L = closestPairIndices.size();  // may be less than K
    for (size_t n = 0; n < L; n++) {
        auto& p = closestPairIndices[n];
        const size_t i = p.first;
        const size_t j = p.second;
        p.first = Aindices[i];  // points[indices[i]] <-- sortedXPoints[i]
        p.second = Bindices[j];
    }
}

#endif // CLOSEST_PAIRS_H
