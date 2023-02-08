#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <utility>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <numeric>
#include <bits/stdc++.h>
#include <float.h>
#include <queue>
#include <omp.h>
#include "util.h"

std::pair<std::vector<int>, std::vector<float>> 
  radius_neighbors(VectorRef query, MatrixRef search_sapce, float radius, Metric distance_metric);
std::pair<std::vector<int>, std::vector<float>> 
  kneighbors(VectorRef query, MatrixRef search_space, int k, Metric distance_metric);
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> 
  batch_kneighbors(MatrixRef queries, MatrixRef points, int k, Metric distance_metric, int n_jobs=-1);

class FastMeanShift {
public:
  Metric distance_metric;

  // whether to turn on the debug mode.
  // when debug = true, the following class attributes (i.e., some intermediate results) will be recorded, 
  // otherwise they will not be recorded. 
  bool debug; 
  // the cluster centers and the intensity (i.e., the number of points) in each of their regions before merging too close centers
  std::vector<std::pair<Vector, int>> _centers_and_intensities;   
  
  FastMeanShift();
  FastMeanShift(Metric distance_metric);
  void setDistanceMetric(Metric distance_metric);
  void turnOnDebug();
  void turnOffDebug();
  
  float estimate_bandwidth(MatrixRef X, float quantile=0.3, int n_samples=-1, int random_state=0, int n_jobs=-1);

  std::pair<Vector, std::pair<int, int>> 
    _mean_shift_single_seed(Vector my_mean, MatrixRef X, float bandwidth, Metric distance_metric, int max_iter = 300);

  Matrix get_bin_seeds(MatrixRef X, float bin_size, int min_bin_freq = 1);

  Matrix _mean_shift(
    MatrixRef X,
    float bandwidth,
    Metric distance_metric, 
    Matrix seeds = Matrix(0, 0),
    bool bin_seeding = false,
    int min_bin_freq = 1,
    bool cluster_all = true,
    int max_iter = 300,
    int n_jobs = -1
  );

  Matrix mean_shift_density_peaks_detecting(
    MatrixRef X,
    float bandwidth,
    Matrix seeds = Matrix(0, 0),
    bool bin_seeding = false,
    int min_bin_freq = 1,
    bool cluster_all = true,
    int max_iter = 300,
    int n_jobs = -1
  );

  std::pair<Matrix, IntegerVector> mean_shift_clustering(
    MatrixRef X,
    float bandwidth,
    Matrix seeds = Matrix(0, 0),
    bool bin_seeding = false,
    int min_bin_freq = 1,
    bool cluster_all = true,
    int max_iter = 300,
    int n_jobs = -1
  );

};

