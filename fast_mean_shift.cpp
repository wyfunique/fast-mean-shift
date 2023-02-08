#include "fast_mean_shift.h"

/*
  Fast mean shift clustering algorithm.
  
  Mean shift clustering aims to discover *blobs* in a smooth density of
  samples. It is a centroid based algorithm, which works by updating candidates
  for centroids to be the mean of the points within a given region. These
  candidates are then filtered in a post-processing stage to eliminate
  near-duplicates to form the final set of centroids.
  Seeding is performed using a binning technique for scalability.

  This is a C++ implemenation of the sklearn MeanShift algorithm 
  (https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/cluster/_mean_shift.py),
  with Eigen3 and OpenMP acceleration,
  and Python API provided via Pybind11.
*/

std::pair<std::vector<int>, std::vector<float>> radius_neighbors(VectorRef query, MatrixRef search_space, float radius, Metric distance_metric) {
  ColVector distances = one_to_all_distances(query, search_space, distance_metric);
  std::vector<int> nbr_ids;
  std::vector<float> nbr_distances;
  for (int i = 0; i < distances.rows(); i++) {
    if (distances(i) <= radius) {
      nbr_ids.push_back(i);
      nbr_distances.push_back(distances(i));
    }
  }
  return std::pair<std::vector<int>, std::vector<float>>(nbr_ids, nbr_distances);
}

std::pair<std::vector<int>, std::vector<float>> kneighbors(VectorRef query, MatrixRef search_space, int k, Metric distance_metric) {
    if (query.rows() > 1)
      throw std::runtime_error("'kneighbors' only accepts single query, " + std::to_string(query.rows()) + " received.");
    std::vector<std::pair<float, int>> dist_and_id;
    ColVector distances = one_to_all_distances(query, search_space, distance_metric);
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>> > maxheap;
    for (int i = 0; i < distances.rows(); i++) {
      if (maxheap.size() < k) {
        maxheap.push(std::pair<float, int>(distances(i), i));
      }
      else {
        if (maxheap.top().first > distances(i)) {
          maxheap.pop();
          maxheap.push(std::pair<float, int>(distances(i), i));
        }
      }
    }
    std::vector<int> ids;
    std::vector<float> dists;
    while (!maxheap.empty()) {
      std::pair<float, int> dist_id = maxheap.top();
      maxheap.pop();
      ids.push_back(dist_id.second);
      dists.push_back(dist_id.first);
    }
    std::reverse(ids.begin(), ids.end());
    std::reverse(dists.begin(), dists.end());
    return std::pair<std::vector<int>, std::vector<float>>(ids, dists);
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> 
  batch_kneighbors(MatrixRef queries, MatrixRef search_space, int k, Metric distance_metric, int n_jobs) {
    int num_queries = queries.rows();
    std::vector<std::vector<int>> ids(num_queries, std::vector<int>());
    std::vector<std::vector<float>> dists(num_queries, std::vector<float>());

    if (n_jobs >= 1)
        omp_set_num_threads(n_jobs);

  #pragma omp parallel for  
    for (int i = 0; i < queries.rows(); i++) {
      std::pair<std::vector<int>, std::vector<float>> single_query_res = kneighbors(queries.row(i), search_space, k, distance_metric);
      ids[i] = single_query_res.first;
      dists[i] = single_query_res.second;
    }

    return std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>>(ids, dists);
}   

FastMeanShift::FastMeanShift() {this->distance_metric = Metric::DIST_L2;}
FastMeanShift::FastMeanShift(Metric distance_metric) {this->distance_metric = distance_metric;}
void FastMeanShift::setDistanceMetric(Metric distance_metric) {this->distance_metric = distance_metric;}
void FastMeanShift::turnOnDebug() {this->debug = true;}
void FastMeanShift::turnOffDebug() {
  this->debug = false;
  this->_centers_and_intensities.clear();
}

float _estimate_bandwidth(MatrixRef X, float quantile, Metric distance_metric, int n_jobs) {
  int n_neighbors = (int)(X.rows() * quantile);
  if (n_neighbors < 1)  // cannot fit batch_kneighbors with n_neighbors = 0
      n_neighbors = 1;
 
  float bandwidth = 0.0;
  std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> 
    nbrs_ids_and_distances = batch_kneighbors(X, X, n_neighbors, distance_metric, n_jobs);
  std::vector<std::vector<float>> d = nbrs_ids_and_distances.second;
  
  for (std::vector<float>& knn_distances : d) {
    // get the furtherest neighbor distance, i.e., distance of the last one in the kNN, 
    // since the neighbors returned by `batch_kneighbors` are sorted by their distances to the query ascendingly.  
    bandwidth += knn_distances.back(); 
  }
  
  // get the avearge furtherest distance between any point and its k nearest neighbors, 
  // which is the estimated value for bandwidth.
  return bandwidth / X.rows();
}

float FastMeanShift::estimate_bandwidth(MatrixRef X, float quantile, int n_samples, int random_state, int n_jobs) {
  if (n_samples > 0) {
    std::default_random_engine rng;
    rng.seed(random_state);
    std::vector<int> indices(X.rows()) ; // vector of indices from 0 to X.rows()-1.
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., X.rows()-1.
    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<int> sample_indices(indices.begin(), indices.begin() + n_samples);
    Matrix samples = X(sample_indices, Eigen::all);
    return _estimate_bandwidth(samples, quantile, this->distance_metric, n_jobs);
  }
  // Never assign `X` to `samples` or assign `samples` to `X`, becasue 
  // assignment (`=`) operation on Eigen matrix is actually deep copying from the right to left.
  // So (1) `X = samples` will modify X itself since X is passed by reference
  // and (2) `samples = X` will generate a copy of X which occupies large memory space.
  return _estimate_bandwidth(X, quantile, this->distance_metric, n_jobs);
}

// separate function for each seed's iterative loop
std::pair<Vector, std::pair<int, int>> FastMeanShift::_mean_shift_single_seed(Vector my_mean, MatrixRef X, float bandwidth, Metric distance_metric, int max_iter) {
  // For each seed, climb gradient until convergence or max_iter
  float stop_thresh = 1e-3 * bandwidth;  // when mean has converged
  int completed_iterations = 0;
  Matrix points_within(0, 0);
  while (true) {
    // Find mean of points within bandwidth
    std::pair<std::vector<int>, std::vector<float>> i_nbrs_and_dist = radius_neighbors(my_mean, X, bandwidth, distance_metric);
    std::vector<int> i_nbrs = i_nbrs_and_dist.first;
    points_within = X(i_nbrs, Eigen::all);
    if (points_within.rows() == 0)
      break;  // Depending on seeding strategy this condition may occur
    Vector my_old_mean = my_mean;  // save the old mean
    my_mean = points_within.colwise().mean();
    // If converged or at max_iter, adds the cluster
    if ((my_mean - my_old_mean).norm() < stop_thresh || completed_iterations == max_iter) 
      break;
    completed_iterations++;
  }
  return std::pair<Vector, std::pair<int, int>>(my_mean, std::pair<int, int>(points_within.rows(), completed_iterations));
}

struct compareSize {
    bool operator()(std::pair<Vector, int> &left, std::pair<Vector, int> &right) { 
        return left.second > right.second; 
    };
} descending;

Matrix FastMeanShift::_mean_shift(
    MatrixRef X,
    float bandwidth,
    Metric distance_metric,
    Matrix seeds,
    bool bin_seeding,
    int min_bin_freq,
    bool cluster_all,
    int max_iter,
    int n_jobs
) {
  if (bandwidth < 0)
      bandwidth = estimate_bandwidth(X, 0.3, 0, n_jobs);

  if (seeds.size() == 0)
    if (bin_seeding) {
      seeds = get_bin_seeds(X, bandwidth, min_bin_freq);
    }
    else {
      seeds = X;
    }
  int n_samples = X.rows();
  int n_features = X.cols();
  std::vector<std::pair<Vector, int>> center_intensity_dict;

  // execute iterations on all seeds in parallel
  std::vector<std::pair<Vector, std::pair<int, int>>> all_res(seeds.rows(), std::pair<Vector, std::pair<int, int>>(Vector(0), std::pair<int, int>(-1, -1)));
  if (n_jobs >= 1)
    omp_set_num_threads(n_jobs);

#pragma omp parallel for  
  for (int i = 0; i < seeds.rows(); i++) {
    all_res[i] = _mean_shift_single_seed(seeds.row(i), X, bandwidth, distance_metric, max_iter);
  }
   
  // copy results in a dictionary
  for (int i = 0; i < seeds.rows(); i++) {
    if (all_res[i].second.first > 0)  // i.e. len(points_within) > 0
        center_intensity_dict.push_back(std::pair<Vector, int>(all_res[i].first, all_res[i].second.first));
  }

  if (center_intensity_dict.empty())
      // nothing near seeds
      throw std::runtime_error(
          "No point was within bandwidth=" + std::to_string(bandwidth) + " of any seed. Try a different seeding strategy or increase the bandwidth."
      );

  // POST PROCESSING: remove near duplicate points
  // If the distance between two kernels is less than the bandwidth,
  // then we have to remove one because it is a duplicate. Remove the
  // one with fewer points.
  sort(center_intensity_dict.begin(), center_intensity_dict.end(), descending);  
  Matrix sorted_centers(center_intensity_dict.size(), center_intensity_dict[0].first.cols()); 
  for (int i = 0; i < center_intensity_dict.size(); i++)
    sorted_centers.row(i) = center_intensity_dict[i].first;

  if (this->debug)
    this->_centers_and_intensities = center_intensity_dict;

  IntegerVector unique = IntegerVector::Ones(sorted_centers.rows());
  for (int i = 0; i < sorted_centers.rows(); i++) {
    if (unique(i)) {
        std::pair<std::vector<int>, std::vector<float>> neighbor_idxs_distances = radius_neighbors(sorted_centers.row(i), sorted_centers, bandwidth, distance_metric);
        std::vector<int> neighbor_idxs = neighbor_idxs_distances.first;
        for (int nbr_idx : neighbor_idxs)
            unique(nbr_idx) = 0;
        unique(i) = 1;  // leave the current point as unique
    }
  }
  
  int num_clusters = 0;
  for (int i = 0; i < unique.cols(); i++)
    num_clusters += (unique(i) ? 1 : 0);

  Matrix cluster_centers(num_clusters, sorted_centers.cols()); 
  int idx_cluster = 0;
  for (int i = 0; i < unique.cols(); i++)
    if (unique(i)) {
      cluster_centers.row(idx_cluster) = sorted_centers.row(i);
      idx_cluster++;
    }
      
  return cluster_centers;
}

Matrix FastMeanShift::mean_shift_density_peaks_detecting(
  MatrixRef X,
  float bandwidth,
  Matrix seeds,
  bool bin_seeding,
  int min_bin_freq,
  bool cluster_all,
  int max_iter,
  int n_jobs
) {
    Metric distance_metric = this->distance_metric;
    return _mean_shift(
        X, bandwidth, distance_metric, seeds, bin_seeding, min_bin_freq, cluster_all, max_iter, n_jobs
    );
}

std::pair<Matrix, IntegerVector> FastMeanShift::mean_shift_clustering(
  MatrixRef X,
  float bandwidth,
  Matrix seeds,
  bool bin_seeding,
  int min_bin_freq,
  bool cluster_all,
  int max_iter,
  int n_jobs
) {
    Metric distance_metric = this->distance_metric;
    Matrix cluster_centers = _mean_shift(
        X, bandwidth, distance_metric, seeds, bin_seeding, min_bin_freq, cluster_all, max_iter, n_jobs
    );

    // ASSIGN LABELS: a point belongs to the cluster that it is closest to
    IntegerVector labels = IntegerVector::Zero(X.rows());
    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>> 
        idxs_and_distances = batch_kneighbors(X, cluster_centers, 1, distance_metric, n_jobs);
    std::vector<std::vector<int>> idxs = idxs_and_distances.first;
    std::vector<std::vector<float>> distances = idxs_and_distances.second;
    if (cluster_all) {
        int lb = 0;
        for (int i = 0; i < idxs.size(); i++)
            for (int j = 0; j < idxs[i].size(); j++) {
                labels(lb) = idxs[i][j];
                lb++;
            }
    }
    else {
        labels = labels.array() - 1; // un-clustered points will be labeled with -1.
        for (int i = 0; i < distances.size(); i++) {
            if (distances[i].size() == 0)
                throw std::runtime_error("Unknown error in mean_shift_clustering");
            if (distances[i][0] <= bandwidth) {
                labels(i) = idxs[i][0];
            }
        }
    }
    return std::pair<Matrix, IntegerVector>(cluster_centers, labels);
}

std::string integer_vector_to_string(IntegerVectorRef vec) {
  std::string res = "";
  for (int i = 0; i < vec.cols(); i++) {
    res += std::to_string(vec(i)) + ",";
  }
  return res;
}

IntegerVector string_to_integer_vector(std::string& signature) {
  if (signature == ",")
    return IntegerVector(0); // empty Vector
  int pos = 0;
  std::vector<int> vec;
  while (pos < signature.length()) {
    int next_pos = signature.find(",", pos);
    vec.push_back(std::stoi(signature.substr(pos, next_pos-pos)));
    pos = next_pos + 1;
  }
  IntegerVector res(vec.size());
  for (int i = 0; i < res.cols(); i++)
    res(i) = vec[i];
  return res;
}

/*
  Find seeds for mean_shift.
  Finds seeds by first binning data onto a grid whose lines are
  spaced bin_size apart, and then choosing those bins with at least
  min_bin_freq points.
  Parameters
  ----------
  X : array-like of shape (n_samples, n_features)
      Input points, the same points that will be used in mean_shift.
  bin_size : float
      Controls the coarseness of the binning. Smaller values lead
      to more seeding (which is computationally more expensive). If you're
      not sure how to set this, set it to the value of the bandwidth.
  min_bin_freq : int, default=1
      Only bins with at least min_bin_freq will be selected as seeds.
      Raising this value decreases the number of seeds found, which
      makes mean_shift computationally cheaper.
  Returns
  -------
  bin_seeds : array-like of shape (n_samples, n_features)
      Points used as initial kernel positions in mean shift algorithm.
*/
Matrix FastMeanShift::get_bin_seeds(MatrixRef X, float bin_size, int min_bin_freq) {    
  if (bin_size == 0)
    return X;
  
  // Bin points
  std::unordered_map<std::string, int> bin_sizes;
  for (int i = 0; i < X.rows(); i++) {
    IntegerVector binned_point = Eigen::round(X.row(i).array() / bin_size).cast<int>();
    std::string key = integer_vector_to_string(binned_point);
    if (bin_sizes.find(key) == bin_sizes.end())
      bin_sizes[key] = 0;
    bin_sizes[key] += 1;
  }
  
  // Select only those bins as seeds which have enough members
  std::vector<IntegerVector> _bin_seeds;
  for (std::pair<std::string, int> point_and_freq : bin_sizes) {
    if (point_and_freq.second >= min_bin_freq) {
      _bin_seeds.push_back(string_to_integer_vector(point_and_freq.first));
    }
  }
  if (_bin_seeds.size() == X.rows()) {
    std::cout << "[Warning] Binning data failed with provided bin_size="
            << bin_size
            << ", using data points as seeds." << std::endl;
    return X;
  }
  if (_bin_seeds.size() == 0) {
    throw std::runtime_error("Binning data found 0 proper seeds with provided bin_size=" + std::to_string(bin_size) + ", please try a larger bin_size.");
  }
  Matrix bin_seeds(_bin_seeds.size(), _bin_seeds[0].cols());
  for (int i = 0; i < _bin_seeds.size(); i++) {
    bin_seeds.row(i) = _bin_seeds[i].array().cast<VectorElemType>() * bin_size;
  }

  return bin_seeds;
}


namespace py = pybind11;
PYBIND11_MODULE(fast_mean_shift, m) {
    m.doc() = "pybind11 APIs for fast-mean-shift backend"; 
      
    py::enum_<Metric>(m, "Metric")
    .value("DIST_L2", Metric::DIST_L2)
    .value("DIST_COS", Metric::DIST_COS)
    .export_values();
  
    py::class_<FastMeanShift>(m, "FastMeanShift")
        .def( py::init<>() )
        .def( py::init<Metric>(), 
              py::arg("distance_metric") 
        )
        .def("setDistanceMetric", 
             static_cast<void (FastMeanShift::*)(Metric)> (&FastMeanShift::setDistanceMetric), 
             py::arg("distance_metric")
            )
        .def("estimate_bandwidth", 
            static_cast<float (FastMeanShift::*)(MatrixRef, float, int, int, int)> (&FastMeanShift::estimate_bandwidth),
            py::arg("X"), 
            py::arg("quantile") = 0.3,
            py::arg("n_samples") = -1,
            py::arg("random_state") = 0, 
            py::arg("n_jobs") = -1
          )
        .def("mean_shift_density_peaks_detecting", 
            static_cast<Matrix (FastMeanShift::*)(MatrixRef, float, Matrix, bool, int, bool, int, int)> (&FastMeanShift::mean_shift_density_peaks_detecting),
            py::arg("X"), 
            py::arg("bandwidth"),
            py::arg("seeds") = Matrix(0, 0), 
            py::arg("bin_seeding") = false, 
            py::arg("min_bin_freq") = 1, 
            py::arg("cluster_all") = true, 
            py::arg("max_iter") = 300, 
            py::arg("n_jobs") = -1
          )
        .def("mean_shift_clustering", 
            static_cast<std::pair<Matrix, IntegerVector> (FastMeanShift::*)(MatrixRef, float, Matrix, bool, int, bool, int, int)> (&FastMeanShift::mean_shift_clustering),
            py::arg("X"), 
            py::arg("bandwidth"),
            py::arg("seeds") = Matrix(0, 0), 
            py::arg("bin_seeding") = false, 
            py::arg("min_bin_freq") = 1, 
            py::arg("cluster_all") = true, 
            py::arg("max_iter") = 300, 
            py::arg("n_jobs") = -1
          )
        .def_readwrite("distance_metric", &FastMeanShift::distance_metric);
}
