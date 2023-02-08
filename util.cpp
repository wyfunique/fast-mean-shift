#include "util.h"

float L2_distance(VectorRef p1, VectorRef p2) {
  return (p1 - p2).norm();
}

float cosine_distance(VectorRef p1, VectorRef p2) {
  return 1.0 - p1.dot(p2) / (p1.norm() * p2.norm());
}

ColVector L2_distances(VectorRef query, MatrixRef points) {
  return (points.rowwise() - query).rowwise().norm();
}

ColVector cosine_distances(VectorRef query, MatrixRef points) {
  float query_norm = query.norm();
  return 1.0 - (points * query.transpose()).array() / (points.rowwise().norm() * query_norm).array();
}

ColVector one_to_all_distances(VectorRef query, MatrixRef points, Metric distance_metric) {
    if (distance_metric == Metric::DIST_L2) {
        return L2_distances(query, points);
    }
    else if (distance_metric == Metric::DIST_COS) {
        return cosine_distances(query, points);
    }
    else {
      throw std::runtime_error("Unsupported distance metric: " + std::to_string(distance_metric));
    }
}

Matrix pairwise_distances(MatrixRef points, Metric distance_metric) {
  Matrix res = Matrix::Zero(points.rows(), points.rows());
#pragma omp parallel for  
  for (int i = 0; i < points.rows(); i++) {
    for (int j = i+1; j < points.rows(); j++) {
      if (distance_metric == Metric::DIST_L2) {
        res(i, j) = L2_distance(points.row(i), points.row(j));
      }
      else if (distance_metric == Metric::DIST_COS) {
        res(i, j) = cosine_distance(points.row(i), points.row(j));
      }
      else {
        throw std::runtime_error("Unsupported distance metric: " + std::to_string(distance_metric));
      }
    }
  }
  return res;
}
