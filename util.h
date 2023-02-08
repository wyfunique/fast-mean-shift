#ifndef UTIL_H
#define UTIL_H

#include <Eigen/Dense>
#include <stdexcept>
#include <vector> 
#include <iostream>
#include <string>
#include <assert.h>   
#include <sys/stat.h>
#include <unordered_map>

enum Metric {
  DIST_L2 = 0,
  DIST_COS = 1
};

typedef float VectorElemType;
// The default Matrix is defined as row matrix.
typedef Eigen::Matrix<VectorElemType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Ref<Matrix> MatrixRef; 
// The default Vector is defined as row vector.
typedef Eigen::Matrix<VectorElemType, 1, Eigen::Dynamic> Vector;
typedef Eigen::Ref<Vector> VectorRef;
typedef Eigen::Matrix<VectorElemType, Eigen::Dynamic, 1> ColVector;

typedef Eigen::Matrix<int, 1, Eigen::Dynamic> IntegerVector;
typedef Eigen::Ref<IntegerVector> IntegerVectorRef;

float L2_distance(VectorRef p1, VectorRef p2);
float cosine_distance(VectorRef p1, VectorRef p2);
ColVector L2_distances(VectorRef query, MatrixRef points);
ColVector cosine_distances(VectorRef query, MatrixRef points);

ColVector one_to_all_distances(VectorRef query, MatrixRef points, Metric distance_metric);
Matrix pairwise_distances(MatrixRef points, Metric distance_metric);

#endif
