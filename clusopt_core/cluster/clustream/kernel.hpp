#pragma once

#include <vector>
#include <cmath>
#include <cassert>
#include <cstdio>

const double RADIUS_FACTOR = 1.8;
const double EPSILON = 0.00005;
const double MIN_VARIANCE = 1e-50;

typedef std::vector<double> Point;

class Kernel {
public:
    double t;
    unsigned int m;

    unsigned int n;
    Point ls;
    Point ss;
    double lst;
    double sst;

    unsigned int dim;

    Point center;

    Kernel(double* datapoint, unsigned int dim, long timestamp, double t, unsigned int m);
    void insert(double* datapoint, long timestamp);
    void add(Kernel& other);
    double get_relevance_stamp();
    double get_mu_time();
    double get_sigma_time();
    double get_quantile(double z);
    double get_radius();
    double get_deviation();
    Point get_center();
    double get_inclusion_probability(double* datapoint);
    Point get_variance_vector();
    double calc_normalized_distance(double* datapoint);
    double inverse_error(double x);
};