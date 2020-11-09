#include "kernel.hpp"

Kernel::Kernel(double* datapoint, unsigned int dim, long timestamp, double t, unsigned int m) :ls(dim), ss(dim) {
    this->t = t;
    this->m = m;
    this->dim = dim;
    n = 1;
    for (unsigned int i = 0;i < dim;i++) {
        double data = datapoint[i];
        ls[i] = data;
        ss[i] = data * data;
    }
    lst = timestamp;
    sst = timestamp * timestamp;
    center = get_center();
}
void Kernel::insert(double* datapoint, long timestamp) {
    n++;
    for (unsigned int i = 0;i < dim;i++) {
        double data = datapoint[i];
        ls[i] += data;
        ss[i] += data * data;
    }
    lst += timestamp;
    sst += timestamp * timestamp;
    center = get_center();
}
void Kernel::add(Kernel& other) {
    n += other.n;
    for (unsigned int i = 0;i < other.ls.size();i++) {
        ls[i] += other.ls[i];
        ss[i] += other.ss[i];
    }
    lst += other.lst;
    sst += other.sst;
    center = get_center();
}
double Kernel::get_relevance_stamp() {
    if (n < (2 * m))
        return get_mu_time();
    return get_mu_time() + get_sigma_time() * get_quantile(((double)m) / (2 * n));
}
double Kernel::get_mu_time() {
    return lst / n;
}
double Kernel::get_sigma_time() {
    return sqrt(sst / n - (lst / n) * (lst / n));
}
double Kernel::get_quantile(double z) {
    assert(z >= 0 && z <= 1);
    return sqrt(2) * inverse_error(2 * z - 1);
}
double Kernel::get_radius() {
    if (n == 1)
        return 0;
    return get_deviation() * RADIUS_FACTOR;
}
double Kernel::get_deviation() {
    Point variance = get_variance_vector();
    double sum_of_deviation = 0;
    for (unsigned int i = 0;i < variance.size();i++) {
        sum_of_deviation += sqrt(variance[i]);
    }
    return sum_of_deviation / variance.size();
}
Point Kernel::get_center() {
    if (n == 1)
        return ls;
    Point ans(ls.size());
    for (unsigned int i = 0;i < ls.size();i++) {
        ans[i] = ls[i] / n;
    }
    return ans;
}
double Kernel::get_inclusion_probability(double* datapoint) {
    if (n == 1) {
        double distance = 0;
        for (unsigned int i = 0;i < ls.size();i++) {
            double d = ls[i] - datapoint[i];
            distance += d * d;
        }
        distance = sqrt(distance);
        if (distance < EPSILON)
            return 1;
        return 0;
    }
    else {
        double dist = calc_normalized_distance(datapoint);
        if (dist <= get_radius())
            return 1;
        else
            return 0;
    }
}
Point Kernel::get_variance_vector() {
    Point ans(ls.size());
    for (unsigned int i = 0;i < ls.size();i++) {
        double linear_sum = ls[i];
        double squared_sum = ss[i];

        double ls_div_n = linear_sum / n;
        double ls_div_squared = ls_div_n * ls_div_n;
        double ss_div_n = squared_sum / n;

        ans[i] = ss_div_n - ls_div_squared;

        // if(ans[i]<=0.0)
        // 	ans[i]=MIN_VARIANCE;
        if (ans[i] <= 0.0) {
            if (ans[i] > -EPSILON) {
                ans[i] = MIN_VARIANCE;
            }
        }
    }
    return ans;
}
double Kernel::calc_normalized_distance(double* point) {
    // variance=get_variance_vector();
    double ans = 0;

    for (unsigned int i = 0;i < center.size();i++) {
        double diff = center[i] - point[i];
        ans += (diff * diff); // variance[i];
    }
    return sqrt(ans);
}
double Kernel::inverse_error(double x) {
    double z = sqrt(M_PI) * x;
    double res = (z) / 2;

    double z2 = z * z;
    double zProd = z * z2; // z^3
    res += (1.0 / 24) * zProd;

    zProd *= z2;  // z^5
    res += (7.0 / 960) * zProd;

    zProd *= z2;  // z^7
    res += (127 * zProd) / 80640;

    zProd *= z2;  // z^9
    res += (4369 * zProd) / 11612160;

    zProd *= z2;  // z^11
    res += (34807 * zProd) / 364953600;

    zProd *= z2;  // z^13
    res += (20036983 * zProd) / 797058662400;

    return res;
}