#include "clustream.hpp"

CluStream::CluStream(int h, int m, int t) {
    time_window = h;
    this->m = m;
    this->t = t;
    timestamp = 0;
    points_fitted = 0;
    points_forgot = 0;
    points_merged = 0;
    dim = 0;
}
void CluStream::init_kernels_offline(ndarray cluster_centers, ndarray initpoints) {
    auto cluster_centers_buff = cluster_centers.request();
    auto initpoints_buff = initpoints.request();
    if (initpoints_buff.ndim != 2)
        throw std::runtime_error("initpoints must be a matrix");
    if (cluster_centers_buff.ndim != 2)
        throw std::runtime_error("cluster_centers must be a matrix");
    if (cluster_centers_buff.shape[1] != initpoints_buff.shape[1])
        throw std::runtime_error("dimension of cluster_centers is different from initpoints");
    if (cluster_centers_buff.shape[0] != m)
        throw std::runtime_error("number of cluster_centers must be equal to m");
    if (dim == 0)
        dim = initpoints_buff.shape[1];
    else if (dim != initpoints_buff.shape[1])
        throw std::runtime_error("initpoints must have a consistent number of columns");
    if (kernels.size() > 0)
        throw std::runtime_error("kernels already initialized or semi-initialized");
    unsigned int lines = initpoints_buff.shape[0];
    double* initpoints_ptr = (double*)initpoints_buff.ptr;
    double* cluster_centers_ptr = (double*)cluster_centers_buff.ptr;

    std::vector<double> zero_vector_tmp(dim, 0);
    double* zero_vector = &(zero_vector_tmp[0]);

    for (unsigned int i = 0;(int)i < m;i++) {
        //no points assigned, yet
        kernels.push_back(Kernel(zero_vector, dim, 0, t, m));
        kernels[i].n = 0;
    }

    for (unsigned int i = 0;i < lines;i++) {
        unsigned int kernel_index = 0;
        double min_distance = double_max;
        double* current_center = cluster_centers_ptr;
        for (unsigned int j = 0; (int)j < m; j++) {
            double dist = c_distance(initpoints_ptr, current_center, dim);
            if (dist < min_distance) {
                kernel_index = j;
                min_distance = dist;
            }
            current_center += dim;
        }
        //add point on timestamp==lines
        kernels[kernel_index].insert(initpoints_ptr, lines);
        initpoints_ptr += dim;
    }
    timestamp = lines + 1;
}
void CluStream::online_cluster(double* datapoint) {
    // 0. Initialize
    if ((int)kernels.size() != m) {
        //timestamp == m because it was fed at once in the algorithm
        kernels.push_back(Kernel(datapoint, dim, m, t, m));
        return;
    }


    // 1. Determine closest kernel
    Kernel* closest_kernel = NULL;
    double min_distance = double_max;
    for (unsigned int i = 0; i < kernels.size(); i++) { //O(n)
        double dist = c_distance(datapoint, &(kernels[i].center[0]), dim);
        if (dist < min_distance) {
            closest_kernel = &kernels[i];
            min_distance = dist;
        }
    }
    // 2. Check whether instance fits into closestKernel
    double radius = 0;
    if (closest_kernel->n == 1) {
        // Special case: estimate radius by determining the distance to the
        // next closest cluster
        radius = double_max;
        Point center = closest_kernel->center;
        for (unsigned int i = 0; i < kernels.size(); i++) { //O(n)
            if (&kernels[i] == closest_kernel) {
                continue;
            }

            double dist = distance(kernels[i].center, center);
            radius = std::min(dist, radius);
        }
    }
    else
        radius = closest_kernel->get_radius();

    if (min_distance < radius) {
        // Date fits, put into kernel and be happy
#ifdef DEBUG
        printf("%ld fits\n", timestamp);
#endif
        points_fitted++;
        closest_kernel->insert(datapoint, timestamp);
        return;
    }

    // 3. Date does not fit, we need to free
    // some space to insert a new kernel
    long threshold = timestamp - time_window; // Kernels before this can be forgotten

    // 3.1 Try to forget old kernels
    for (unsigned int i = 0; i < kernels.size(); i++) {
        if (kernels[i].get_relevance_stamp() < threshold) {
            kernels[i] = Kernel(datapoint, dim, timestamp, t, m);
#ifdef DEBUG
            printf("%ld forgot kernel\n", timestamp);
#endif
            points_forgot++;
            return;
        }
    }
    // 3.2 Merge closest two kernels
    int closest_a = 0;
    int closest_b = 0;
    min_distance = double_max;
    for (unsigned int i = 0; i < kernels.size(); i++) { //O(n(n+1)/2)
        Point center_a = kernels[i].center;
        for (unsigned int j = i + 1; j < kernels.size(); j++) {
            double dist = distance(center_a, kernels[j].center);
            if (dist < min_distance) {
                min_distance = dist;
                closest_a = i;
                closest_b = j;
            }
        }
    }
#ifdef DEBUG
    printf("%ld merged kernel\n", timestamp);
#endif
    points_merged++;
    kernels[closest_a].add(kernels[closest_b]);
    kernels[closest_b] = Kernel(datapoint, dim, timestamp, t, m);
}

void CluStream::batch_online_cluster(ndarray batch) {
    auto buff = batch.request();
    if (buff.ndim != 2)
        throw std::runtime_error("batch must be a matrix");
    unsigned int lines = buff.shape[0];
    if (dim == 0)
        dim = buff.shape[1];
    else if (dim != buff.shape[1])
        throw std::runtime_error("batch must have a consistent number of columns");
    double* ptr = (double*)buff.ptr;
    for (unsigned int i = 0;i < lines;i++) {
        online_cluster(ptr);
        timestamp++;
        ptr += dim;
    }
}
ndarray CluStream::get_kernel_centers() {
    ndarray ans(m * dim);
    double* ptr = (double*)ans.request().ptr;
    for (Kernel kernel : kernels) {
        std::copy_n(&(kernel.center[0]), dim, ptr);
        ptr += dim;
    }
    ans.resize({ (size_t)m,(size_t)dim });
    return ans;
}

double c_distance(double* a, double* b, unsigned int dim) {
    double ans = 0;
    for (unsigned int i = 0;i < dim;i++) {
        double diff = b[i] - a[i];
        ans += diff * diff;
    }
    return sqrt(ans);
}

double distance(Point& a, Point& b) {
    double ans = 0;
    for (unsigned int i = 0;i < a.size();i++) {
        double diff = b[i] - a[i];
        ans += diff * diff;
    }
    return sqrt(ans);
}