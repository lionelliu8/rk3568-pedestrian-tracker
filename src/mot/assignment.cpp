#include "assignment.h"

#include <algorithm>
#include <limits>

namespace mot {

float iou(const Box& a, const Box& b) {
    const float w = std::max(0.0f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const float h = std::max(0.0f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const float inter = w * h;
    const float uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<std::vector<float>> iou_matrix(const std::vector<Box>& a, const std::vector<Box>& b) {
    std::vector<std::vector<float>> m(a.size(), std::vector<float>(b.size()));
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) m[i][j] = iou(a[i], b[j]);
    return m;
}

Assignment linear_assignment(const std::vector<std::vector<float>>& cost, int cols, float max_cost) {
    Assignment out;
    const int rows = static_cast<int>(cost.size());
    if (rows == 0 || cols == 0) {
        for (int r = 0; r < rows; ++r) out.unmatched_rows.push_back(r);
        for (int c = 0; c < cols; ++c) out.unmatched_cols.push_back(c);
        return out;
    }
    // Square problem; forbidden pairs and padding get a cost that is never worth taking.
    const int n = std::max(rows, cols);
    const double forbidden = 1e6;
    std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, forbidden));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            if (cost[r][c] <= max_cost) a[r + 1][c + 1] = cost[r][c];

    // O(n^3) Hungarian algorithm with potentials (1-based arrays).
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, inf);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = inf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<char> row_done(rows, false), col_done(cols, false);
    for (int j = 1; j <= n; ++j) {
        const int r = p[j] - 1, c = j - 1;
        if (r < rows && c < cols && a[r + 1][c + 1] < forbidden) {
            out.matches.emplace_back(r, c);
            row_done[r] = col_done[c] = true;
        }
    }
    for (int r = 0; r < rows; ++r)
        if (!row_done[r]) out.unmatched_rows.push_back(r);
    for (int c = 0; c < cols; ++c)
        if (!col_done[c]) out.unmatched_cols.push_back(c);
    return out;
}

}  // namespace mot
