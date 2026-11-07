#ifndef PEDESTRIAN_TRACKER_MOT_ASSIGNMENT_H
#define PEDESTRIAN_TRACKER_MOT_ASSIGNMENT_H

#include <array>
#include <utility>
#include <vector>

namespace mot {

using Box = std::array<float, 4>;  // x1, y1, x2, y2 in pixels

float iou(const Box& a, const Box& b);

// rows x cols IoU matrix.
std::vector<std::vector<float>> iou_matrix(const std::vector<Box>& a, const std::vector<Box>& b);

struct Assignment {
    std::vector<std::pair<int, int>> matches;  // (row, col)
    std::vector<int> unmatched_rows;
    std::vector<int> unmatched_cols;
};

// Minimum-cost bipartite matching (Hungarian algorithm, rectangular matrices allowed).
// `cols` is given explicitly so that a matrix with zero rows still reports its columns
// as unmatched. Pairs whose cost is above `max_cost` are never matched.
Assignment linear_assignment(const std::vector<std::vector<float>>& cost, int cols, float max_cost);

}  // namespace mot

#endif
