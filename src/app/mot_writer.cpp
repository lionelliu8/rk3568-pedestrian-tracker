#include "mot_writer.h"

#include <iomanip>
#include <stdexcept>

MotWriter::MotWriter(const std::string& path) : file_(path) {
    if (!file_) throw std::runtime_error("cannot write " + path);
    file_ << std::fixed << std::setprecision(2);
}

void MotWriter::write(int frame, const std::vector<DetectBox>& boxes) {
    for (const auto& b : boxes) {
        file_ << frame << ',' << static_cast<int>(b.trackID) << ',' << b.x1 << ',' << b.y1 << ','
              << b.width() << ',' << b.height() << ',' << b.confidence << ",-1,-1,-1\n";
    }
}
