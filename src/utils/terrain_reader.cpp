#include "terrain_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

std::string TerrainReader::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

bool TerrainReader::load_ascii_grid(const std::string& filename, TerrainData& terrain) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open terrain file: " << filename << std::endl;
        return false;
    }

    std::cout << "Loading terrain from: " << filename << std::endl;

    // Read header
    std::string line, key;
    real_t value;

    // ncols
    if (!std::getline(file, line)) return false;
    std::istringstream(line) >> key >> value;
    terrain.ncols = static_cast<int>(value);

    // nrows
    if (!std::getline(file, line)) return false;
    std::istringstream(line) >> key >> value;
    terrain.nrows = static_cast<int>(value);

    // xllcorner
    if (!std::getline(file, line)) return false;
    std::istringstream(line) >> key >> terrain.xllcorner;

    // yllcorner
    if (!std::getline(file, line)) return false;
    std::istringstream(line) >> key >> terrain.yllcorner;

    // cellsize
    if (!std::getline(file, line)) return false;
    std::istringstream(line) >> key >> terrain.cellsize;

    // nodata_value (optional)
    std::streampos pos = file.tellg();
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> key;
        if (key == "NODATA_value" || key == "nodata_value") {
            iss >> terrain.nodata_value;
        } else {
            // Not a header line, go back
            file.seekg(pos);
        }
    }

    std::cout << "  Grid size: " << terrain.ncols << " x " << terrain.nrows << std::endl;
    std::cout << "  Lower-left corner: (" << terrain.xllcorner << ", " << terrain.yllcorner << ")" << std::endl;
    std::cout << "  Cell size: " << terrain.cellsize << " m" << std::endl;

    // Allocate memory
    int n_cells = terrain.ncols * terrain.nrows;
    terrain.elevation.resize(n_cells);

    // Read elevation data (row-major, from top to bottom)
    int count = 0;
    for (int j = terrain.nrows - 1; j >= 0; j--) {  // Read from top row
        for (int i = 0; i < terrain.ncols; i++) {
            if (!(file >> value)) {
                std::cerr << "Error: Unexpected end of file" << std::endl;
                return false;
            }
            terrain.elevation[j * terrain.ncols + i] = value;
            count++;
        }
    }

    if (count != n_cells) {
        std::cerr << "Error: Expected " << n_cells << " values, got " << count << std::endl;
        return false;
    }

    file.close();

    std::cout << "  Elevation data loaded: " << count << " cells" << std::endl;

    return true;
}

bool TerrainReader::load_binary(const std::string& filename, TerrainData& terrain,
                                int ncols, int nrows) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open binary terrain file: " << filename << std::endl;
        return false;
    }

    terrain.ncols = ncols;
    terrain.nrows = nrows;
    terrain.cellsize = 1.0;  // Default, should be set externally

    int n_cells = ncols * nrows;
    terrain.elevation.resize(n_cells);

    // Read raw binary data (assuming float32)
    std::vector<float> temp(n_cells);
    file.read(reinterpret_cast<char*>(temp.data()), n_cells * sizeof(float));

    if (!file.good()) {
        std::cerr << "Error: Failed to read binary data" << std::endl;
        return false;
    }

    // Convert to real_t
    for (int i = 0; i < n_cells; i++) {
        terrain.elevation[i] = static_cast<real_t>(temp[i]);
    }

    file.close();

    std::cout << "Binary terrain loaded: " << ncols << " x " << nrows << std::endl;

    return true;
}

real_t TerrainReader::bilinear_interpolate(real_t q11, real_t q12, real_t q21, real_t q22,
                                           real_t x, real_t y) {
    // Bilinear interpolation formula
    // q11: value at (0, 0)
    // q12: value at (0, 1)
    // q21: value at (1, 0)
    // q22: value at (1, 1)
    // x, y: normalized coordinates [0, 1]

    real_t r1 = q11 * (1.0 - x) + q21 * x;
    real_t r2 = q12 * (1.0 - x) + q22 * x;
    return r1 * (1.0 - y) + r2 * y;
}

bool TerrainReader::interpolate_to_grid(const TerrainData& terrain,
                                        real_t* z_output,
                                        int nx, int ny,
                                        real_t xmin, real_t ymin,
                                        real_t dx, real_t dy) {
    if (terrain.elevation.empty()) {
        std::cerr << "Error: Terrain data is empty" << std::endl;
        return false;
    }

    std::cout << "Interpolating terrain to simulation grid..." << std::endl;
    std::cout << "  Input grid: " << terrain.ncols << " x " << terrain.nrows << std::endl;
    std::cout << "  Output grid: " << nx << " x " << ny << std::endl;

    int points_outside = 0;
    int points_nodata = 0;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Cell center coordinates in simulation grid
            real_t x = xmin + (i + 0.5) * dx;
            real_t y = ymin + (j + 0.5) * dy;

            // Convert to terrain grid coordinates
            real_t x_terrain = (x - terrain.xllcorner) / terrain.cellsize;
            real_t y_terrain = (y - terrain.yllcorner) / terrain.cellsize;

            // Check if outside terrain bounds
            if (x_terrain < 0 || x_terrain >= terrain.ncols - 1 ||
                y_terrain < 0 || y_terrain >= terrain.nrows - 1) {
                // Outside terrain domain - use zero elevation
                z_output[j * nx + i] = 0.0;
                points_outside++;
                continue;
            }

            // Bilinear interpolation
            int i1 = static_cast<int>(std::floor(x_terrain));
            int j1 = static_cast<int>(std::floor(y_terrain));
            int i2 = i1 + 1;
            int j2 = j1 + 1;

            // Ensure indices are valid
            i2 = std::min(i2, terrain.ncols - 1);
            j2 = std::min(j2, terrain.nrows - 1);

            // Get corner values
            real_t q11 = terrain.elevation[j1 * terrain.ncols + i1];
            real_t q21 = terrain.elevation[j1 * terrain.ncols + i2];
            real_t q12 = terrain.elevation[j2 * terrain.ncols + i1];
            real_t q22 = terrain.elevation[j2 * terrain.ncols + i2];

            // Check for nodata values
            if (q11 == terrain.nodata_value || q21 == terrain.nodata_value ||
                q12 == terrain.nodata_value || q22 == terrain.nodata_value) {
                z_output[j * nx + i] = 0.0;
                points_nodata++;
                continue;
            }

            // Normalized coordinates within cell [0, 1]
            real_t x_norm = x_terrain - i1;
            real_t y_norm = y_terrain - j1;

            // Interpolate
            z_output[j * nx + i] = bilinear_interpolate(q11, q12, q21, q22, x_norm, y_norm);
        }
    }

    if (points_outside > 0) {
        std::cout << "  Warning: " << points_outside << " points outside terrain domain (set to 0)" << std::endl;
    }
    if (points_nodata > 0) {
        std::cout << "  Warning: " << points_nodata << " points with nodata (set to 0)" << std::endl;
    }

    std::cout << "  Interpolation complete" << std::endl;

    return true;
}

void TerrainReader::create_synthetic_terrain(real_t* z_output,
                                             int nx, int ny,
                                             real_t xmin, real_t ymin,
                                             real_t dx, real_t dy,
                                             int terrain_type) {
    real_t xmax = xmin + nx * dx;
    real_t ymax = ymin + ny * dy;
    real_t x_center = 0.5 * (xmin + xmax);
    real_t y_center = 0.5 * (ymin + ymax);
    real_t Lx = xmax - xmin;
    real_t Ly = ymax - ymin;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            real_t x = xmin + (i + 0.5) * dx;
            real_t y = ymin + (j + 0.5) * dy;

            real_t z = 0.0;

            switch (terrain_type) {
                case 0:
                    // Flat bed
                    z = 0.0;
                    break;

                case 1:
                    // Parabolic channel
                    z = 0.2 * (x - x_center) * (x - x_center) / (Lx * Lx) * 100.0;
                    break;

                case 2:
                    // Gaussian hill
                    {
                        real_t dx_hill = (x - x_center) / Lx;
                        real_t dy_hill = (y - y_center) / Ly;
                        real_t r2 = dx_hill * dx_hill + dy_hill * dy_hill;
                        z = 5.0 * std::exp(-20.0 * r2);
                    }
                    break;

                case 3:
                    // Linear slope in x-direction
                    z = 0.01 * (x - xmin);
                    break;

                case 4:
                    // V-shaped valley
                    {
                        real_t dist_from_center = std::abs(y - y_center);
                        z = 0.1 * dist_from_center;
                    }
                    break;

                case 5:
                    // Step (dam)
                    z = (x > x_center) ? 0.0 : 2.0;
                    break;

                default:
                    z = 0.0;
            }

            z_output[j * nx + i] = z;
        }
    }

    std::cout << "Synthetic terrain created (type " << terrain_type << ")" << std::endl;
}

bool TerrainReader::save_ascii_grid(const std::string& filename, const TerrainData& terrain) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file: " << filename << std::endl;
        return false;
    }

    // Write header
    file << "ncols         " << terrain.ncols << "\n";
    file << "nrows         " << terrain.nrows << "\n";
    file << "xllcorner     " << terrain.xllcorner << "\n";
    file << "yllcorner     " << terrain.yllcorner << "\n";
    file << "cellsize      " << terrain.cellsize << "\n";
    file << "NODATA_value  " << terrain.nodata_value << "\n";

    // Write data (row-major, from top to bottom)
    for (int j = terrain.nrows - 1; j >= 0; j--) {
        for (int i = 0; i < terrain.ncols; i++) {
            file << terrain.elevation[j * terrain.ncols + i];
            if (i < terrain.ncols - 1) file << " ";
        }
        file << "\n";
    }

    file.close();

    std::cout << "Terrain saved to: " << filename << std::endl;

    return true;
}

void TerrainReader::print_statistics(const TerrainData& terrain) {
    if (terrain.elevation.empty()) {
        std::cout << "Terrain data is empty" << std::endl;
        return;
    }

    real_t z_min = terrain.elevation[0];
    real_t z_max = terrain.elevation[0];
    real_t z_sum = 0.0;
    int count = 0;

    for (const auto& z : terrain.elevation) {
        if (z != terrain.nodata_value) {
            z_min = std::min(z_min, z);
            z_max = std::max(z_max, z);
            z_sum += z;
            count++;
        }
    }

    real_t z_mean = (count > 0) ? z_sum / count : 0.0;

    std::cout << "\n=== Terrain Statistics ===" << std::endl;
    std::cout << "Grid size:    " << terrain.ncols << " x " << terrain.nrows << std::endl;
    std::cout << "Cell size:    " << terrain.cellsize << " m" << std::endl;
    std::cout << "Elevation range: [" << z_min << ", " << z_max << "] m" << std::endl;
    std::cout << "Mean elevation:  " << z_mean << " m" << std::endl;
    std::cout << "Valid cells:     " << count << " / " << terrain.elevation.size() << std::endl;
    std::cout << "=========================\n" << std::endl;
}
