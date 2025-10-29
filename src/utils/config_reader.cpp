#include "config_reader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

ConfigReader::ConfigReader() {
}

std::string ConfigReader::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";

    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

std::string ConfigReader::remove_comment(const std::string& line) const {
    size_t pos = line.find('#');
    if (pos != std::string::npos) {
        return line.substr(0, pos);
    }
    return line;
}

bool ConfigReader::load(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open configuration file: " << filename << std::endl;
        return false;
    }

    std::string line;
    int line_num = 0;

    while (std::getline(file, line)) {
        line_num++;

        // Remove comments
        line = remove_comment(line);
        line = trim(line);

        // Skip empty lines
        if (line.empty()) continue;

        // Parse key = value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            std::cerr << "Warning: Invalid line " << line_num << ": " << line << std::endl;
            continue;
        }

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));

        params_[key] = value;
    }

    file.close();

    std::cout << "Loaded " << params_.size() << " parameters from " << filename << std::endl;

    return true;
}

std::string ConfigReader::get_string(const std::string& key, const std::string& default_val) const {
    auto it = params_.find(key);
    if (it != params_.end()) {
        return it->second;
    }
    return default_val;
}

int ConfigReader::get_int(const std::string& key, int default_val) const {
    auto it = params_.find(key);
    if (it != params_.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            std::cerr << "Warning: Invalid integer for key '" << key << "'" << std::endl;
        }
    }
    return default_val;
}

real_t ConfigReader::get_real(const std::string& key, real_t default_val) const {
    auto it = params_.find(key);
    if (it != params_.end()) {
        try {
            return std::stod(it->second);
        } catch (...) {
            std::cerr << "Warning: Invalid real number for key '" << key << "'" << std::endl;
        }
    }
    return default_val;
}

bool ConfigReader::get_bool(const std::string& key, bool default_val) const {
    auto it = params_.find(key);
    if (it != params_.end()) {
        std::string val = it->second;
        std::transform(val.begin(), val.end(), val.begin(), ::tolower);

        if (val == "true" || val == "yes" || val == "1" || val == "on") {
            return true;
        } else if (val == "false" || val == "no" || val == "0" || val == "off") {
            return false;
        }
    }
    return default_val;
}

bool ConfigReader::parse_params(SimParams& params) {
    // Grid parameters
    params.nx = get_int("nx", 512);
    params.ny = get_int("ny", 512);
    params.dx = get_real("dx", 1.0);
    params.dy = get_real("dy", 1.0);
    params.xmin = get_real("xmin", 0.0);
    params.ymin = get_real("ymin", 0.0);
    params.xmax = params.xmin + params.nx * params.dx;
    params.ymax = params.ymin + params.ny * params.dy;

    // Physical parameters
    params.g = get_real("gravity", Constants::GRAVITY);
    params.cfl = get_real("cfl", 0.5);
    params.h_dry = get_real("h_dry", Constants::DRY_TOLERANCE);
    params.friction_type = get_real("friction_type", 0.0);

    // Time parameters
    params.t_start = get_real("t_start", 0.0);
    params.t_end = get_real("t_end", 10.0);
    params.dt_max = get_real("dt_max", 0.1);
    params.output_interval = get_real("output_interval", 1.0);

    // Boundary conditions
    params.bc_type[0] = get_int("bc_left", 0);
    params.bc_type[1] = get_int("bc_right", 0);
    params.bc_type[2] = get_int("bc_bottom", 0);
    params.bc_type[3] = get_int("bc_top", 0);

    // Solver options
    params.use_lts = get_bool("use_lts", false);
    params.riemann_solver = get_int("riemann_solver", 1);
    params.slope_limiter = get_int("slope_limiter", 0);
    params.order = get_int("order", 2);

    return true;
}

void ConfigReader::print() const {
    std::cout << "\n=== Configuration Parameters ===" << std::endl;
    for (const auto& pair : params_) {
        std::cout << "  " << pair.first << " = " << pair.second << std::endl;
    }
    std::cout << "================================\n" << std::endl;
}
