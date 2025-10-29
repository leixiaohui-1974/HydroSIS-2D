#include "vtk_writer.h"
#include <iostream>
#include <cmath>
#include <cstring>

void VTKWriter::write_structured_grid(
    const std::string& filename,
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t xmin, real_t ymin,
    real_t time,
    int halo_width) {

    // Use ASCII format by default (easier debugging)
    write_ascii(filename, cells, nx, ny, dx, dy, xmin, ymin, time, halo_width);
}

void VTKWriter::write_ascii(
    const std::string& filename,
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t xmin, real_t ymin,
    real_t time,
    int halo_width) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Compute interior grid dimensions (exclude halos)
    int nx_interior = nx - 2 * halo_width;
    int ny_interior = ny - 2 * halo_width;
    int n_points = nx_interior * ny_interior;

    // VTK header
    file << "# vtk DataFile Version 3.0\n";
    file << "HydroSIS-2D simulation at time " << time << " s\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx_interior << " " << ny_interior << " 1\n";
    file << "ORIGIN " << (xmin + halo_width * dx) << " "
         << (ymin + halo_width * dy) << " 0.0\n";
    file << "SPACING " << dx << " " << dy << " 1.0\n";
    file << "POINT_DATA " << n_points << "\n";

    // Write water depth
    file << "\nSCALARS depth float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            file << cells[idx].U.h << "\n";
        }
    }

    // Write velocity components
    file << "\nVECTORS velocity float\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            real_t u = 0.0, v = 0.0;
            if (cells[idx].U.h > Constants::MIN_DEPTH) {
                u = cells[idx].U.qx / cells[idx].U.h;
                v = cells[idx].U.qy / cells[idx].U.h;
            }
            file << u << " " << v << " 0.0\n";
        }
    }

    // Write velocity magnitude
    file << "\nSCALARS velocity_magnitude float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            real_t u = 0.0, v = 0.0;
            if (cells[idx].U.h > Constants::MIN_DEPTH) {
                u = cells[idx].U.qx / cells[idx].U.h;
                v = cells[idx].U.qy / cells[idx].U.h;
            }
            real_t vel_mag = sqrt(u * u + v * v);
            file << vel_mag << "\n";
        }
    }

    // Write bed elevation
    file << "\nSCALARS bed_elevation float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            file << cells[idx].z << "\n";
        }
    }

    // Write water surface elevation
    file << "\nSCALARS surface_elevation float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            real_t eta = cells[idx].z + cells[idx].U.h;
            file << eta << "\n";
        }
    }

    // Write Froude number
    file << "\nSCALARS froude_number float 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            real_t Fr = 0.0;
            if (cells[idx].U.h > Constants::MIN_DEPTH) {
                real_t u = cells[idx].U.qx / cells[idx].U.h;
                real_t v = cells[idx].U.qy / cells[idx].U.h;
                real_t vel_mag = sqrt(u * u + v * v);
                real_t c = sqrt(Constants::GRAVITY * cells[idx].U.h);
                Fr = vel_mag / (c + 1e-10);
            }
            file << Fr << "\n";
        }
    }

    // Write wet/dry flag
    file << "\nSCALARS wet_dry int 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            file << (cells[idx].is_wet ? 1 : 0) << "\n";
        }
    }

    file.close();
}

std::string VTKWriter::generate_filename(
    const std::string& base,
    int step,
    const std::string& ext) {

    std::ostringstream oss;
    oss << base << "_" << std::setfill('0') << std::setw(6) << step << ext;
    return oss.str();
}
