#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include "hydrosis_types.h"
#include <string>
#include <fstream>
#include <iomanip>

/**
 * @brief VTK file writer for visualization in ParaView
 */
class VTKWriter {
public:
    /**
     * @brief Write structured grid data to VTK file
     * @param filename Output filename
     * @param cells Cell data array
     * @param nx, ny Grid dimensions
     * @param dx, dy Grid spacing
     * @param xmin, ymin Domain origin
     * @param time Current simulation time
     * @param halo_width Number of ghost cells to exclude
     */
    static void write_structured_grid(
        const std::string& filename,
        const CellData* cells,
        int nx, int ny,
        real_t dx, real_t dy,
        real_t xmin, real_t ymin,
        real_t time,
        int halo_width = 2);

    /**
     * @brief Write ASCII VTK file (easier debugging)
     */
    static void write_ascii(
        const std::string& filename,
        const CellData* cells,
        int nx, int ny,
        real_t dx, real_t dy,
        real_t xmin, real_t ymin,
        real_t time,
        int halo_width = 2);

    /**
     * @brief Write binary VTK file (smaller size, faster)
     */
    static void write_binary(
        const std::string& filename,
        const CellData* cells,
        int nx, int ny,
        real_t dx, real_t dy,
        real_t xmin, real_t ymin,
        real_t time,
        int halo_width = 2);

    /**
     * @brief Generate filename with timestep number
     */
    static std::string generate_filename(
        const std::string& base,
        int step,
        const std::string& ext = ".vtk");

private:
    static void swap_endian_float(float* val);
    static void swap_endian_double(double* val);
};

#endif // VTK_WRITER_H
