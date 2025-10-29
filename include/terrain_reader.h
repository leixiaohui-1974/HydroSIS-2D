#ifndef TERRAIN_READER_H
#define TERRAIN_READER_H

#include "hydrosis_types.h"
#include <string>
#include <vector>

/**
 * @brief Terrain file reader for loading bed elevation data
 *
 * Supports multiple formats:
 * - ASCII Grid (.asc) - ArcGIS/GDAL format
 * - Binary Grid (.bin) - Raw binary format
 * - Future: NetCDF (.nc)
 */
class TerrainReader {
public:
    /**
     * @brief Terrain data structure
     */
    struct TerrainData {
        int ncols;              ///< Number of columns
        int nrows;              ///< Number of rows
        real_t xllcorner;       ///< X coordinate of lower-left corner
        real_t yllcorner;       ///< Y coordinate of lower-left corner
        real_t cellsize;        ///< Cell size (uniform dx = dy)
        real_t nodata_value;    ///< No-data value
        std::vector<real_t> elevation;  ///< Elevation data (row-major)

        TerrainData() : ncols(0), nrows(0), xllcorner(0.0), yllcorner(0.0),
                        cellsize(1.0), nodata_value(-9999.0) {}
    };

    /**
     * @brief Load terrain from ASCII Grid file
     * @param filename Path to .asc file
     * @param terrain Output terrain data
     * @return true if successful
     */
    static bool load_ascii_grid(const std::string& filename, TerrainData& terrain);

    /**
     * @brief Load terrain from binary file
     * @param filename Path to .bin file
     * @param terrain Output terrain data
     * @param ncols Number of columns (must be known)
     * @param nrows Number of rows (must be known)
     * @return true if successful
     */
    static bool load_binary(const std::string& filename, TerrainData& terrain,
                           int ncols, int nrows);

    /**
     * @brief Interpolate terrain data to simulation grid
     * @param terrain Input terrain data
     * @param z_output Output elevation array (size: nx * ny)
     * @param nx Number of cells in x-direction (output grid)
     * @param ny Number of cells in y-direction (output grid)
     * @param xmin Minimum x-coordinate (output grid)
     * @param ymin Minimum y-coordinate (output grid)
     * @param dx Cell size in x-direction (output grid)
     * @param dy Cell size in y-direction (output grid)
     * @return true if successful
     */
    static bool interpolate_to_grid(const TerrainData& terrain,
                                    real_t* z_output,
                                    int nx, int ny,
                                    real_t xmin, real_t ymin,
                                    real_t dx, real_t dy);

    /**
     * @brief Create a synthetic terrain for testing
     * @param z_output Output elevation array
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     * @param xmin Minimum x-coordinate
     * @param ymin Minimum y-coordinate
     * @param dx Cell size in x-direction
     * @param dy Cell size in y-direction
     * @param terrain_type Type of synthetic terrain
     *        0: Flat
     *        1: Parabolic channel
     *        2: Gaussian hill
     *        3: Linear slope
     */
    static void create_synthetic_terrain(real_t* z_output,
                                         int nx, int ny,
                                         real_t xmin, real_t ymin,
                                         real_t dx, real_t dy,
                                         int terrain_type = 0);

    /**
     * @brief Save terrain to ASCII Grid file
     * @param filename Output filename
     * @param terrain Terrain data to save
     * @return true if successful
     */
    static bool save_ascii_grid(const std::string& filename, const TerrainData& terrain);

    /**
     * @brief Print terrain statistics
     * @param terrain Terrain data
     */
    static void print_statistics(const TerrainData& terrain);

private:
    /**
     * @brief Trim whitespace from string
     */
    static std::string trim(const std::string& str);

    /**
     * @brief Bilinear interpolation
     */
    static real_t bilinear_interpolate(real_t q11, real_t q12, real_t q21, real_t q22,
                                       real_t x, real_t y);
};

#endif // TERRAIN_READER_H
