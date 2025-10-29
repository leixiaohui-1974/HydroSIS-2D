#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include "hydrosis_types.h"
#include <string>
#include <map>

/**
 * @brief Configuration file reader for simulation parameters
 *
 * Supports simple INI-style configuration files
 */
class ConfigReader {
public:
    ConfigReader();

    /**
     * @brief Load configuration from file
     * @param filename Configuration file path
     * @return true if successful
     */
    bool load(const std::string& filename);

    /**
     * @brief Parse configuration and fill SimParams
     * @param params Output simulation parameters
     * @return true if successful
     */
    bool parse_params(SimParams& params);

    /**
     * @brief Get string value
     */
    std::string get_string(const std::string& key, const std::string& default_val = "") const;

    /**
     * @brief Get integer value
     */
    int get_int(const std::string& key, int default_val = 0) const;

    /**
     * @brief Get real value
     */
    real_t get_real(const std::string& key, real_t default_val = 0.0) const;

    /**
     * @brief Get boolean value
     */
    bool get_bool(const std::string& key, bool default_val = false) const;

    /**
     * @brief Print all loaded parameters
     */
    void print() const;

private:
    std::map<std::string, std::string> params_;

    /**
     * @brief Trim whitespace from string
     */
    std::string trim(const std::string& str) const;

    /**
     * @brief Remove comments from line
     */
    std::string remove_comment(const std::string& line) const;
};

#endif // CONFIG_READER_H
