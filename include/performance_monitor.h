#ifndef PERFORMANCE_MONITOR_H
#define PERFORMANCE_MONITOR_H

#include <chrono>
#include <string>
#include <map>
#include <vector>

/**
 * @brief Performance monitoring and profiling tool
 */
class PerformanceMonitor {
public:
    PerformanceMonitor();

    /**
     * @brief Start timing a section
     */
    void start(const std::string& section);

    /**
     * @brief Stop timing a section
     */
    void stop(const std::string& section);

    /**
     * @brief Get elapsed time for a section (in seconds)
     */
    double get_elapsed(const std::string& section) const;

    /**
     * @brief Get call count for a section
     */
    int get_count(const std::string& section) const;

    /**
     * @brief Get average time per call
     */
    double get_average(const std::string& section) const;

    /**
     * @brief Print performance report
     */
    void print_report() const;

    /**
     * @brief Save report to file
     */
    void save_report(const std::string& filename) const;

    /**
     * @brief Reset all timers
     */
    void reset();

private:
    struct TimerData {
        double total_time;
        int call_count;
        std::chrono::high_resolution_clock::time_point start_time;
        bool is_running;

        TimerData() : total_time(0.0), call_count(0), is_running(false) {}
    };

    std::map<std::string, TimerData> timers_;
};

/**
 * @brief RAII-style timer for automatic timing
 */
class ScopedTimer {
public:
    ScopedTimer(PerformanceMonitor& monitor, const std::string& section)
        : monitor_(monitor), section_(section) {
        monitor_.start(section_);
    }

    ~ScopedTimer() {
        monitor_.stop(section_);
    }

private:
    PerformanceMonitor& monitor_;
    std::string section_;
};

#endif // PERFORMANCE_MONITOR_H
