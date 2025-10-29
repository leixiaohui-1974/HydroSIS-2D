#include "performance_monitor.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

PerformanceMonitor::PerformanceMonitor() {
}

void PerformanceMonitor::start(const std::string& section) {
    auto& timer = timers_[section];

    if (timer.is_running) {
        std::cerr << "Warning: Timer '" << section << "' is already running" << std::endl;
        return;
    }

    timer.start_time = std::chrono::high_resolution_clock::now();
    timer.is_running = true;
}

void PerformanceMonitor::stop(const std::string& section) {
    auto stop_time = std::chrono::high_resolution_clock::now();

    auto it = timers_.find(section);
    if (it == timers_.end() || !it->second.is_running) {
        std::cerr << "Warning: Timer '" << section << "' is not running" << std::endl;
        return;
    }

    auto& timer = it->second;
    auto duration = std::chrono::duration<double>(stop_time - timer.start_time).count();

    timer.total_time += duration;
    timer.call_count++;
    timer.is_running = false;
}

double PerformanceMonitor::get_elapsed(const std::string& section) const {
    auto it = timers_.find(section);
    if (it != timers_.end()) {
        return it->second.total_time;
    }
    return 0.0;
}

int PerformanceMonitor::get_count(const std::string& section) const {
    auto it = timers_.find(section);
    if (it != timers_.end()) {
        return it->second.call_count;
    }
    return 0;
}

double PerformanceMonitor::get_average(const std::string& section) const {
    auto it = timers_.find(section);
    if (it != timers_.end() && it->second.call_count > 0) {
        return it->second.total_time / it->second.call_count;
    }
    return 0.0;
}

void PerformanceMonitor::print_report() const {
    if (timers_.empty()) {
        std::cout << "No performance data available" << std::endl;
        return;
    }

    // Calculate total time
    double total_time = 0.0;
    for (const auto& pair : timers_) {
        total_time += pair.second.total_time;
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Performance Report                            ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";

    std::cout << std::left << std::setw(25) << "║ Section"
              << std::right << std::setw(12) << "Time (s)"
              << std::setw(10) << "Calls"
              << std::setw(12) << "Avg (ms)"
              << std::setw(10) << "Percent" << "  ║\n";

    std::cout << "╠════════════════════════════════════════════════════════════╣\n";

    // Sort by total time
    std::vector<std::pair<std::string, TimerData>> sorted_timers(timers_.begin(), timers_.end());
    std::sort(sorted_timers.begin(), sorted_timers.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_time > b.second.total_time;
              });

    for (const auto& pair : sorted_timers) {
        const std::string& name = pair.first;
        const TimerData& timer = pair.second;

        double avg_ms = (timer.total_time / timer.call_count) * 1000.0;
        double percent = (timer.total_time / total_time) * 100.0;

        std::cout << "║ " << std::left << std::setw(23) << name.substr(0, 23)
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << timer.total_time
                  << std::setw(10) << timer.call_count
                  << std::setw(12) << std::fixed << std::setprecision(3) << avg_ms
                  << std::setw(9) << std::fixed << std::setprecision(1) << percent << "% ║\n";
    }

    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ " << std::left << std::setw(23) << "TOTAL"
              << std::right << std::setw(12) << std::fixed << std::setprecision(3) << total_time
              << std::setw(10) << ""
              << std::setw(12) << ""
              << std::setw(10) << "100.0%" << "  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
}

void PerformanceMonitor::save_report(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    file << "# HydroSIS-2D Performance Report\n\n";
    file << "Section,Total Time (s),Call Count,Average Time (ms),Percentage\n";

    double total_time = 0.0;
    for (const auto& pair : timers_) {
        total_time += pair.second.total_time;
    }

    std::vector<std::pair<std::string, TimerData>> sorted_timers(timers_.begin(), timers_.end());
    std::sort(sorted_timers.begin(), sorted_timers.end(),
              [](const auto& a, const auto& b) {
                  return a.second.total_time > b.second.total_time;
              });

    for (const auto& pair : sorted_timers) {
        const std::string& name = pair.first;
        const TimerData& timer = pair.second;

        double avg_ms = (timer.total_time / timer.call_count) * 1000.0;
        double percent = (timer.total_time / total_time) * 100.0;

        file << name << ","
             << timer.total_time << ","
             << timer.call_count << ","
             << avg_ms << ","
             << percent << "\n";
    }

    file << "\nTotal," << total_time << ",,,100.0\n";

    file.close();
    std::cout << "Performance report saved to " << filename << std::endl;
}

void PerformanceMonitor::reset() {
    timers_.clear();
}
