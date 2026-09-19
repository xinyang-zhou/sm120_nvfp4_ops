#pragma once

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sm120_nvfp4_benchmark {

inline std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &time);
#else
  gmtime_r(&time, &utc);
#endif
  std::ostringstream stream;
  stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return stream.str();
}

inline std::string json_escape(const std::string& value) {
  std::ostringstream stream;
  for (unsigned char character : value) {
    switch (character) {
      case '"': stream << "\\\""; break;
      case '\\': stream << "\\\\"; break;
      case '\b': stream << "\\b"; break;
      case '\f': stream << "\\f"; break;
      case '\n': stream << "\\n"; break;
      case '\r': stream << "\\r"; break;
      case '\t': stream << "\\t"; break;
      default:
        if (character < 0x20) {
          stream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec << std::setfill(' ');
        } else {
          stream << static_cast<char>(character);
        }
    }
  }
  return stream.str();
}

inline std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '"') escaped += '"';
    escaped += character;
  }
  escaped += '"';
  return escaped;
}

inline std::string json_number(double value, int precision = 9) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

inline std::string command_line(int argc, char** argv) {
  std::ostringstream stream;
  for (int i = 0; i < argc; ++i) {
    if (i != 0) stream << ' ';
    const std::string argument(argv[i]);
    if (argument.find_first_of(" \t\n\"'\\$`") == std::string::npos) {
      stream << argument;
      continue;
    }
    stream << '\'';
    for (char character : argument) {
      if (character == '\'') stream << "'\\''";
      else stream << character;
    }
    stream << '\'';
  }
  return stream.str();
}

inline void write_text_file(const std::string& path, const std::string& text) {
  if (path.empty()) return;
  std::ofstream output(path, std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot open output file: " + path);
  }
  output << text;
  if (!output) {
    throw std::runtime_error("failed while writing output file: " + path);
  }
}

inline void append_csv_row(
    const std::string& path, const std::string& header,
    const std::string& row) {
  if (path.empty()) return;
  bool needs_header = true;
  {
    std::ifstream existing(path, std::ios::binary);
    if (existing && existing.peek() != std::ifstream::traits_type::eof()) {
      std::string existing_header;
      std::getline(existing, existing_header);
      if (!existing_header.empty() && existing_header.back() == '\r') {
        existing_header.pop_back();
      }
      if (existing_header != header) {
        throw std::runtime_error(
            "CSV header does not match the requested schema: " + path);
      }
      needs_header = false;
    }
  }
  std::ofstream output(path, std::ios::out | std::ios::app);
  if (!output) {
    throw std::runtime_error("cannot open CSV output file: " + path);
  }
  if (needs_header) output << header << '\n';
  output << row << '\n';
  if (!output) {
    throw std::runtime_error("failed while writing CSV output file: " + path);
  }
}

}  // namespace sm120_nvfp4_benchmark
