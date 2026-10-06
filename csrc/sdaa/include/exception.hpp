#pragma once

#include <exception>
#include <string>

// Main ELF 0x200e0 constructor, 0x1cf40 what(), original exception.hpp:9.
class EPException : public std::exception {
    std::string message;
public:
    EPException(std::string reason, const char* file, int line)
        : message(reason + " at " + file + ":" + std::to_string(line)) {}
    const char* what() const noexcept override { return message.c_str(); }
};
