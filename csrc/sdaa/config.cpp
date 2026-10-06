#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pwd.h>
#include <unistd.h>
#include <string>
#include <unordered_map>

std::unordered_map<std::string, std::string> envMap;

// Main ELF 0x3a840. Original user-home config parsing, first '=' per line.
void getEnvFromFile() {
    const passwd* user = getpwuid(getuid());
    if (!user || !user->pw_dir) return;
    char path[256];
    std::snprintf(path, sizeof(path), "%s/.tccl.conf", user->pw_dir);
    FILE* file = std::fopen(path, "r");
    if (!file) return;
    char* line = nullptr;
    std::size_t capacity = 0;
    ssize_t length;
    while ((length = getline(&line, &capacity, file)) != -1) {
        if (line[length - 1] == '\n') line[length - 1] = '\0';
        for (int i = 0; line[i] != '\0'; ++i) {
            if (line[i] != '=') continue;
            char key[256], value[256];
            std::strncpy(key, line, i);
            key[i] = '\0';
            std::strncpy(value, line + i + 1, length - i - 1);
            envMap[key] = value;
            break;
        }
    }
    std::free(line);
    std::fclose(file);
}
