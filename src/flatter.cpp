#include "flatter.h"

#include "monitor.h"

namespace flatter {

void initialize() {
    std::string s;
    char* logfile = getenv("FLATTER_LOG");
    if (logfile != nullptr) {
        s = std::string(logfile);
    }
    initialize(s);
}

void initialize(const std::string& logfile_name) {
    if (logfile_name.length() != 0) {
        flatter::Monitor::getInstance().set_logfile(logfile_name);
    }
}

void finalize() {
    flatter::Monitor::getInstance().stop();
}

}