#pragma once

#include <memory>

namespace flatter {

class Profile {
public:
    Profile();
    Profile(unsigned int n);

    bool is_valid() const;

    double &operator[](unsigned int i);
    const double &operator[](unsigned int i) const;

    Profile subprofile(unsigned int start, unsigned int end);
    
    double get_drop() const;
    double get_spread() const;

private:
    bool is_valid_;
    std::shared_ptr<double[]> profile_elems;
    unsigned int n;
};

}