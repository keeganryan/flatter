#pragma once

#include <flatter/data/matrix.h>

#include <memory>

#include "profile.h"

namespace flatter {

class Lattice {
public:
    Lattice();
    Lattice(unsigned int rank, unsigned int dimension);
    Lattice(Matrix B);

    void resize(unsigned int rank, unsigned int dimension);

    Matrix basis() const;
    Matrix& basis();
    unsigned int rank() const;
    unsigned int dimension() const;

    Profile profile;

    friend std::istream& operator>>(std::istream& is, Lattice& L);
    friend std::ostream& operator<<(std::ostream& os, Lattice& L);
private:
    Matrix B;
};

}