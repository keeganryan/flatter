#include <chrono>
#include <flatter/data/matrix.h>
#include <flatter/data/lattice.h>
#include <flatter/problems.h>
#include <flatter/computation_context.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <sstream>
#include <string.h>

#include <flatter/monitor.h>
#include <flatter/flatter.h>

void print_help() {
  std::cout << "Usage: flatter [-h] [-v] [-alpha ALPHA | -rhf RHF | -delta DELTA] [-logcond LOGCOND] [INFILE [OUTFILE]]" << std::endl;
  std::cout << "\tINFILE -\tinput lattice (FPLLL format). Defaults to STDIN" << std::endl;
  std::cout << "\tOUTFILE -\toutput lattice (FPLLL format). Defaults to STDOUT" << std::endl;
  std::cout << "\t-h -\thelp message." << std::endl;
  std::cout << "\t-v -\tverbose output." << std::endl;
  std::cout << "\t-q -\tdo not output lattice." << std::endl;
  std::cout << "\t-p -\toutput profiles." << std::endl;
  std::cout << "\tReduction quality - up to one of the following. Default to RHF 1.0219" << std::endl;
  std::cout << "\t\t-alpha ALPHA -\tReduce to given parameter alpha" << std::endl;
  std::cout << "\t\t-rhf RHF -\tReduce analogous to given root hermite factor" << std::endl;
  std::cout << "\t\t-delta DELTA -\tReduce analogous to LLL with particular delta (approximate)" << std::endl;
  std::cout << "\t-logcond LOGCOND -\tBound on condition number." << std::endl;
}

std::shared_ptr<std::istream> inp_lat;
std::shared_ptr<std::ostream> out_lat;
bool show_help = false;
bool verbose = false;
bool quiet = false;
bool red_quality_set = false;
bool logcond_set = false;
bool show_profile = false;
double alpha = 0.0;
double logcond = 0.0;

bool parse_args(int argc, char** argv) {
  if (argc < 1) {
    return false;
  }

  int arg_ind = 1;
  while (arg_ind < argc) {
    char* arg = argv[arg_ind];

    if (strcmp(arg, "-h") == 0) {
      show_help = true;
      return true;
    } else if (strcmp(arg, "-v") == 0) {
      verbose = true;
    } else if (strcmp(arg, "-q") == 0) {
      if (out_lat != nullptr) {
        std::cerr << "Cannot set output file in quiet (-q) mode" << std::endl;
        return false;
      }
      quiet = true;
    } else if (strcmp(arg, "-p") == 0) {
      show_profile = true;
    } else if (strcmp(arg, "-alpha") == 0) {
      if (red_quality_set || arg_ind + 1 >= argc) {
        return false;
      }
      arg_ind++;
      double alpha_param = atof(argv[arg_ind]);
      alpha = alpha_param;
      red_quality_set = true;
    } else if (strcmp(arg, "-rhf") == 0) {
      if (red_quality_set || arg_ind + 1 >= argc) {
        return false;
      }
      arg_ind++;
      double rhf_param = atof(argv[arg_ind]);
      alpha = 2 * log2(rhf_param);
      red_quality_set = true;
    } else if (strcmp(arg, "-delta") == 0) {
      if (red_quality_set || arg_ind + 1 >= argc) {
        return false;
      }
      arg_ind++;
      double delta_param = atof(argv[arg_ind]);
      // double delta = 0.18 / sqrt(log2(this->rhf));
      // delta = 0.255 / sqrt(alpha)
      // alpha = (0.255 / delta) ** 2
      alpha = pow(0.255 / delta_param, 2);
      red_quality_set = true;
    } else if (strcmp(arg, "-logcond") == 0) {
      if (logcond_set || arg_ind + 1 >= argc) {
        return false;
      }
      arg_ind++;
      logcond = atof(argv[arg_ind]);
      logcond_set = true;
    } else {
      // Either input file or output file
      if (inp_lat == nullptr) {
        // input file is not set yet
        if (strcmp(arg, "-") == 0) {
          inp_lat.reset(&std::cin, [](...){});
        } else {
          inp_lat.reset(new std::ifstream(arg));
        }
      } else if (out_lat == nullptr) {
        // input file is set, but output is not
        if (quiet) {
          std::cerr << "Cannot set output file in quiet (-q) mode" << std::endl;
          return false;
        }
        if (strcmp(arg, "-") == 0) {
          out_lat.reset(&std::cout, [](...){});
        } else {
          out_lat.reset(new std::ofstream(arg));
        }
      } else {
        std::cerr << "Too many input/output files specified" << std::endl;
        return false;
      }
    }
    arg_ind++;
  }

  return true;
}

int main(int argc, char** argv) {
  if (!parse_args(argc, argv)) {
    print_help();
    return -1;
  }

  if (show_help) {
    print_help();
    return 0;
  }

  if (inp_lat == nullptr) {
    inp_lat.reset(&std::cin, [](...){});
  }
  if (out_lat == nullptr) {
    out_lat.reset(&std::cout, [](...){});
  }
  if (!red_quality_set) {
    // Corresponds to RHF 1.0219
    alpha = 0.06250805094100162;
  }

  flatter::Lattice L;
  (*inp_lat) >> L;

  if (verbose) {
    std::cerr << "Input lattice of rank " << L.rank() << " and dimension " << L.dimension() << std::endl;
    unsigned int max_sz = 0;
    flatter::MatrixData<mpz_t> dB = L.basis().data<mpz_t>();
    for (unsigned int i = 0; i < dB.nrows(); i++) {
      for (unsigned int j = 0; j < dB.ncols(); j++) {
        unsigned int entry_sz = mpz_sizeinbase(dB(i,j), 2);
        max_sz = std::max(max_sz, entry_sz);
      }
    }
    std::cerr << "Largest entry is " << max_sz << " bits in length." << std::endl;
    if (L.basis().is_upper_triangular()) {
      flatter::Profile prof_in(L.rank());
      double logdet = 0;
      for (unsigned int i = 0; i < L.rank(); i++) {
        signed long exp;
        double v = mpz_get_d_2exp(&exp, dB(i,i));
        prof_in[i] = log2(fabs(v)) + exp;
        logdet += prof_in[i];
      }
      std::cerr << "Lattice determinant is 2^(" << logdet << ")" << std::endl;
      if (show_profile) {
        std::cerr << "Input profile:" << std::endl;
        for (unsigned int i = 0; i < L.rank(); i++) {
          std::cerr << prof_in[i];
          if (i < L.rank() - 1) {
            std::cerr << " ";
          } else {
            std::cerr << std::endl;
          }
        }
      }
    } else {
      std::cerr << "Skipped determining input profile, as input is not lower-triangular." << std::endl;
    }
    std::cerr << "Target reduction quality alpha = " << alpha << ", rhf = " << (pow(2, alpha / 2)) << std::endl;
  }


  flatter::initialize();

  flatter::Matrix U(
    flatter::ElementType::MPZ,
    L.rank(), L.rank()
  );
  flatter::ComputationContext cc;

  auto goal = flatter::LatticeReductionGoal::from_slope(L.rank(), alpha);
  flatter::LatticeReductionParams params(L, U, 1.02);
  params.goal = goal;
  if (logcond_set) {
    params.log_cond = logcond;
  }

  flatter::LatticeReduction latred(params, cc);

  auto start = std::chrono::high_resolution_clock::now();

  if (omp_get_active_level() == 0) {
      #pragma omp parallel num_threads(cc.nthreads())
      {
          #pragma omp single
          {
            latred.solve();
          }
      }
  } else {
      #pragma omp taskgroup
      {
        latred.solve();
      }
  }
 
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

  if (verbose) {
    std::cerr << "Reduction took " << (microseconds / 1000) << " milliseconds." << std::endl;
  }
  if (show_profile) {
    std::cerr << "Output profile:" << std::endl;
    for (unsigned int i = 0; i < L.rank(); i++) {
      std::cerr << L.profile[i];
      if (i < L.rank() - 1) {
        std::cerr << " ";
      } else {
        std::cerr << std::endl;
      }
    }
  }
  if (verbose) {
    double logdet = 0;
    unsigned int n = L.rank();
    for (unsigned int i = 0; i < n; i++) {
      logdet += L.profile[i];
    }
    // Compute actual RHF
    // |v_0| = RHF^n * det^(1/n)
    // log |v_0| = n * log(RHF) + logdet / n
    // log(RHF) = (log(|v_0|) - logdet / n) / n
    double log_rhf = (L.profile[0] - logdet / n) / n;
    double rhf = pow(2, log_rhf);

    // Compute actual alpha
    double drop = L.profile.get_drop();
    double alpha = drop / n;
    std::cerr << "Achieved reduction quality alpha = " << alpha << ", rhf = " << rhf << std::endl;
  }

  if (!quiet) {
    (*out_lat) << L;
  }


  flatter::finalize();

  return 0;
}
