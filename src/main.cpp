#include "cdr_solver.hpp"

#include <deal.II/base/utilities.h>

#include <string>
#include <vector>

using namespace dealii;

int main(int argc, char *argv[]) {
  Parameters parameters;
  std::vector<char *> kept = {argv[0]};

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--threads" && i + 1 < argc) {
      parameters.n_threads = Utilities::string_to_int(argv[++i]);
      continue;
    }
    if (arg.rfind("--threads=", 0) == 0) {
      parameters.n_threads = Utilities::string_to_int(arg.substr(10));
      continue;
    }
    kept.push_back(argv[i]);
  }

  int filtered_argc = static_cast<int>(kept.size());
  char **filtered_argv = kept.data();

  const unsigned int thread_limit =
      parameters.n_threads == 0 ? numbers::invalid_unsigned_int
                                : parameters.n_threads;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
      filtered_argc, filtered_argv, thread_limit);

  CdrProblem<2, 2> problem(parameters);
  problem.run();
}
