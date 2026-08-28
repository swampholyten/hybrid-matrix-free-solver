#include "cdr_solver.hpp"

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <string>
#include <vector>

using namespace dealii;

int main(int argc, char *argv[]) {
  Parameters parameters;
  std::vector<char *> kept = {argv[0]};

  const auto take_int = [&](int &i) {
    return Utilities::string_to_int(argv[++i]);
  };

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--threads") {
      parameters.n_threads = take_int(i);
      continue;
    }
    if (arg.rfind("--threads=", 0) == 0) {
      parameters.n_threads = Utilities::string_to_int(arg.substr(10));
      continue;
    }
    if (arg == "--refine") {
      parameters.n_refinements = take_int(i);
      continue;
    }
    if (arg.rfind("--refine=", 0) == 0) {
      parameters.n_refinements = Utilities::string_to_int(arg.substr(9));
      continue;
    }
    kept.push_back(argv[i]);
  }

  int filtered_argc = static_cast<int>(kept.size());
  char **filtered_argv = kept.data();

  const unsigned int thread_limit =
      parameters.n_threads == 0 ? numbers::invalid_unsigned_int
                                : parameters.n_threads;

  // deal.II 9.5 calls Kokkos::initialize() before it applies the
  // MPI_InitFinalize thread argument. Kokkos then keeps a persistent
  // thread pool. If that pool has the same size as TBB, the two runtimes
  // oversubscribe the cores and MatrixFree's cell_loop does not speed up.
  // Pin Kokkos to one thread first; MPI_InitFinalize then sizes TBB.
  MultithreadInfo::set_thread_limit(1);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
      filtered_argc, filtered_argv, thread_limit);

  CdrProblem<2, 2> problem(parameters);
  problem.run();
}
