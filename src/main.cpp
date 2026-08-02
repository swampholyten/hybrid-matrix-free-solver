#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <iostream>

using namespace dealii;

int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  const MPI_Comm comm = MPI_COMM_WORLD;
  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);
  const unsigned int n_ranks = Utilities::MPI::n_mpi_processes(comm);

  ConditionalOStream pcout(std::cout, rank == 0);

  pcout << "========================================\n"
        << "  deal.II version: " << DEAL_II_PACKAGE_VERSION << "\n"
        << "  Running with " << n_ranks << " MPI process(es)\n"
        << "========================================\n";

  // Quick sanity check
  parallel::distributed::Triangulation<2> triangulation(comm);
  GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
  triangulation.refine_global(2);

  pcout << "Rank 0 setup mesh successfully!\n"
        << "Total global active cells: "
        << triangulation.n_global_active_cells() << std::endl;

  // every rank report its local workload
  std::cout << "  -> Rank " << rank << " owns "
            << triangulation.n_locally_owned_active_cells() << " cells.\n";
}
