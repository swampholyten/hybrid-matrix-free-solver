#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <memory>

#include "cdr_operator.hpp"
#include "cdr_problem.hpp"
#include "mg_preconditioner.hpp"

using namespace dealii;

enum class PreconditionerType { None, Jacobi, Multigrid };

struct Parameters {
  unsigned int dim = 2;
  unsigned int degree = 2;
  unsigned int n_refinements = 5;
  unsigned int n_cycles = 1;
  double mu = 1.0;
  double gamma = 1.0;
  double tolerance = 1e-9;
  PreconditionerType preconditioner = PreconditionerType::Multigrid;
};

// Wall times of a single refinement cycle, for the scaling study.
struct CycleTimings {
  double setup = 0.0;
  double rhs = 0.0;
  double solve = 0.0;
};

struct SolveResult {
  unsigned int n_iterations = 0;
  double relative_residual = 0.0;
};

// Distributed matrix-free solver.
template <int dim, int fe_degree> class CdrProblem {
public:
  using Number = double;
  using LevelNumber = float; // mixed precision on the multigrid levels (todo).
  using SystemMatrixType = Operator<dim, fe_degree, Number>;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using Multigrid = MultigridPreconditioner<dim, fe_degree, LevelNumber>;

  CdrProblem(const Parameters &parameters);

  auto run() -> void;

private:
  auto setup_system() -> void;
  auto assemble_rhs() -> void;
  auto solve() -> SolveResult;
  auto benchmark_operator() const -> void;
  auto compute_errors() const -> std::pair<double, double>;
  auto output_results(const unsigned int cycle) const -> void;

  auto dirichlet_boundary_ids() const -> std::set<types::boundary_id>;
  auto is_root() const -> bool;
  auto run_cycle(const unsigned int cycle) -> void;
  auto print_header() const -> void;
  auto print_convergence_table() -> void;

  const Parameters parameters;
  const Coefficients<dim> coefficients;
  const MPI_Comm communicator;
  ConditionalOStream pcout;
  TimerOutput timer;
  ConvergenceTable convergence_table;

  parallel::distributed::Triangulation<dim> triangulation;
  const MappingQ1<dim> mapping;
  const FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<Number> constraints;
  SystemMatrixType system_matrix;
  VectorType solution;
  VectorType system_rhs;

  Multigrid multigrid;
  CycleTimings timings;
};

template <int dim, int fe_degree>
CdrProblem<dim, fe_degree>::CdrProblem(const Parameters &parameters)
    : parameters(parameters), coefficients(parameters.mu, parameters.gamma),
      communicator(MPI_COMM_WORLD),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      timer(communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
      triangulation(communicator,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                        dim>::construct_multigrid_hierarchy),
      fe(fe_degree), dof_handler(triangulation) {}

template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::is_root() const -> bool {
  return Utilities::MPI::this_mpi_process(communicator) == 0;
}

template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::dirichlet_boundary_ids() const
    -> std::set<types::boundary_id> {
  std::set<types::boundary_id> ids;
  for (const types::boundary_id id : triangulation.get_boundary_ids()) {
    if (id != neumann_boundary_id) {
      ids.insert(id);
    }
  }
  return ids;
}

template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::setup_system() -> void {
  TimerOutput::Scope scope(timer, "setup");
  Timer wall;

  const bool use_multigrid =
      parameters.preconditioner == PreconditionerType::Multigrid;

  dof_handler.distribute_dofs(fe);
  if (use_multigrid) {
    dof_handler.distribute_mg_dofs();
  }

  constraints.clear();
  constraints.reinit(dof_handler.locally_owned_dofs(),
                     DoFTools::extract_locally_relevant_dofs(dof_handler));

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  for (const types::boundary_id id : dirichlet_boundary_ids()) {
    VectorTools::interpolate_boundary_values(mapping, dof_handler, id,
                                             ExactSolution<dim>(), constraints);
  }
  constraints.close();

  auto additional_data = matrix_free_data<dim, Number>(
      update_gradients | update_JxW_values | update_quadrature_points);

  additional_data.mapping_update_flags_boundary_faces =
      update_values | update_JxW_values | update_quadrature_points |
      update_normal_vectors;

  auto matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  matrix_free->reinit(mapping, dof_handler, constraints,
                      QGauss<1>(fe_degree + 1), additional_data);

  system_matrix.initialize(matrix_free);
  system_matrix.set_coefficients(coefficients);
  system_matrix.initialize_dof_vector(solution);
  system_matrix.initialize_dof_vector(system_rhs);

  if (parameters.preconditioner == PreconditionerType::Jacobi) {
    system_matrix.compute_diagonal();
  }

  if (use_multigrid) {
    multigrid.build(dof_handler, mapping, coefficients,
                    dirichlet_boundary_ids());
  }

  timings.setup = wall.wall_time();
}
