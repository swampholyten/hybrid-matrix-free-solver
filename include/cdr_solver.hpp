#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>

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

  CycleTimings timings;
};
