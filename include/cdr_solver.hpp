#pragma once

#include <_ctype.h>
#include <cstdlib>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/numerics/vector_tools.h>
#include <filesystem>
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

  // Number of operator applications timed by the builtin benchmark
  unsigned int n_benchmarks_applications = 0;
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

template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::assemble_rhs() -> void {
  TimerOutput::Scope scope(timer, "assemble rhs");
  Timer wall;

  const MatrixFree<dim, Number> &matrix_free = *system_matrix.get_matrix_free();
  const RightHandSide<dim> right_hand_side(coefficients);
  const ExactSolution<dim> exact_solution;

  solution = 0.0;
  constraints.distribute(solution);
  solution.update_ghost_values();

  system_rhs = 0.0;

  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(matrix_free);
  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell) {
    phi.reinit(cell);
    phi.read_dof_values_plain(solution);
    phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

    system_matrix.do_quadrature_points(
        phi, -1.0, [&](const auto &phi_, const unsigned int q) {
          return evaluate_lanewise<dim, Number>(
              [&](const Point<dim> &p) { return right_hand_side.value(p); },
              phi_.quadrature_point(q));
        });

    phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                          system_rhs);
  }
  system_rhs.compress(VectorOperation::add);

  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_face(
      matrix_free, true);
  const unsigned int first_face = matrix_free.n_inner_face_batches();
  const unsigned int last_face =
      first_face + matrix_free.n_boundary_face_batches();

  for (unsigned int face = first_face; face < last_face; ++face) {
    if (matrix_free.get_boundary_id(face) != neumann_boundary_id) {
      continue;
    }

    phi_face.reinit(face);
    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      const auto normal = phi_face.normal_vector(q);
      const auto point = phi_face.quadrature_point(q);

      VectorizedArray<Number> h;
      for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
        Point<dim> p;
        Tensor<1, dim> n;
        for (unsigned int d = 0; d < dim; ++d) {
          p[d] = point[d][v];
          n[d] = normal[d][v];
        }
        h[v] = exact_solution.gradient(p) * n;
      }

      phi_face.submit_value(coefficients.mu * h, q);
    }
    phi_face.integrate_scatter(EvaluationFlags::values, system_rhs);
  }
  system_rhs.compress(VectorOperation::add);

  solution.zero_out_ghost_values();
  timings.rhs = wall.wall_time();
}

template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::solve() -> SolveResult {
  TimerOutput::Scope scope(timer, "solve");
  Timer wall;

  SolverControl solver_control(5000,
                               parameters.tolerance * system_rhs.l2_norm());

  typename SolverFGMRES<VectorType>::AdditionalData fgmres_data;
  fgmres_data.max_basis_size = 50;
  SolverFGMRES<VectorType> solver(solver_control, fgmres_data);

  VectorType correction;
  system_matrix.initialize_dof_vector(correction);

  const auto solve_with = [&](const auto &preconditioner) {
    solver.solve(system_matrix, correction, system_rhs, preconditioner);
  };

  switch (parameters.preconditioner) {
  case PreconditionerType::None:
    solve_with(PreconditionIdentity());
    break;
  case PreconditionerType::Jacobi:
    solve_with(*system_matrix.get_matrix_diagonal_inverse());
    break;
  case PreconditionerType::Multigrid:
    solve_with(multigrid.get());
    break;
  }

  timings.solve = wall.wall_time();

  VectorType residual;
  system_matrix.initialize_dof_vector(residual);
  system_matrix.vmult(residual, correction);
  residual -= system_rhs;

  solution += correction;
  constraints.distribute(solution);

  return {solver_control.last_step(),
          residual.l2_norm() / system_rhs.l2_norm()};
}

// Times bare operator applications, which isolates the matrix-free cell loop
// from the Krylov vector algebra and the multigrid.
template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::benchmark_operator() const -> void {
  VectorType src, dst;
  system_matrix.initialize_dof_vector(src);
  system_matrix.initialize_dof_vector(dst);
  src = 1.0;

  system_matrix.vmult(dst, src); // warmup the caches
  MPI_Barrier(communicator);

  Timer wall;
  for (unsigned int i = 0; i < parameters.n_benchmarks_applications; ++i) {
    system_matrix.vmult(dst, src);
  }

  const double time = Utilities::MPI::max(wall.wall_time(), communicator) /
                      parameters.n_benchmarks_applications;

  if (is_root()) {
    std::cout << "VMULT ranks=" << Utilities::MPI::n_mpi_processes(communicator)
              << " dofs=" << dof_handler.n_dofs() << " time=" << time
              << " mdofs_per_s=" << dof_handler.n_dofs() / time / 1e6
              << std::endl;
  }
}

template <int dim, int fe_degree>
auto CdrProblem<dim, fe_degree>::compute_errors() const
    -> std::pair<double, double> {
  const QGauss<dim> quadrature(fe_degree + 2);
  Vector<double> cellwise_error(triangulation.n_active_cells());

  const auto global_error = [&](const VectorTools::NormType norm) {
    VectorTools::integrate_difference(mapping, dof_handler, solution,
                                      ExactSolution<dim>(), cellwise_error,
                                      quadrature, norm);
    return VectorTools::compute_global_error(triangulation, cellwise_error,
                                             norm);
  };

  solution.update_ghost_values();
  const std::pair<double, double> errors = {
      global_error(VectorTools::L2_norm),
      global_error(VectorTools::H1_seminorm),
  };
  solution.zero_out_ghost_values();

  return errors;
}
