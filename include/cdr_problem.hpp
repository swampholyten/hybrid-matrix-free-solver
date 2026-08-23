#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

using namespace dealii;

constexpr types::boundary_id neumann_boundary_id = 1;

// Coefficients of	-div(mu grad u) + beta . grad u + gamma u = f.
template <int dim, typename Number = double> struct Coefficients {
  Coefficients(const Number mu = 1.0, const Number gamma = 1.0)
      : mu(mu), gamma(gamma) {
    for (unsigned int d = 0; d < dim; ++d) {
      beta[d] = static_cast<Number>(1.0 + 0.5 * d);
    }
  }

  template <typename OtherNumber>
  Coefficients(const Coefficients<dim, OtherNumber> &other)
      : mu(static_cast<Number>(other.mu)),
        gamma(static_cast<Number>(other.gamma)) {
    for (unsigned int d = 0; d < dim; ++d) {
      beta[d] = static_cast<Number>(other.beta[d]);
    }
  }

  double mu;
  double gamma;
  Tensor<1, dim, Number> beta;
};

enum class PreconditionerType { None, Jacobi, Multigrid };

struct Parameters {
  unsigned int dim = 2;

  unsigned int degree = 2;

  unsigned int n_refinements = 5;

  unsigned int n_cycles = 1;

  double mu = 1.0;

  double gamma = 1.0;

  double tolerance = 1e-9;

  PreconditionerType preconditioner = PreconditionerType::None;
};

template <int DIM, int FE_DEGREE> class CdrProblem {
public:
  using Number = double;

  using LevelNumber = float;

  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using LevelVectorType = LinearAlgebra::distributed::Vector<LevelNumber>;

  CdrProblem(const Parameters &parameters);
  void run();

protected:
  void setup();

  void assemble();

  unsigned int solve();

  void setup_multigrid();

  void compute_error();

  void output(unsigned int cycle) const;

  const Parameters parameters;

  const Coefficients<DIM> coefficients;

  const MPI_Comm mpi_communicator;

  const unsigned int mpi_size;

  const unsigned int mpi_rank;

  ConditionalOStream pcout;

  TimerOutput timer;

  ConvergenceTable convergence_table;

  parallel::distributed::Triangulation<DIM> mesh;

  const MappingQ1<DIM> mapping;

  const FE_Q<DIM> fe;

  DoFHandler<DIM> dof_handler;

  AffineConstraints<Number> constraints;
};
