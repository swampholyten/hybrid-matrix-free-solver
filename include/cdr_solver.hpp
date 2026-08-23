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
