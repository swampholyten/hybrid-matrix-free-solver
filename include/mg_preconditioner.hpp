#pragma once

#include "cdr_problem.hpp"
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

using namespace dealii;

template <int dim, int fe_degree, typename LevelNumber>
class MultigridPreconditioner {
public:
  auto build(const DoFHandler<dim> &, const MappingQ1<dim> &,
             const Coefficients<dim> &, const std::set<types::boundary_id> &)
      -> void {
    // TODO
  }

  auto get() const -> PreconditionIdentity {
    // TODO
    return PreconditionIdentity();
  }
};
