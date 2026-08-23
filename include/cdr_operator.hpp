#pragma once

#include "cdr_problem.hpp"

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

using namespace dealii;

// Cell-loop parallelism comes from MPI (the mesh is distributed) and from the
// SIMD lanes of VectorizedArray. The shared memory task scheduler is left off,
// as in step-37 from the deal.II tutorials.
template <int dim, typename Number>
auto matrix_free_data(const UpdateFlags cell_flags) ->
    typename MatrixFree<dim, Number>::AdditionalData {
  typename MatrixFree<dim, Number>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
  data.mapping_update_flags = cell_flags;
  return data;
}

// Matrix-free convention-diffussion-reaction operator
//
// a(u, v) = (mu grad u, grad v) + (beta . grad u, v) + (gamma u, v).
//
template <int dim, int fe_degree, typename Number>
class Operator : public MatrixFreeOperators::Base<
                     dim, LinearAlgebra::distributed::Vector<Number>> {
public:
  using value_type = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;
  using Base = MatrixFreeOperators::Base<dim, VectorType>;
  using FEEval = FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>;

  auto set_coefficients(const Coefficients<dim, Number> &c) -> void {
    coefficients = c;
  }

  // Core quadrature kernel evaluating local cell integrals for the CDR
  // operator.
  template <typename Source>
  auto do_quadrature_points(FEEval &phi, const Number sign,
                            const Source &source) const -> void {
    for (unsigned int q = 0; q < phi.n_q_points; ++q) {
      const auto value = phi.get_value(q);
      const auto gradient = phi.get_gradient(q);

      VectorizedArray<Number> convection = VectorizedArray<Number>();
      for (unsigned int d = 0; d < dim; ++d) {
        convection += coefficients.beta[d] * gradient[d];
      }

      phi.submit_gradient((sign * coefficients.mu) * gradient, q);
      phi.submit_value(
          sign * (convection + coefficients.gamma * value) + source(phi, q), q);
    }
  }

  // Convenience overload for standard operator evaluation (A*u) without a
  // source term.
  auto do_quadrature_points(FEEval &phi, const Number sign = 1.0) const
      -> void {
    do_quadrature_points(phi, sign, [](const FEEval &, const unsigned int) {
      return VectorizedArray<Number>();
    });
  }

private:
  Coefficients<dim, Number> coefficients;
};
