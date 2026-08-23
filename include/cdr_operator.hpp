#pragma once

#include "cdr_problem.hpp"

#include <deal.II/matrix_free/matrix_free.h>

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
