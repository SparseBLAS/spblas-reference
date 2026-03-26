#pragma once
#include <spblas/views/view_base.hpp>

namespace spblas {
namespace view_v {
namespace diag {
class implicit_unit {};
class implicit_zero {};
class explicit_diag {};

template <typename d>
concept diag =
    std::is_same_v<d, implicit_unit> || std::is_same_v<d, implicit_zero> ||
    std::is_same_v<d, explicit_diag>;
} // namespace diag

namespace uplo {
class full {};
class lower {};
class upper {};
class diag {};

template <typename d>
concept uplo = std::is_same_v<d, full> || std::is_same_v<d, lower> ||
               std::is_same_v<d, upper> || std::is_same_v<d, diag>;
} // namespace uplo

class none {};
class conj {};
class trans {};

template <typename T, typename U>
struct decide_diag {};

template <typename T>
struct decide_diag<T, diag::implicit_unit> {
  using type = diag::implicit_unit;
};

template <typename T>
struct decide_diag<T, diag::implicit_zero> {
  using type = diag::implicit_zero;
};

template <typename T>
struct decide_diag<T, diag::explicit_diag> {
  using type = T;
};

/**
 * This is a view contain all possiblitity how kernel can intepret the matrix
 * with specific order. This will not touch the matrix_opt itself, it will leave
 * the operation to backend to decide what to do.
 *
 * @tparam Conjugate  we conjugate the matrix
 * @tparam Transpose  transpose the matrix
 * @tparam Diagonal how do we treat the diagonal
 * @tparam UpLo  whether we access the upper or lower part
 * @tparam matrix_opt  matrix handle
 */
template <typename matrix_opt, typename Conjugate = none,
          typename Transpose = none, diag::diag Diagonal = diag::explicit_diag,
          uplo::uplo UpLo = uplo::full>
class general : public spblas::view_base {
public:
using scalar_type = typename std::remove_cvref_t<matrix_opt>::scalar_type;
  using scalar_reference = typename std::remove_cvref_t<matrix_opt>::scalar_reference;
  using index_type = typename std::remove_cvref_t<matrix_opt>::index_type;
  using offset_type = typename std::remove_cvref_t<matrix_opt>::offset_type;
  using uplo = UpLo;
  general(matrix_opt&& t) : obj(t) {}
  auto& base() {
    return obj;
  }

  auto& base() const {
    return obj;
  }

  matrix_opt& obj;
};


template <typename T>
struct is_instantiation_of_general {
  static constexpr bool value = false;
};

template <typename matrix_opt, typename Conjugate, typename Transpose, diag::diag Diagonal,
          uplo::uplo UpLo>
struct is_instantiation_of_general<general<matrix_opt, Conjugate, Transpose, Diagonal, UpLo>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_general_v =
    is_instantiation_of_general<std::remove_cvref_t<T>>::value;

template <typename matrix_opt>
auto conjugate(matrix_opt&& matrix) {
  return general<matrix_opt, conj>(matrix);
}

template <typename matrix_opt, typename Transpose, typename Diagonal,
          typename UpLo>
auto conjugate(general<matrix_opt, conj, Transpose, Diagonal, UpLo>&& matrix) {
  return general<matrix_opt, none, Transpose, Diagonal, UpLo>(matrix.obj);
}

template <typename Transpose, typename Diagonal, typename UpLo,
          typename matrix_opt>
auto conjugate(general<matrix_opt, none, Transpose, Diagonal, UpLo>&& matrix) {
  return general<matrix_opt, conj, Transpose, Diagonal, UpLo>(matrix.obj);
}

template <typename matrix_opt>
auto transpose(matrix_opt&& matrix) {
  return general<matrix_opt, none, trans>(matrix);
}

template <typename Conjugate, typename Diagonal, typename UpLo,
          typename matrix_opt>
  requires(!std::is_same_v<UpLo, uplo::diag>)
auto transpose(general<matrix_opt, Conjugate, none, Diagonal, UpLo>&& matrix) {
  return general<matrix_opt, Conjugate, trans, Diagonal, UpLo>(matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename UpLo,
          typename matrix_opt>
  requires(!std::is_same_v<UpLo, uplo::diag>)
auto transpose(general<matrix_opt, Conjugate, trans, Diagonal, UpLo>&& matrix) {
  return general<matrix_opt, Conjugate, none, Diagonal, UpLo>(matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename Transpose,
          typename matrix_opt>
auto transpose(
    general<matrix_opt, Conjugate, Transpose, Diagonal, uplo::diag>&& matrix) {
  return general<matrix_opt, Conjugate, none, Diagonal, uplo::diag>(matrix.obj);
}

template <typename matrix_opt, typename TreatDiag = diag::explicit_diag>
auto diagonal(matrix_opt&& matrix, TreatDiag = {}) {
  return general<matrix_opt, none, none, TreatDiag, uplo::diag>(matrix);
}

template <typename Conjugate, typename Transpose, typename Diagonal,
          typename UpLo, typename matrix_opt, typename TreatDiag>
auto diagonal(
    general<matrix_opt, Conjugate, Transpose,
            typename decide_diag<Diagonal, TreatDiag>::type, UpLo>&& matrix,
    TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename matrix_opt, typename TreatDiag = diag::explicit_diag>
auto triangle(matrix_opt&& matrix, uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, none, none, TreatDiag, uplo::lower>(matrix);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::full>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::lower>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::upper>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::diag>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::full>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, trans,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::lower>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::upper>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, trans,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::diag>&& matrix,
    uplo::lower, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename matrix_opt, typename TreatDiag = diag::explicit_diag>
auto triangle(matrix_opt&& matrix, uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, none, none, TreatDiag, uplo::upper>(matrix);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::full>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::lower>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::upper>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, none, Diagonal, uplo::diag>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::full>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, trans,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::lower>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, trans,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::upper>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(
    general<matrix_opt, Conjugate, trans, Diagonal, uplo::diag>&& matrix,
    uplo::upper, TreatDiag = {}) {
  return general<matrix_opt, Conjugate, none,
                 typename decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.obj);
}

} // namespace view
} // namespace spblas
