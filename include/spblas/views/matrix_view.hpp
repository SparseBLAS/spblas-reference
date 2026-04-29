#pragma once
#include <spblas/views/view_base.hpp>
#include <type_traits>

namespace spblas {
namespace matrix_view {
namespace diag {
// always treat the diagonal value as unit
class implicit_unit {};

// always treat the diagonal value as zero
class implicit_zero {};

// use the matrix diagonal
class explicit_diag {};

template <typename d>
concept diag =
    std::is_same_v<d, implicit_unit> || std::is_same_v<d, implicit_zero> ||
    std::is_same_v<d, explicit_diag>;
} // namespace diag

namespace uplo {
// full matrix
class full {};

// take the lower triangular part of the matrix
class lower {};

// take the upper triangular part of the matrix
class upper {};

// take the diagonal part of the matrix
class diag {};

template <typename d>
concept uplo = std::is_same_v<d, full> || std::is_same_v<d, lower> ||
               std::is_same_v<d, upper> || std::is_same_v<d, diag>;
} // namespace uplo

namespace __detail {

// help the diagonal transformation
// implicit_* will overwite the previous type
// explicit_diag keeps the type
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

} // namespace __detail

/**
 * This is a view contain all possiblitity how kernel can intepret the matrix
 * with specific order. This will not touch the matrix_opt itself, it will leave
 * the operation to backend to decide what to do.
 *
 * @tparam Conjugate  whether to conjugate the matrix
 * @tparam Transpose  whether to transpose the matrix
 * @tparam Diagonal  how to handle diagonal
 * @tparam UpLo  how to access the part of matrix
 */
template <typename matrix_opt, typename Conjugate = std::false_type,
          typename Transpose = std::false_type,
          diag::diag Diagonal = diag::explicit_diag,
          uplo::uplo UpLo = uplo::full>
class legacy_pattern : public spblas::view_base {
public:
  legacy_pattern(matrix_opt&& t) : obj(t) {}

  auto& base() {
    return obj;
  }

  auto& base() const {
    return obj;
  }

private:
  matrix_opt& obj;
};

template <typename T>
struct is_instantiation_of_legacy_pattern {
  static constexpr bool value = false;
};

template <typename matrix_opt, typename Conjugate, typename Transpose,
          diag::diag Diagonal, uplo::uplo UpLo>
struct is_instantiation_of_legacy_pattern<
    legacy_pattern<matrix_opt, Conjugate, Transpose, Diagonal, UpLo>> {
  static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_legacy_pattern_v =
    is_instantiation_of_legacy_pattern<std::remove_cvref_t<T>>::value;

template <typename matrix_opt>
auto conjugate(matrix_opt&& matrix) {
  return legacy_pattern<matrix_opt, std::true_type>(matrix);
}

template <typename matrix_opt, typename Transpose, typename Diagonal,
          typename UpLo>
auto conjugate(legacy_pattern<matrix_opt, std::true_type, Transpose, Diagonal,
                              UpLo>&& matrix) {
  return legacy_pattern<matrix_opt, std::false_type, Transpose, Diagonal, UpLo>(
      matrix.base());
}

template <typename Transpose, typename Diagonal, typename UpLo,
          typename matrix_opt>
auto conjugate(legacy_pattern<matrix_opt, std::false_type, Transpose, Diagonal,
                              UpLo>&& matrix) {
  return legacy_pattern<matrix_opt, std::true_type, Transpose, Diagonal, UpLo>(
      matrix.base());
}

template <typename matrix_opt>
auto transpose(matrix_opt&& matrix) {
  return legacy_pattern<matrix_opt, std::false_type, std::true_type>(matrix);
}

template <typename Conjugate, typename Diagonal, typename UpLo,
          typename matrix_opt>
  requires(!std::is_same_v<UpLo, uplo::diag>)
auto transpose(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                              UpLo>&& matrix) {
  return legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal, UpLo>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename UpLo,
          typename matrix_opt>
  requires(!std::is_same_v<UpLo, uplo::diag>)
auto transpose(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                              UpLo>&& matrix) {
  return legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal, UpLo>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename Transpose,
          typename matrix_opt>
auto transpose(legacy_pattern<matrix_opt, Conjugate, Transpose, Diagonal,
                              uplo::diag>&& matrix) {
  return legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                        uplo::diag>(matrix.base());
}

template <typename matrix_opt, typename TreatDiag = diag::explicit_diag>
auto diagonal(matrix_opt&& matrix, TreatDiag = {}) {
  return legacy_pattern<matrix_opt, std::false_type, std::false_type, TreatDiag,
                        uplo::diag>(matrix);
}

template <typename Conjugate, typename Transpose, typename Diagonal,
          typename UpLo, typename matrix_opt, typename TreatDiag>
auto diagonal(
    legacy_pattern<matrix_opt, Conjugate, Transpose,
                   typename __detail::decide_diag<Diagonal, TreatDiag>::type,
                   UpLo>&& matrix,
    TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename matrix_opt, typename TreatDiag = diag::explicit_diag>
auto triangle(matrix_opt&& matrix, uplo::lower, TreatDiag = {}) {
  return legacy_pattern<matrix_opt, std::false_type, std::false_type, TreatDiag,
                        uplo::lower>(matrix);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::full>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::lower>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::upper>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::diag>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::full>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::true_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::lower>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::upper>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::true_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::diag>&& matrix,
              uplo::lower, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename matrix_opt, typename TreatDiag = diag::explicit_diag>
auto triangle(matrix_opt&& matrix, uplo::upper, TreatDiag = {}) {
  return legacy_pattern<matrix_opt, std::false_type, std::false_type, TreatDiag,
                        uplo::upper>(matrix);
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::full>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::lower>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::upper>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::upper>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::false_type, Diagonal,
                             uplo::diag>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::full>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::true_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::lower>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::true_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::lower>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::upper>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

template <typename Conjugate, typename Diagonal, typename matrix_opt,
          typename TreatDiag = diag::explicit_diag>
auto triangle(legacy_pattern<matrix_opt, Conjugate, std::true_type, Diagonal,
                             uplo::diag>&& matrix,
              uplo::upper, TreatDiag = {}) {
  return legacy_pattern<
      matrix_opt, Conjugate, std::false_type,
      typename __detail::decide_diag<Diagonal, TreatDiag>::type, uplo::diag>(
      matrix.base());
}

} // namespace matrix_view
} // namespace spblas
