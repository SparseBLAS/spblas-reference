#include <gtest/gtest.h>
#include <spblas/views/collection.hpp>
#include <type_traits>

class temp {
public:
    using scalar_type = float;
  using scalar_reference = float&;
  using index_type = int;
  using offset_type = int;
};

using ::testing::StaticAssertTypeEq;
// only for the testing
using namespace spblas::view_v;

TEST(Tag, Conjugate) {
  temp t;
  StaticAssertTypeEq<
      decltype(conjugate(t)),
      general<temp&, conj, none, diag::explicit_diag, uplo::full>>();
  StaticAssertTypeEq<
      decltype(conjugate(conjugate(t))),
      general<temp&, none, none, diag::explicit_diag, uplo::full>>();
  EXPECT_EQ(&(conjugate(t).obj), &t);
  EXPECT_EQ(&(conjugate(conjugate(t)).obj), &t);
}

TEST(Tag, Tranpose) {
  temp t;
  StaticAssertTypeEq<
      decltype(transpose(t)),
      general<temp&, none, trans, diag::explicit_diag, uplo::full>>();
  StaticAssertTypeEq<
      decltype(transpose(transpose(t))),
      general<temp&, none, none, diag::explicit_diag, uplo::full>>();
  EXPECT_EQ(&(transpose(t).obj), &t);
  EXPECT_EQ(&(transpose(transpose(t)).obj), &t);
}

TEST(Tag, Diagonal) {
  temp t;
  StaticAssertTypeEq<
      decltype(diagonal(t)),
      general<temp&, none, none, diag::explicit_diag, uplo::diag>>();
  EXPECT_EQ(&(diagonal(t).obj), &t);
}

TEST(Tag, Lower) {
  temp t;
  StaticAssertTypeEq<
      decltype(triangle(t, uplo::lower())),
      general<temp&, none, none, diag::explicit_diag, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower()), uplo::lower())),
      general<temp&, none, none, diag::explicit_diag, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(t, uplo::lower(), diag::implicit_unit())),
      general<temp&, none, none, diag::implicit_unit, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(t, uplo::lower(), diag::implicit_zero())),
      general<temp&, none, none, diag::implicit_zero, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower(), diag::implicit_unit()),
                        uplo::lower(), diag::explicit_diag())),
      general<temp&, none, none, diag::implicit_unit, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower(), diag::implicit_zero()),
                        uplo::lower(), diag::explicit_diag())),
      general<temp&, none, none, diag::implicit_zero, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower(), diag::implicit_unit()),
                        uplo::lower(), diag::implicit_unit())),
      general<temp&, none, none, diag::implicit_unit, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower(), diag::implicit_zero()),
                        uplo::lower(), diag::implicit_zero())),
      general<temp&, none, none, diag::implicit_zero, uplo::lower>>();
  //   Unit or Zero Diag will overwrite the old one
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower(), diag::implicit_unit()),
                        uplo::lower(), diag::implicit_zero())),
      general<temp&, none, none, diag::implicit_zero, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower(), diag::implicit_zero()),
                        uplo::lower(), diag::implicit_unit())),
      general<temp&, none, none, diag::implicit_unit, uplo::lower>>();

  EXPECT_EQ(&(triangle(t, uplo::lower()).obj), &t);
  EXPECT_EQ(&(triangle(t, uplo::lower(), diag::implicit_zero()).obj), &t);
  EXPECT_EQ(&(triangle(t, uplo::lower(), diag::implicit_unit()).obj), &t);
  EXPECT_EQ(&(triangle(triangle(t, uplo::lower(), diag::implicit_zero()),
                       uplo::lower(), diag::implicit_unit())
                  .obj),
            &t);
  EXPECT_EQ(&(triangle(triangle(t, uplo::lower(), diag::implicit_unit()),
                       uplo::lower(), diag::implicit_zero())
                  .obj),
            &t);
}

TEST(Tag, Upper) {
  temp t;
  StaticAssertTypeEq<
      decltype(triangle(t, uplo::upper())),
      general<temp&, none, none, diag::explicit_diag, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper()), uplo::upper())),
      general<temp&, none, none, diag::explicit_diag, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(t, uplo::upper(), diag::implicit_unit())),
      general<temp&, none, none, diag::implicit_unit, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(t, uplo::upper(), diag::implicit_zero())),
      general<temp&, none, none, diag::implicit_zero, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper(), diag::implicit_unit()),
                        uplo::upper(), diag::explicit_diag())),
      general<temp&, none, none, diag::implicit_unit, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper(), diag::implicit_zero()),
                        uplo::upper(), diag::explicit_diag())),
      general<temp&, none, none, diag::implicit_zero, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper(), diag::implicit_unit()),
                        uplo::upper(), diag::implicit_unit())),
      general<temp&, none, none, diag::implicit_unit, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper(), diag::implicit_zero()),
                        uplo::upper(), diag::implicit_zero())),
      general<temp&, none, none, diag::implicit_zero, uplo::upper>>();
  //   Unit or Zero Diag will overwrite the old one
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper(), diag::implicit_unit()),
                        uplo::upper(), diag::implicit_zero())),
      general<temp&, none, none, diag::implicit_zero, uplo::upper>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper(), diag::implicit_zero()),
                        uplo::upper(), diag::implicit_unit())),
      general<temp&, none, none, diag::implicit_unit, uplo::upper>>();

  EXPECT_EQ(&(triangle(t, uplo::upper()).obj), &t);
  EXPECT_EQ(&(triangle(t, uplo::upper(), diag::implicit_zero()).obj), &t);
  EXPECT_EQ(&(triangle(t, uplo::upper(), diag::implicit_unit()).obj), &t);
  EXPECT_EQ(&(triangle(triangle(t, uplo::upper(), diag::implicit_zero()),
                       uplo::upper(), diag::implicit_unit())
                  .obj),
            &t);
  EXPECT_EQ(&(triangle(triangle(t, uplo::upper(), diag::implicit_unit()),
                       uplo::upper(), diag::implicit_zero())
                  .obj),
            &t);
}

TEST(Tag, MixUpperAndLower) {
  temp t;
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::upper()), uplo::lower())),
      general<temp&, none, none, diag::explicit_diag, uplo::diag>>();
  StaticAssertTypeEq<
      decltype(triangle(triangle(t, uplo::lower()), uplo::upper())),
      general<temp&, none, none, diag::explicit_diag, uplo::diag>>();
}

TEST(Tag, GetTransposeOfUpper) {
  temp t;
  StaticAssertTypeEq<
      decltype(transpose(triangle(t, uplo::upper()))),
      general<temp&, none, trans, diag::explicit_diag, uplo::upper>>();
  // Lower(M^T) = Upper(M)^T
  StaticAssertTypeEq<
      decltype(triangle(transpose(t), uplo::lower())),
      general<temp&, none, trans, diag::explicit_diag, uplo::upper>>();
}

TEST(Tag, GetTransposeOfLower) {
  temp t;
  StaticAssertTypeEq<
      decltype(transpose(triangle(t, uplo::lower()))),
      general<temp&, none, trans, diag::explicit_diag, uplo::lower>>();
  // Upper(M^T) = Lower(M)^T
  StaticAssertTypeEq<
      decltype(triangle(transpose(t), uplo::upper())),
      general<temp&, none, trans, diag::explicit_diag, uplo::lower>>();
}

TEST(Tag, LongChain) {
  temp t;

  StaticAssertTypeEq<
      decltype(triangle(transpose(triangle(t, uplo::lower())), uplo::upper(),
                        diag::implicit_zero())),
      general<temp&, none, trans, diag::implicit_zero, uplo::lower>>();
  StaticAssertTypeEq<
      decltype(conjugate(triangle(transpose(triangle(t, uplo::lower())),
                                  uplo::lower(), diag::implicit_zero()))),
      general<temp&, conj, none, diag::implicit_zero, uplo::diag>>();
  StaticAssertTypeEq<
      decltype(transpose(conjugate(
          triangle(transpose(triangle(t, uplo::lower(), diag::implicit_zero())),
                   uplo::lower(), diag::implicit_unit())))),
      general<temp&, conj, none, diag::implicit_unit, uplo::diag>>();
}
