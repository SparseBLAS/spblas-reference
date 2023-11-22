

template <typename C>
concept matrix =
    /*
      is an instance of csr_view OR
      is an instance of csc_view OR
      is an instance of dense_view OR
      is an mdspan OR
      is an instance of matrix.
    */
    ;

template <matrix A, matrix B, matrix C>
void multiply(A&& a, B&& b, C&& c) {
  using T = matrix_value_t<A>;
  using I = matrix_index_t<A>;
  using O = matrix_offset_t<A>;
}

template <matrix A, matrix B, matrix C, matrix D>
void multiply(A&& a, B&& b, C&& c, D&& d) {
  using T = matrix_value_t<A>;
  using I = matrix_index_t<A>;
  using O = matrix_offset_t<A>;
}

/*
  There are three types of things one could pass into a Sparse BLAS function:

  1) A view.  This is a non-owning, lightweight object that reference data owned
     by the user in a particular format.  Since it only views data, but does not
  own it, it cannot reallocate its memory to insert more values if it runs out
  of space. See csr_view below.

  2) An opaque matrix object.  This object owns its data and is able to
  reallocate memory to grow.  When it is constructed, it either creates a copy
  of user data or "steals" it via move construction.  The internal format is
  opaque and unknown to the user.  Extracting data will likely require a copy.
     See matrix below.

  3) A known-format matrix object.  This object owns its data and is able to
  reallocate memory to grow.  When it is constructed, it either creates a copy
  of user data or "steals" it via move construction.  Since its format is known,
     users can easily "steal" its data via a move-like primitive.
*/

template <typename T, typename I = std::size_t,
          typename O = /* implementation-defined */>
class csr_view {
public:
  using scalar_type = T;
  using index_type = I;
  using offset_type = O;

  csr_view(I m, I n, O nnz, T* values, I* rowptr, I* colind)
      : m_(m), n_(n), nnz_(nnz), values_(values), rowptr_(rowptr),
        colind_(colind) {}

  T* values_data() {
    return values_;
  }
  I* rowptr_data() {
    return rowptr_;
  }
  I* colind_data() {
    return colind_;
  }

private:
  T* values_;
  I* rowptr_;
  I* colind_;
  I m_, n_, nnz_;

  /* Implementation-defined stuff */
};

template <typename T, typename I>
class matrix {
public:
  matrix(I m, I n);

  void insert(I i, I j, T value);

  template <Matrix M>
  matrix(const Matrix& m) {
    /* Make a copy of matrix `m` */
  }

  template <Matrix M>
    requires(std::is_rvalue_reference_v<M>)
  matrix(Matrix&& m) {
    /* Steal all of the data in `m` */
  }

private:
  /* implementation defined stuff */
};

template <typename T, typename I, typename O>
class csr_matrix {
public:
  csr_matrix(I m, I n);

private:
  std::vector<T> values_;
  std::vector<I> colind_;
  std::vector<O> rowptr_;
  I m_, n_;
  O nnz_;
};
