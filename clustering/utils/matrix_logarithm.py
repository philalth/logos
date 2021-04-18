import numpy


def matrix_logarithm(a):
    """
    Computes the logarithm of a symmetric matrix. The logarithm is only
    defined for square matrices. It is calculated as follows:

    log(A) = V * log(A') * V^(T)

    where A' is the diagonal matrix with the eigenvalues of A and V consists
    of the corresponding eigenvectors.

    Parameters
    ----------
    a : (n,n) array
        real symmetric input matrix

    Returns
    -------
    natural logarithm of a
    """
    eigenvalues, eigenvector_matrix = numpy.linalg.eigh(a)
    eigenvalue_matrix = numpy.diag(eigenvalues)
    eigenvalue_matrix_logarithm = _diagonal_matrix_logarithm(eigenvalue_matrix)

    return eigenvector_matrix @ eigenvalue_matrix_logarithm @ numpy.transpose(eigenvector_matrix)


def _diagonal_matrix_logarithm(a):
    """
    Computes the natural logarithm of a diagonal matrix by calculating
    the elementwise logarithm for each diagonal entry. If one of the
    entries is less than zero the result will be a complex number.

    Parameters
    ----------
    a: (n,n) array
        diagonal input matrix

    Returns
    -------
    natural logarithm of a
    """
    assert _is_diagonal(a)

    log_matrix = numpy.array(a, complex)
    for i in range(a.shape[0]):
        log_matrix[i][i] = numpy.log(a[i][i] + 0j)
    return log_matrix


def _is_diagonal(a):
    """
    Returns true if and only if a matrix is diagonal.

    Parameters
    ----------
    a: (n,n) array
        matrix to be checked

    Returns
    -------
    Whether the matrix is diagonal
    """
    return numpy.count_nonzero(a - numpy.diag(numpy.diagonal(a))) == 0
