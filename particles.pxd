cimport numpy


cdef void quicksortByColumn(numpy.ndarray[numpy.double_t, ndim=2] numpyArray, unsigned int sortByColumn)
cdef void _quicksortByColumn(double* array, int left, int right, unsigned int nColumns, unsigned int sortByColumn)