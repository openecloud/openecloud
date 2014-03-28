cdef: 
	double rand() nogil
	double randn() nogil
	unsigned long randi(unsigned long n) nogil

cpdef object seed(unsigned long x)
