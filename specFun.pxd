cdef:
	double binom(unsigned int k, double p, unsigned int n) nogil
	double betainc(double a, double b, double x) nogil
	double gammainc(double a, double x) nogil
	double betaincinv(double a, double b, double p, double acc=?) nogil
	double gammaincinv(double a, double p, double acc=?) nogil
