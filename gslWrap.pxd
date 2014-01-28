# Random Number Generation

cdef: 
	void seed(unsigned long x)
	double rand()
	double randn()
	unsigned long randi(unsigned long n)


# Gamma, Beta Functions
cdef:
	double binom(unsigned int k, double p, unsigned int n)
	double betainc(double a, double b, double x)
	double gammainc(double a, double x)
	double betaincinv(double a, double b, double p, double acc=?)
	double gammaincinv(double a, double p, double acc=?)

