ctypedef struct node:
    node* leftChild
    node* rightChild
    double location
    unsigned int axis
    unsigned int leaf           # 0 if inner node, else leaf size.
    unsigned int* inds

cdef class KDTree:
    cdef:
        unsigned int nCoords, nPoints, nCoordsData
        double[:,:] data    
        double[:] weights
        unsigned int[:] inds  
        node* root
        node* lastQueryNode
        
        void query(KDTree self, double[:] pointIn, unsigned int* indBest, double* distanceBest)
        unsigned int remove(KDTree self, unsigned int removeInd)
