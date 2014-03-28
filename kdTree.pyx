#cython: nonecheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy
cimport numpy
cimport cython
from constants cimport *
from libc.stdlib cimport malloc, free

cdef extern from "math.h":
    double fabs(double x) nogil
    double sqrt(double x) nogil

# Inspired by http://en.wikipedia.com/kdtree .
# Tailored to the particleManagement requirements, so if
# used in other context probably has to be modified.
cdef class KDTree:
         
    def __init__(KDTree self, double[:,:] data, double[:] weights = numpy.empty(0, dtype=numpy.double), 
                 unsigned int leafSize = 10):
        cdef:
            unsigned int ii
            unsigned int[:,::1] inds       
            double[::1] maxes
            double[::1] mins    
        
        self.data = data
        self.nPoints = data.shape[0]
        self.nCoords = data.shape[1]  
        self.nCoordsData = data.strides[0]/sizeof(double)
        mins = numpy.amin(self.data, axis=0)     
        maxes = numpy.amax(self.data, axis=0) 
        self.inds = numpy.arange(self.nPoints,dtype=numpy.uintc)  
        if weights.shape[0]==0:
            self.weights = numpy.ones(self.nCoords)
        else:
            self.weights = weights
        for ii in range(self.nCoords):
            mins[ii]*= self.weights[ii]
            maxes[ii]*= self.weights[ii]
        self.root = _build(&self.data[0,0], &self.inds[0], self.nPoints, self.nCoords, 
                           &mins[0], &maxes[0], self.nCoordsData, &self.weights[0], leafSize)
        self.lastQueryNode = self.root
     
    # Query for nearest neighbor.         
    cdef void query(KDTree self, double[:] pointIn, unsigned int* indBest, double* distanceBest):
        cdef:
            unsigned int ii    
            double[:] point = pointIn.copy()
        distanceBest[0] = doubleMaxVal
        indBest[0] = 0
        for ii in range(self.nCoords):
            point[ii]*= self.weights[ii]
        _query(&self.data[0,0], self.root, &point[0], distanceBest, indBest, self.nCoords, 
               &self.lastQueryNode, self.nCoordsData, &self.weights[0])
    
    # Remove point from kdTree.   
    cdef unsigned int remove(KDTree self, unsigned int removeInd):
        cdef:
            unsigned int success
        # First try if point is in node from last query.
        success = _remove(self.lastQueryNode, removeInd, &self.data[0,0], self.nCoords, 
                          self.nCoordsData, &self.weights[0], &self.lastQueryNode)
        # Either point is in lastQueryNode but whole node has 
        # to be removed OR point isn't in lastQueryNode.
        # Start from root to remove point.
        if success != 1:
            success = _remove(self.root, removeInd, &self.data[0,0], self.nCoords, 
                              self.nCoordsData, &self.weights[0], &self.lastQueryNode)
            # Some error. Point was not found in tree.
            if success==1:
                self.nPoints-= 1
                return 1
        else:
            return 1
        return 0
              
    def __dealloc__(KDTree self):
        _freeNodes(self.root)


# Removes a point with given index from kdTree.
# Does not rebalance the tree or anything, so it might
# be necessary to rebuild the tree in some cases.
# Returns 0 if nothing was removed.
# Returns 1 if a node got successfully removed.
# Returns 2 if a the node has to be set to NULL on the parent level.
cdef unsigned int _remove(node* currentNode, unsigned int removeInd, double* data, unsigned int nCoords, 
                          unsigned int nCoordsData, double* weights, node** lastQueryNode) nogil:
    cdef:
        unsigned int ii, jj, success
        unsigned int* inds
        node* tempNode
    if currentNode is not NULL:
        if currentNode.leaf > 1:
            inds = currentNode.inds
            for ii in range(currentNode.leaf):
                if inds[ii] == removeInd:
                    inds[ii] = inds[currentNode.leaf-1]
                    inds[currentNode.leaf-1] = removeInd
                    currentNode.leaf-= 1 
                    return 1
            return 0
        elif currentNode.leaf == 1:
            if currentNode.inds[0] == removeInd:
                return 2
            else:
                return 0
        elif currentNode.leaf == 0:
            if data[removeInd*nCoordsData+currentNode.axis]*weights[currentNode.axis] < currentNode.location:
                success = _remove(currentNode.leftChild, removeInd, data, nCoords, nCoordsData, weights, lastQueryNode)
                if success == 2:
                    if lastQueryNode[0] == currentNode.leftChild:
                        lastQueryNode[0] = currentNode
                    free(currentNode.leftChild)
                    currentNode.leftChild = <node*> NULL
                    return 1
                else:
                    return success
            else:
                success = _remove(currentNode.rightChild, removeInd, data, nCoords, nCoordsData, weights, lastQueryNode)
                if success == 2:
                    if lastQueryNode[0] == currentNode.rightChild:
                        lastQueryNode[0] = currentNode
                    free(currentNode.rightChild)
                    currentNode.rightChild = <node*> NULL
                    return 1
                else:
                    return success
    else:
        return 0

       
        
cdef void _freeNodes(node* currentNode):
    if currentNode is not NULL:
        _freeNodes(currentNode.leftChild)
        _freeNodes(currentNode.rightChild)  
        free(currentNode) 

# Heavily inspired by the scipy cKDTree implementation.
# Uses sliding midpoint rule. Other build rules were slower in building and queries in my tests.
# Tests included modified sliding midpoint, real median (with pre-sorted arrays), ...
cdef node* _build(double* data, unsigned int* inds, unsigned int nPoints, unsigned int nCoords, 
                  double* mins, double* maxes, unsigned int nCoordsData, double* weights, unsigned int leafSize):
    
    cdef:
        unsigned int axis, ii, jj, tempInd
        int kk, ll
        double location, temp
        node* currentNode
        double* mids
    if nPoints<1:
        currentNode = <node*> NULL     
    # leaf size should be something like 10-50. 
    # Here I chose a small leaf size to have faster queries with
    # the cost of slower building. This is optimized for the openECLOUD typical use-case.
    elif nPoints<=leafSize:                   
        currentNode = <node*> malloc(sizeof(node))
        currentNode.inds = inds
        currentNode.leaf = nPoints    
        currentNode.leftChild = <node*> NULL
        currentNode.rightChild = <node*> NULL
    else:
        currentNode = <node*> malloc(sizeof(node))
        
        temp = 0.
        axis = nCoords
        for ii in range(nCoords):
            if temp<=maxes[ii]-mins[ii]:
                temp = maxes[ii]-mins[ii]
                axis = ii
        if temp<=sqrt(maxes[axis]**2+mins[axis]**2)*1.e4*machineEpsilon:
            # All points are identical.
            currentNode.inds = inds
            currentNode.leaf = nPoints    
            currentNode.leftChild = <node*> NULL
            currentNode.rightChild = <node*> NULL
            return currentNode
        currentNode.axis = axis            
        location = 0.5*(maxes[axis]+mins[axis])
        currentNode.leaf = 0    

        kk = 0
        ll = nPoints-1
        while kk<=ll:
            if data[inds[kk]*nCoordsData+axis]*weights[axis]<location:
                kk += 1
            elif data[inds[ll]*nCoordsData+axis]*weights[axis]>=location:
                ll -= 1
            else:
                tempInd = inds[ll]
                inds[ll] = inds[kk]
                inds[kk] = tempInd
                ll -= 1
                kk += 1   
        # slide midpoint if necessary 
        if kk==0:
            location = maxes[axis]
            for ii in range(nPoints):
                if data[inds[ii]*nCoordsData+axis]*weights[axis]<location:
                    location = data[inds[ii]*nCoordsData+axis]*weights[axis]
            location-= 100.*machineEpsilon*fabs(location)    # Some safety margin.
        elif ll==(nPoints-1):           
            location = mins[axis]       
            for ii in range(nPoints):
                if data[inds[ii]*nCoordsData+axis]*weights[axis]>location:
                    location = data[inds[ii]*nCoordsData+axis]*weights[axis]
            location+= 100.*machineEpsilon*fabs(location)    # Some safety margin.
        currentNode.location = location                              
        mids = <double*> malloc(sizeof(double)*nCoords)
        for ii in range(nCoords):
            mids[ii] = maxes[ii]
        mids[axis] = location             
        currentNode.leftChild = _build(data, inds, kk, nCoords, mins, mids, nCoordsData, weights, leafSize)                
        for ii in range(nCoords):
            mids[ii] = mins[ii]
        mids[axis] = location       
        currentNode.rightChild = _build(data, &inds[kk], nPoints-kk, nCoords, mids, maxes, nCoordsData, weights, leafSize)
        free(mids)
    return currentNode
  


cdef void _query(double* data, node* currentNode, double* point, double* distanceBest, unsigned int* indBest, 
                 unsigned int nCoords, node** foundNode, unsigned int nCoordsData, double* weights) nogil:
    cdef:
        unsigned int ii, jj
        double testDistance
        unsigned int* inds
    
    if currentNode is NULL:
        pass
    elif currentNode.leaf != 0:
        inds = currentNode.inds
        for ii in range(currentNode.leaf):
            testDistance = 0.
            for jj in range(nCoords):
                testDistance += (data[inds[ii]*nCoordsData+jj]*weights[jj] - point[jj])**2
            if testDistance < distanceBest[0]:
                indBest[0] = inds[ii]
                distanceBest[0] = testDistance
                foundNode[0] = currentNode
    else:
        if point[currentNode.axis] < currentNode.location:
            _query(data, currentNode.leftChild, point, distanceBest, indBest, nCoords, foundNode, nCoordsData, weights)
        else:
            _query(data, currentNode.rightChild, point, distanceBest, indBest, nCoords, foundNode, nCoordsData, weights)
        if (point[currentNode.axis] - currentNode.location)**2 < distanceBest[0]:          
            if point[currentNode.axis] < currentNode.location:
                _query(data, currentNode.rightChild, point, distanceBest, indBest, nCoords, foundNode, nCoordsData, weights)
            else:
                _query(data, currentNode.leftChild, point, distanceBest, indBest, nCoords, foundNode, nCoordsData, weights)      
    


