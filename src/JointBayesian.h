#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
// Python has to be included first!
#include "numpy/arrayobject.h"
#include <stdlib.h> 
#include <stdio.h>
#include <iostream>

class CJointBayesian
{
private:   
   PyObject *pFunc;
   const int m_SIZE;
   const int m_threshold;
public:  
   CJointBayesian(const int size, const int threshold);
   ~CJointBayesian();
   bool Verify(float* person1, float* person2);
};
