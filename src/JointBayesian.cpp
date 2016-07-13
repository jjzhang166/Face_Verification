#include "JointBayesian.h"

	CJointBayesian::CJointBayesian(const int size, const int threshold)
: m_SIZE(size), m_threshold(threshold)
{
	//----------------------------------------
	// import sys
	// add sys path current directory
	//----------------------------------------
	PyObject *sys = PyImport_ImportModule("sys");
	PyObject *path = PyObject_GetAttrString(sys, "path");
	PyList_Append(path, PyString_FromString("."));
	PyList_Append(path, PyString_FromString("src/"));

	std::cout<< "Python API:import sys path done."<< std::endl;
	// Build the 2D array in C++
	//--------------------------------
	// import the main module (jb_function.py)
	//--------------------------------
	PyObject *pModule = PyImport_ImportModule("jb_function");
	if (!pModule){
		std::cerr << "Python API:module error" << std::endl;
		Py_DECREF(pModule);
		return ;
	}
	PyObject *pClass = PyObject_GetAttrString(pModule, "JointBayesian");
	Py_DECREF(pModule);
	if (!pClass){
		std::cerr << "Python API:Class JointBayesian error" << std::endl;
		Py_DECREF(pClass);
		return ;
	}
	else{
		std::cout<< "Python API:Get Class JointBayesian."<< std::endl;	
	}	
	PyObject *pClassObject = PyObject_CallObject(pClass, NULL);
	Py_DECREF(pClass);
	if (!pClassObject){
		std::cerr << "Python API:Class Object error" << std::endl;
		Py_DECREF(pClassObject);
		return ;
	}
	else{
		std::cout<< "Python API:Initial a class object."<< std::endl;  
	}	

	pFunc = PyObject_GetAttrString(pClassObject, "verification");
	if (!pFunc)
		std::cout<<"Python API:pFunc error"<<std::endl;

	Py_DECREF(sys);
	Py_DECREF(path);  

}

bool CJointBayesian::Verify(float* person1, float* person2)
{    
	import_array1(-1);
	npy_intp* dims = new npy_intp[1];
	dims[0] = m_SIZE;
	PyObject *pdata = PyArray_SimpleNewFromData( 1, dims, NPY_FLOAT, person1);
	PyObject *pdata2 = PyArray_SimpleNewFromData( 1, dims, NPY_FLOAT, person2);
	if (!pdata){
		std::cerr << "Python API:pdata error" << std::endl;
		return 1;
	}
	if (!pdata2){
		std::cerr << "Python API:pdata2 error" << std::endl;
		return 1;
	}
	PyObject *arg = PyTuple_New(2);
	PyTuple_SetItem(arg, 0, pdata);  
	PyTuple_SetItem(arg, 1, pdata2);  

	PyObject *pReturn = PyObject_CallObject(pFunc, arg);
	double ratio;
	if (pReturn!=NULL){
		ratio = PyFloat_AsDouble(pReturn);
		std::cout <<"Python API:returned value "<< ratio << std::endl;
		Py_DECREF(pReturn);
	}

	return ratio > m_threshold;
}

CJointBayesian::~CJointBayesian()
{
	Py_DECREF(pFunc);
}













































