#ifndef SURFACEH
#define SURFACEH
#include "Hitable.cuh"

class Surface : public Hitable
{
public:
	Material* matPtr;
};

#endif