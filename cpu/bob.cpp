#include "bob.h"

Bob::Bob() {
	m = 0.f;
	l = 0.f;
	x = new float[DIM];
	v = new float[DIM];
	f = new float[DIM];
	for(int i = 0; i < DIM; i++) {
		x[i] = 0.f;
		v[i] = 0.f;
		f[i] = 0.f;
	}
}

Bob::Bob(float mass, float length, float* init_x): m(mass), l(length) {
	x = new float[DIM];
	v = new float[DIM];
	f = new float[DIM];
	for(int i = 0; i < DIM; i++) {
		x[i] = init_x[i];
		v[i] = 0.f;
		f[i] = 0.f;
	}
}

Bob::Bob(const Bob& b): m(b.m), l(b.l) /* copy constructor */
{
	x = new float[DIM];
	v = new float[DIM];
	f = new float[DIM];
	for(int i = 0; i < DIM; i++) {
		x[i] = b.x[i];
		v[i] = b.v[i];
		f[i] = b.f[i];
	}
}

Bob& Bob::operator=(const Bob& b) {
	if(this != &b)
	{
		this->m = b.m;
		this->l = b.l;
		this->x = new float[DIM];
		this->v = new float[DIM];
		this->f = new float[DIM];
		for(int i = 0; i < DIM; i++)
		{
			this->x[i] = b.x[i];
			this->v[i] = b.v[i];
			this->f[i] = b.f[i];
		}
	}
	return *this;
}
