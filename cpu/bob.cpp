#include "bob.h"

Bob& Bob::operator=(const Bob& b) {
	if(this != &b)
	{
		this.m = b.m;
		this.l = b.l
		for(int i = 0; i < DIM; i++)
		{
			this.x[i] = b.x[i];
			this.v[i] = b.v[i];
			this.f[i] = b.f[i];
		}
	}
}

Bob::Bob(float mass, float length, float init_x): m(mass), l(length) {
	for(int i = 0; i < DIM; i++) {
		x[i] = init_x[i];
		v[i] = 0;
		f[i] = 0;
	}
}

Bob::Bob(const Bob& b): m(b.m), l(b.l) /* copy constructor */
{
	for(int i = 0; i < DIM; i++) {
		x[i] = b.x[i];
		v[i] = b.v[i];
		f[i] = b.f[i];
	}
}
