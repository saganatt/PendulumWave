#ifndef BOB_H_
#define BOB_H_

#include <iostream>
#include <string>
using namespace std;

const int DIM = 3;

class Bob
{
	public:
		float m; /* mass */
		float* x; /* position in DIM-D space */
		float* v; /* velocity */
		float* f; /* forces accumulated in each dimension */
		float l; /* pendulum length */
		Bob();
		Bob(float mass, float length, float* init_x);
		Bob(const Bob& b);
		Bob& operator=(const Bob& b);
};

#endif /* BOB_H_ */
