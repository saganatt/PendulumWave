#include <iostream>
#include <string>
using namespace std;

const int DIM = 3;

class Bob
{
	public:
		float m; /* mass */
		float[DIM] x; /* position in DIM-D space */
		float[DIM] v; /* velocity */
		float[DIM] f; /* forces accumulated in each dimension */
		float l; /* pendulum length */
		Bob(float mass, float length, float init_x);
		Bob(const Bob& b);
		Bob& operator=(const Bob& b);
};
