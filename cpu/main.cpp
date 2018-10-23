#include <stdio.h>
#include "bob.cpp"
#include "rod.h"

int main() {
	float m = 1.0f;
	float l = 3.5f;
	float* x = new float[3];
	x[0] = 3.0f;
	x[1] = 1.2f;
	x[2] = 4.5f;
	Bob b(m, l, x);
	cout << b.m << " and length: " << b.l << endl;
}
