#ifndef ROD_H_
#define ROD_H_

#include <iostream>
#include <string>
using namespace std;

class Rod
{
	public:
		int n;
		Bob* bobs;
		float t;
		float kd;
		float T_cycle;
		int cycle_interval;
		Rod(Bob* _bobs, float _kd, float _T_cycle, int _cycle_interval): n(sizeof(_bobs) / sizeof(Bob*)), kd(_kd), T_cycle(_T_cycle), cycle_interval(_cycle_interval) {
			bobs = new Bob[n];
			for(int i = 0; i < n; i++)
			{
				bobs[i] = _bobs[i];
			}
		}	       
};

#endif /* ROD_H_ */
