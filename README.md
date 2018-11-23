# Pendulum Wave (and Newton Craddle)

The simulation was implemented as a mid-term assignment for Graphic Processors in Computational Applications course at Warsaw University of Technology.

The project extends Nvidia's particles sample with two additional settings: Pendulum Wave and Newton Craddle. Sliders, besides standard options, enable to change rope spring and breaking tension of pendulums. 

Moreover, Pendulum Wave simulation can be reset with different value of wave cycle (time difference between moments of flat wave) and number of oscillations of the longest pendulum during one wave cycle. Those two parameters together with gravity affect lengths, amplitudes and proportions of the pendulums.

Newton Craddle implementation is not the ideal one - actually, one would have to consider the whole system of pendulums to process the collisions according to the Law of Conservation of Momentum. That would, however, be impossible in this program, with each thread having information only about single particle.

Enjoy!

<strong>Some pictures:</strong>

![Alt text](img/PendWave_zigzag_front.png?raw=true "Pendulum Wave start")
![Alt text](img/PendWave_zig_zags.png?raw=true "Pendulum Wave")
![Alt text](img/PendWave_interleaves.png?raw=true "Pendulum Wave middle")
![Alt text](img/Newton_init.png?raw=true "Newton craddle start")
