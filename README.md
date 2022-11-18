## This project is a work in progress

The main objective of this project is to find a modified form of the Cross-Newell Phase Diffusion equation, which is capable of predicting a wider class of patterns and pattern defects than its current form.

The general form of patterns considered in this work are those of modulated stripe patterns, whose default emergence is rotationally and translationally invariant. Rayleigh-Benard convection is the canonical example of such patterns.
Rayleigh-Benard convection can be observed by trapping a thin layer of fluid between two plates, and applying heat from below. Above some threshold value of the temperature difference, heat transfer transitions from conduction to convection, creating the emergence of rolls in the fluid.
In an infinite domain, for a range of wave numbers (roll width), straight roll solutions are stable. In experiments however, where the fluid is confined to a finite region, we see the stripe patterns emerge with different orientations at different locations, and eventually collide with one another, forming a mosaic of patches of stripes with defects.

Many PDEs have solutions which exhibit such patterns, however we will take as our preferred PDE to be the Swift Hohenberg equation: $u_t = -(1+\Delta)^2u + Ru - u^3$.
The CN equation may be derived from Swift-Hohenberg, by analyzing the local phase of the pattern, and introducing slow and fast time scales. The idea is that
the phase changes slowly over most of the convecting cell, however the phase changes quickly near defects. The CN equation takes the form: $\tau(k^2)\Theta_T + \nabla \cdot \vec{k}B(k^2)+\epsilon^2 \eta \nabla^4 \Theta = 0$.

The CN equation is limited in its ability to predict defects. Thus, we seek a modification of it, perhaps in the form of an additional term, which is capable of capturing more defects.

We propose a joint optimization procedure involving an autoencoder and a sparse regression to discover reduced order models in terms of the phase of the pattern.

<img width="502" alt="convection_roll_examples" src="https://user-images.githubusercontent.com/55065632/197681167-c628b541-6ae4-4bb4-92be-aaed471f6ac2.png">


## References
<a id="1">[1]</a> 
Rudy, Samuel H, Steven L Brunton, Joshua L Proctor, and J Nathan Kutz. "Data-driven Discovery of Partial Differential Equations." Science Advances 3.4 (2017): E1602614. Web.

<a id="2">[2]</a> 
Champion, Kathleen, Bethany Lusch, J Nathan Kutz, and Steven L Brunton. "Data-driven Discovery of Coordinates and Governing Equations." Proceedings of the National Academy of Sciences - PNAS 116.45 (2019): 22445-2451. Web.

<a id="3">[3]</a> 
Cross, M.C., and Alan C. Newell. "Convection Patterns in Large Aspect Ratio Systems." Physica. D 10.3 (1984): 299-328. Web.

<a id="4">[4]</a> 
Newell, A.C., T. Passot, C. Bowman, N. Ercolani, and R. Indik. "Defects Are Weak and Self-dual Solutions of the Cross-Newell Phase Diffusion Equation for Natural Patterns." Physica. D 97.1 (1996): 185-205. Web.


