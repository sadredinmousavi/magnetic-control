Below is a complete, cleaned-up report you can use as your simulation reference.

---

# Report: Magnetic Microrobot Swarm Force Model for 2D Simulation

## 1. Goal

This report derives and summarizes the force model used to simulate planar magnetic microrobot swarms, consistent with Dong and Sitti's collective magnetic microrobot model and the standard dipole-force formulation in Abbott et al. The model includes:

1. External magnetic forces from source magnets
2. Inter-robot magnetic dipole-dipole forces
3. Capillary interaction forces at the air-water interface
4. Fluid damping / drag
5. Full second-order ODE dynamics for simulation

Dong models the swarm as a 2D mass-spring-damper-like interacting particle system at the air-water interface, where each microrobot experiences external magnetic force, inter-robot magnetic force, capillary force, and damping.

---

## 2. General 3D Magnetic Dipole Field

A magnetic dipole with moment $\mathbf m_s$ located at the origin generates the magnetic field:

$$
\mathbf B(\mathbf r)
=
\frac{\mu_0}{4\pi}
\left(
\frac{3\mathbf r(\mathbf m_s\cdot \mathbf r)}{|\mathbf r|^5}
-
\frac{\mathbf m_s}{|\mathbf r|^3}
\right)
$$

where:

$$
\mu_0 = 4\pi\times 10^{-7}\ \text{T}\cdot\text{m/A}
$$

$$
\mathbf r = \mathbf p-\mathbf p_s
$$

Abbott gives this same dipole field model and notes that it is accurate when the observation point is sufficiently far from the magnet compared with the magnet size.

---

## 3. Magnetic Potential Energy and Force

The magnetic potential energy of a dipole $\mathbf m_r$ in a magnetic field $\mathbf B$ is:

$$
U = -\mathbf m_r\cdot \mathbf B
$$

Therefore the force is:

$$
\mathbf F
=
-\nabla U
=
\nabla(\mathbf m_r\cdot \mathbf B)
$$

Abbott uses this convention: the magnetic object moves to increase $\mathbf b\cdot \mathbf m$, equivalently decreasing $-\mathbf b\cdot \mathbf m$.

Important correction: use $U=-\mathbf m\cdot\mathbf B$, not $U=\mathbf m\cdot\mathbf B$.

---

## 4. Simplified 2D Vertical-Dipole Model

### 4.1 Assumptions

The simplified model assumes:

$$
\mathbf r=[x,y,0]^T
$$

$$
\mathbf m_s=[0,0,m_s]^T
$$

$$
\mathbf m_r=[0,0,m_r]^T
$$

That means all robot positions lie in the $xy$-plane and all magnetic moments are vertical.

---

### 4.2 Simplified Magnetic Field

Because:

$$
\mathbf m_s\cdot \mathbf r = 0
$$

the dipole field becomes:

$$
\mathbf B(\mathbf r)
=
-\frac{\mu_0}{4\pi}
\frac{\mathbf m_s}{|\mathbf r|^3}
$$

So the field at the robot is purely vertical:

$$
B_z
=
-\frac{\mu_0 m_s}{4\pi|\mathbf r|^3}
$$

---

### 4.3 Simplified Potential Energy

Using:

$$
U=-\mathbf m_r\cdot \mathbf B
$$

we get:

$$
U
=
-\left([0,0,m_r]\cdot
\left[
0,0,
-\frac{\mu_0m_s}{4\pi r^3}
\right]
\right)
$$

$$
U
=
\frac{\mu_0m_sm_r}{4\pi r^3}
$$

where:

$$
r=|\mathbf r|
$$

---

### 4.4 Simplified Force

The force is:

$$
\mathbf F=-\nabla U
$$

Let:

$$
K=\frac{\mu_0m_sm_r}{4\pi}
$$

Then:

$$
U=Kr^{-3}
$$

$$
\nabla U
=
-3K\frac{\mathbf r}{r^5}
$$

Therefore:

$$
\mathbf F
=
3K\frac{\mathbf r}{r^5}
$$

So the simplified 2D magnetic force is:

$$
\boxed{
\mathbf F
=
\frac{3\mu_0m_sm_r}{4\pi}
\frac{\mathbf r}{|\mathbf r|^5}
}
$$

This force is repulsive when both dipoles are parallel and vertical.

---

## 5. External Magnetic Force from Controlled Source Magnets

For a source magnet $i$, define:

$$
\mathbf r_{ij}=\mathbf p_j-\mathbf p_i
$$

where $\mathbf p_i$ is the source magnet position and $\mathbf p_j$ is the robot position.

If the vertical component of the source moment is controlled by:

$$
u_i=\cos(\theta_i)
$$

then:

$$
m_{s,z}=m_su_i
$$

The external force on robot $j$ from source magnet $i$ becomes:

$$
\mathbf F^{ext}_{ij}
=
C_{F,i}u_i
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^5}
$$

where:

$$
C_{F,i}
=
\frac{3\mu_0m_sm_r}{4\pi}
$$

For multiple source magnets:

$$
\boxed{
\mathbf F^{ext}_{j}
=
\sum_i
C_{F,i}u_i
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^5}
}
$$

This is consistent with the simplified vertical-dipole interaction model.

---

## 6. General 3D Dipole-Dipole Force

For full 3D simulation, Abbott gives the force between two magnetic dipoles as:

$$
\mathbf f
=
\frac{3\mu_0}{4\pi r^4}
\left[
(\hat{\mathbf r}\cdot \mathbf m_j)\mathbf m_i
+
(\hat{\mathbf r}\cdot \mathbf m_i)\mathbf m_j
+
\left(
\mathbf m_i\cdot \mathbf m_j
-
5(\hat{\mathbf r}\cdot \mathbf m_i)(\hat{\mathbf r}\cdot \mathbf m_j)
\right)\hat{\mathbf r}
\right]
$$

where:

$$
\hat{\mathbf r}=\frac{\mathbf r}{r}
$$

This is the general force law for arbitrary dipole orientations.

Under the 2D vertical-dipole assumptions:

$$
\mathbf m_i=\mathbf m_j=m\hat{\mathbf z}
$$

$$
\hat{\mathbf r}\cdot \hat{\mathbf z}=0
$$

the Abbott formula reduces to:

$$
\mathbf f
=
\frac{3\mu_0m^2}{4\pi r^4}\hat{\mathbf r}
$$

Since:

$$
\hat{\mathbf r}=\frac{\mathbf r}{r}
$$

we get:

$$
\boxed{
\mathbf f
=
\frac{3\mu_0m^2}{4\pi}
\frac{\mathbf r}{r^5}
}
$$

So the simplified force used by Dong is the 2D vertical-dipole special case of Abbott's general dipole-dipole force.

---

## 7. Inter-Robot Magnetic Force

For robot $i$ interacting with robot $j$, define:

$$
\mathbf r_{ij}=\mathbf p_i-\mathbf p_j
$$

This vector points from robot $j$ to robot $i$. Therefore, a positive force along $\mathbf r_{ij}$ pushes robot $i$ away from robot $j$.

The inter-robot magnetic force is:

$$
\boxed{
\mathbf F^{mag,inter}_{ij}
=
\frac{3\mu_0m_im_j}{4\pi}
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^5}
}
$$

For identical disk-shaped robots:

$$
m_i=m_j=m
$$

and:

$$
m=M\pi R^2h
$$

where:

$$
M=\text{magnetization}
$$

$$
R=\text{robot radius}
$$

$$
h=\text{robot thickness}
$$

So:

$$
\boxed{
\mathbf F^{mag,inter}_{ij}
=
\frac{3\mu_0}{4\pi}
\frac{|M\pi R^2h|^2}{|\mathbf r_{ij}|^5}
\mathbf r_{ij}
}
$$

This is the inter-robot magnetic force used by Dong in the pairwise interaction section.

---

## 8. Capillary Force at the Air-Water Interface

Dong includes capillary interaction between robots caused by deformation of the air-water interface.

The capillary force from robot $j$ on robot $i$ is:

$$
\boxed{
\mathbf F^{capillary}_{ij}
=
2\pi\gamma R^2\sin^2(C)
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^2}
}
$$

where:

$$
\gamma \approx 72\ \text{mN/m}
$$

$$
R=\text{robot radius}
$$

$$
C=\text{meniscus slope angle}
$$

$$
\mathbf r_{ij}=\mathbf p_i-\mathbf p_j
$$

Dong notes that the capillary force is usually less than $10\%$ of the magnetic repulsive force at equilibrium.

For scaling, Dong also relates the meniscus slope to the vertical force:

$$
\sin(C)=\frac{F_z}{2\pi\gamma R}
$$

where $F_z$ is mainly the vertical magnetic pulling force.

---

## 9. Fluid Drag / Damping

Dong writes the dynamics with a damping matrix $D$:

$$
D\dot{\mathbf r}_{i,xy}
$$

For simulation, it is clearer to use viscous drag:

$$
\boxed{
\mathbf F^{drag}_i=-c\dot{\mathbf p}_i
}
$$

or, in matrix form:

$$
\boxed{
\mathbf F^{drag}_i=-C_d\dot{\mathbf p}_i
}
$$

where $c>0$ or $C_d$ is positive definite.

Dong's equation uses $D\dot{\mathbf r}$, so in code you can set:

$$
D=-cI
$$

to make the damping physically dissipative.

---

## 10. Complete Force Model for Robot $i$

The total force on robot $i$ is:

$$
\boxed{
\mathbf F_i
=
\mathbf F^{ext}_i
+
\sum_{j\ne i}
\left(
\mathbf F^{mag,inter}_{ij}
+
\mathbf F^{capillary}_{ij}
\right)
+
\mathbf F^{drag}_i
}
$$

Substituting each term:

$$
\boxed{
\mathbf F_i
=
\sum_k
C_{F,k}u_k
\frac{\mathbf r_{ik}^{ext}}{|\mathbf r_{ik}^{ext}|^5}
+
\sum_{j\ne i}
\left[
\frac{3\mu_0m_im_j}{4\pi}
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^5}
+
2\pi\gamma R^2\sin^2(C)
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^2}
\right]
-
c\dot{\mathbf p}_i
}
$$

where:

$$
\mathbf r_{ij}=\mathbf p_i-\mathbf p_j
$$

and:

$$
\mathbf r_{ik}^{ext}=\mathbf p_i-\mathbf p_k^{ext}
$$

---

## 11. Complete System Dynamics

Dong's 2D equation of motion is:

$$
m_i
\frac{d^2\mathbf r_{i,xy}}{dt^2}
=
\mathbf F^{mag,ext}_{i,xy}
+
\sum_{j\ne i}
\left[
\mathbf F^{mag,inter}_{ij}
+
\mathbf F^{capillary}_{ij}
\right]
+
D
\frac{d\mathbf r_{i,xy}}{dt}
$$

This is Equation (2) in Dong.

Using simulation notation:

$$
\boxed{
m_i\ddot{\mathbf p}_i
=
\mathbf F^{ext}_i
+
\sum_{j\ne i}
\left(
\mathbf F^{mag,inter}_{ij}
+
\mathbf F^{capillary}_{ij}
\right)
-
c\dot{\mathbf p}_i
}
$$

---

## 12. First-Order ODE Form for Simulation

Let:

$$
\mathbf v_i=\dot{\mathbf p}_i
$$

Then:

$$
\dot{\mathbf p}_i=\mathbf v_i
$$

$$
\dot{\mathbf v}_i
=
\frac{1}{m_i}
\left[
\mathbf F^{ext}_i
+
\sum_{j\ne i}
\left(
\mathbf F^{mag,inter}_{ij}
+
\mathbf F^{capillary}_{ij}
\right)
-
c\mathbf v_i
\right]
$$

So the state for robot $i$ is:

$$
\mathbf y_i=
\begin{bmatrix}
x_i\\
y_i\\
v_{x,i}\\
v_{y,i}
\end{bmatrix}
$$

For $N$ robots:

$$
\mathbf y=
[x_1,y_1,v_{x,1},v_{y,1},\dots,x_N,y_N,v_{x,N},v_{y,N}]^T
$$

---

## 13. Force Jacobian and Stability

For a single source force:

$$
\mathbf f_i
=
C_Fu_i
\frac{\mathbf r_i}{|\mathbf r_i|^5}
$$

with:

$$
\mathbf r_i=[x_i,y_i]^T
$$

the Jacobian is:

$$
\boxed{
\mathbf J_i
=
\frac{C_Fu_i}{|\mathbf r_i|^7}
\begin{bmatrix}
y_i^2-4x_i^2 & -5x_iy_i\\
-5x_iy_i & x_i^2-4y_i^2
\end{bmatrix}
}
$$

The total force Jacobian is:

$$
\boxed{
\mathbf J_F
=
\sum_i \mathbf J_i
}
$$

For stability at an equilibrium point, using force dynamics:

$$
\mathbf F=-\nabla U
$$

$$
\mathbf J_F=-\nabla^2U
$$

A stable potential-energy minimum requires:

$$
\nabla^2U \succ 0
$$

equivalently:

$$
\mathbf J_F \prec 0
$$

For a $2\times2$ Jacobian, this requires:

$$
\boxed{
\operatorname{tr}(\mathbf J_F)<0
}
$$

$$
\boxed{
\det(\mathbf J_F)>0
}
$$

A single source has:

$$
\operatorname{tr}(\mathbf J_i)
=
-\frac{3C_Fu_i}{|\mathbf r_i|^5}
$$

and:

$$
\det(\mathbf J_i)
=
-\frac{4(C_Fu_i)^2}{|\mathbf r_i|^{10}}
$$

Since the determinant is negative, a single source creates a saddle, not a stable trap. Multiple sources must be combined to create useful equilibria.

---

## 14. Numerical Implementation Notes

### 14.1 Avoid singularities

The magnetic force scales as:

$$
\frac{1}{r^4}
$$

in magnitude, since:

$$
\frac{\mathbf r}{r^5}
$$

has magnitude:

$$
\frac{1}{r^4}
$$

The capillary force scales as:

$$
\frac{1}{r}
$$

in magnitude, since:

$$
\frac{\mathbf r}{r^2}
$$

has magnitude:

$$
\frac{1}{r}
$$

Use a cutoff:

$$
r_{safe}=\max(r,r_{min})
$$

A physically reasonable choice is:

$$
r_{min}=2R
$$

because two circular robots cannot overlap.

---

### 14.2 Recommended simulation force functions

For robots:

$$
\mathbf r_{ij}=\mathbf p_i-\mathbf p_j
$$

$$
r=\max(|\mathbf r_{ij}|,2R)
$$

$$
\mathbf F^{mag}_{ij}
=
K_m\frac{\mathbf r_{ij}}{r^5}
$$

where:

$$
K_m=\frac{3\mu_0m_im_j}{4\pi}
$$

$$
\mathbf F^{cap}_{ij}
=
K_c\frac{\mathbf r_{ij}}{r^2}
$$

where:

$$
K_c=2\pi\gamma R^2\sin^2(C)
$$

---

## 15. Simulation-Ready Summary

Use this full equation:

$$
\boxed{
\dot{\mathbf p}_i=\mathbf v_i
}
$$

$$
\boxed{
\dot{\mathbf v}_i
=
\frac{1}{m_i}
\left[
\sum_k
C_{F,k}u_k
\frac{\mathbf p_i-\mathbf p_k^{ext}}
{|\mathbf p_i-\mathbf p_k^{ext}|^5}
+
\sum_{j\ne i}
\left(
K_{m,ij}
\frac{\mathbf p_i-\mathbf p_j}
{|\mathbf p_i-\mathbf p_j|^5}
+
K_c
\frac{\mathbf p_i-\mathbf p_j}
{|\mathbf p_i-\mathbf p_j|^2}
\right)
-
c\mathbf v_i
\right]
}
$$

with:

$$
K_{m,ij}
=
\frac{3\mu_0m_im_j}{4\pi}
$$

$$
m_i=M_i\pi R_i^2h_i
$$

$$
K_c=2\pi\gamma R^2\sin^2(C)
$$

---

## 16. Final Consistency Check with Dong

Your force model is consistent with Dong if:

1. Robots move only in 2D.
2. Robot magnetic moments remain vertical.
3. Inter-robot magnetic force is repulsive.
4. Inter-robot force uses:

$$
\mathbf F^{mag,inter}_{ij}
=
\frac{3\mu_0}{4\pi}
\frac{|M\pi R^2h|^2}{|\mathbf r_{ij}|^5}
\mathbf r_{ij}
$$

5. Capillary force is included:

$$
\mathbf F^{capillary}_{ij}
=
2\pi\gamma R^2\sin^2(C)
\frac{\mathbf r_{ij}}{|\mathbf r_{ij}|^2}
$$

6. Drag is dissipative:

$$
\mathbf F^{drag}_i=-c\mathbf v_i
$$

7. External magnetic force is derived from the external magnetic potential energy or from the simplified source-force model.

Dong's paper explicitly includes external magnetic force, inter-robot magnetic force, capillary force, and damping in the same dynamic equation, so the complete model above is consistent with their simulation framework.
