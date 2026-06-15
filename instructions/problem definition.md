# Single Stable Equilibrium Point Optimization

The objective of the optimization is to determine the permanent-magnet control inputs

\[
\mathbf{u}=[u_1,u_2,\ldots,u_N]^T
\]

that generate a prescribed stable equilibrium point at a desired workspace location

\[
\mathbf{p}_d.
\]

The optimization problem is formulated as

\[
\min_{\mathbf{u}}
\left(
\sum_{i=1}^{N}u_i^2-E_t
\right)^2
\]

where \(E_t\) denotes the desired actuation effort level.

## Constraints

### 1. Equilibrium Constraint

The net magnetic force at the target position must vanish:

\[
\mathbf{F}(\mathbf{p}_d,\mathbf{u})=\mathbf{0}
\]

which guarantees that the target location is an equilibrium point of the magnetic force field.

### 2. Stability Constraint

Let

\[
\mathbf{H}(\mathbf{p}_d,\mathbf{u})
\]

denote the Hessian matrix of the magnetic potential energy.

A stable equilibrium requires the Hessian to be positive definite.

For a two-dimensional system, positive definiteness is enforced through

\[
\operatorname{tr}(\mathbf{H}) \ge \varepsilon_t
\]

and

\[
\det(\mathbf{H}) \ge \varepsilon_d
\]

where \(\varepsilon_t\) and \(\varepsilon_d\) are small positive margins.

### 3. Eigenvalue Ratio Constraint

To control the anisotropy of the potential well, the ratio of the Hessian eigenvalues is prescribed as

\[
\frac{\lambda_1}{\lambda_2}=r
\]

where \(r\) is a user-defined stiffness ratio.

Using the invariants of a \(2\times2\) matrix, the constraint is expressed as

\[
\operatorname{tr}(\mathbf{H})^2(1-r)^2
-
\Delta(1+r)^2
=
0
\]

with

\[
\Delta=
\operatorname{tr}(\mathbf{H})^2
-
4\det(\mathbf{H})
\]

### 4. Eigenvector Orientation Constraint

The principal axis of the potential well can be aligned with a prescribed angle

\[
\phi_d
\]

The principal-axis orientation is computed as

\[
\phi
=
\frac12
\tan^{-1}
\left(
\frac{2H_{xy}}
{H_{xx}-H_{yy}}
\right)
\]

Because eigenvectors are defined up to sign, the orientation constraint is imposed modulo \(\pi\):

\[
1-\cos\left(2(\phi-\phi_d)\right)=0
\]

## Final Optimization Problem

\[
\begin{aligned}
\min_{\mathbf{u}}
\quad &
\left(
\sum_{i=1}^{N}u_i^2-E_t
\right)^2
\\[4pt]
\text{s.t.}
\quad &
\mathbf{F}(\mathbf{p}_d,\mathbf{u})=\mathbf{0}
\\
&
\operatorname{tr}(\mathbf{H}) \ge \varepsilon_t
\\
&
\det(\mathbf{H}) \ge \varepsilon_d
\\
&
\frac{\lambda_1}{\lambda_2}=r
\\
&
\phi=\phi_d
\end{aligned}
\]

This optimization generates a stable magnetic trap with a prescribed position, stiffness ratio, and principal-axis orientation.

---

# Dual Stable Equilibrium Point Optimization

The objective is to determine the permanent-magnet control inputs

\[
\mathbf{u}=[u_1,u_2,\ldots,u_N]^T
\]

such that two desired locations

\[
\mathbf{p}_1
\]

and

\[
\mathbf{p}_2
\]

become stable equilibrium points of the magnetic force field.

The optimization problem is formulated as

\[
\min_{\mathbf{u}}
\left(
\sum_{i=1}^{N}u_i^2-E_t
\right)^2
+
w_r
\left(
\eta_1+\eta_2
\right)
\]

where

\[
\eta_i
=
\frac{\Delta_i}
{\operatorname{tr}(\mathbf{H}_i)^2}
\]

is an anisotropy measure evaluated at equilibrium point \(i\).

Since

\[
\Delta_i
=
\operatorname{tr}(\mathbf{H}_i)^2
-
4\det(\mathbf{H}_i)
\]

the anisotropy metric can also be written as

\[
\eta_i
=
\left(
\frac{\lambda_{1,i}-\lambda_{2,i}}
{\lambda_{1,i}+\lambda_{2,i}}
\right)^2
\]

Minimizing \(\eta_i\) drives the eigenvalues toward equality and therefore promotes isotropic potential wells.

## Constraints

### 1. Equilibrium Constraints

The net magnetic force must vanish at both desired positions:

\[
\mathbf{F}(\mathbf{p}_1,\mathbf{u})=\mathbf{0}
\]

\[
\mathbf{F}(\mathbf{p}_2,\mathbf{u})=\mathbf{0}
\]

### 2. Stability Constraints at the First Equilibrium

\[
\operatorname{tr}
\left(
\mathbf{H}(\mathbf{p}_1,\mathbf{u})
\right)
\ge
\varepsilon_t
\]

\[
\det
\left(
\mathbf{H}(\mathbf{p}_1,\mathbf{u})
\right)
\ge
\varepsilon_d
\]

### 3. Stability Constraints at the Second Equilibrium

\[
\operatorname{tr}
\left(
\mathbf{H}(\mathbf{p}_2,\mathbf{u})
\right)
\ge
\varepsilon_t
\]

\[
\det
\left(
\mathbf{H}(\mathbf{p}_2,\mathbf{u})
\right)
\ge
\varepsilon_d
\]

## Final Optimization Problem

\[
\begin{aligned}
\min_{\mathbf{u}}
\quad &
\left(
\sum_{i=1}^{N}u_i^2-E_t
\right)^2
+
w_r
\left(
\eta_1+\eta_2
\right)
\\[4pt]
\text{s.t.}
\quad &
\mathbf{F}(\mathbf{p}_1,\mathbf{u})=\mathbf{0}
\\
&
\mathbf{F}(\mathbf{p}_2,\mathbf{u})=\mathbf{0}
\\
&
\operatorname{tr}(\mathbf{H}_1)\ge\varepsilon_t
\\
&
\det(\mathbf{H}_1)\ge\varepsilon_d
\\
&
\operatorname{tr}(\mathbf{H}_2)\ge\varepsilon_t
\\
&
\det(\mathbf{H}_2)\ge\varepsilon_d
\end{aligned}
\]

This optimization simultaneously generates two stable magnetic traps while minimizing actuation effort and promoting nearly isotropic local potential wells at both equilibrium points.