# Governing Equations for Paramagnetic Nanocluster (SPION) Locomotion

## 1. Magnetic Field Generation

The locomotion of the paramagnetic nanoclusters is achieved through the combination of a rotating magnetic field and a magnetic field gradient. The orientation vector of the rotating field is defined as:

\[
\mathbf{u} =
\begin{bmatrix}
\cos(\alpha)\cos(\theta) \\
\cos(\alpha)\sin(\theta) \\
\sin(\alpha)
\end{bmatrix}
\]

where:

- \(\theta\) : in-plane heading angle  
- \(\alpha\) : elevation angle  

The rotating magnetic field is then expressed as:

\[
\mathbf{B}(t) =
B_0 \left[
\cos(\omega t)\mathbf{u} +
\sin(\omega t)(\mathbf{n}\times\mathbf{u})
\right]
\]

where:

| Symbol | Description |
|--------|-------------|
| \(B_0\) | Magnetic field magnitude |
| \(\omega\) | Angular frequency of rotation |
| \(\mathbf{n}\) | Unit normal vector of the rotation plane |
| \(\mathbf{u}\) | Desired field direction |

---

## 2. Magnetic Force

A magnetic field gradient generates a translational force on the magnetic chain:

\[
\mathbf{F}_m = \frac{V\chi}{\mu_0} (\mathbf{B}\cdot\nabla)\mathbf{B}
\]

where:

| Symbol | Description |
|--------|-------------|
| \(V\) | Total volume of the chain |
| \(\chi\) | Magnetic susceptibility |
| \(\mu_0\) | Vacuum permeability |
| \(\mathbf{B}\) | Magnetic flux density |

---

## 3. Hydrodynamic Drag Force

In the low Reynolds number regime:

\[
F_d = \frac{2\pi\eta L v}{\ln\left(\frac{L}{2\sqrt{S/\pi}}\right) + \ln(2) - 0.5}
\]

where:

| Symbol | Description |
|--------|-------------|
| \(\eta\) | Dynamic viscosity |
| \(L\) | Chain length |
| \(S\) | Chain cross-sectional area |
| \(v\) | Translational velocity |

---

## 4. Friction Force

The tumbling chain interacts with the nearby substrate:

\[
F_f = \mu_s W, \quad \mu_s = \frac{\eta}{hP} \hat{v}
\]

where:

| Symbol | Description |
|--------|-------------|
| \(\mu_s\) | Wet friction coefficient |
| \(W\) | Normal load on the chain |
| \(h\) | Gap between chain and surface |
| \(P\) | Pressure on the chain |
| \(\hat{v}\) | Fluid velocity in the gap |

Using the circumferential velocity \(\hat{v} = v\omega\) and \(k = \frac{\eta}{hP}\), the friction force becomes:

\[
F_f = k \omega R W
\]

---

## 5. Force Balance

At low Reynolds number:

\[
\sum F = F_m + F_f - F_d = 0
\]

---

## 6. Velocity Model

Combining magnetic, friction, and drag forces:

\[
v = \frac{k \omega R W + \frac{V\chi}{\mu_0} (\mathbf{B}\cdot\nabla)\mathbf{B}}{\frac{2\pi\eta L}{\ln\left(\frac{L}{2\sqrt{S/\pi}}\right) + \ln(2) - 0.5}}
\]

---

## 7. Dipole–Dipole Interaction Force

For swarm formation and disaggregation:

\[
\mathbf{F}_{m,i} = \frac{3\mu_0}{4\pi} \sum_{j\neq i} \frac{m_i m_j}{r_{ij}^4} \Big[ (1-5(\hat{\mathbf{m}}\cdot\hat{\mathbf{r}}_{ij})^2)\hat{\mathbf{r}}_{ij} + 2(\hat{\mathbf{m}}\cdot\hat{\mathbf{r}}_{ij})\hat{\mathbf{m}} \Big]
\]

where:

| Symbol | Description |
|--------|-------------|
| \(m_i, m_j\) | Magnetic moments of chains |
| \(r_{ij}\) | Distance between chains |
| \(\hat{\mathbf{r}}_{ij}\) | Unit vector between chains |
| \(\hat{\mathbf{m}}\) | Unit magnetic moment vector |

---

## 8. Summary

The enhanced tumbling locomotion results from:

\[
\text{Rotating Magnetic Field} + \text{Gradient Field}
\]

- **Rotating field**: produces torque and chain tumbling  
- **Gradient field**: produces translational motion  
- **Friction**: breaks symmetry of motion  
- **Drag**: opposes movement  
- **Dipole interactions**: govern chain formation, chain length, and disaggregation

\[
\mathbf{F}_{magnetic} + \mathbf{F}_{dipole} + \mathbf{F}_{friction} + \mathbf{F}_{drag} = 0
\]

This set of equations models the motion, aggregation, and disaggregation of SPION nanocluster swarms.