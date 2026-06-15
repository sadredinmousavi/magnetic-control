# Force-Scale Report for Magnetic Microrobot Simulation

## 1. My current simulation case

### Given parameters

| Parameter | Value |
|---|---:|
| External PM side length | \(L_s = 0.02\ \mathrm{m}\) |
| Microrobot side length | \(L_r = 0.0005\ \mathrm{m}\) |
| Magnetization | \(M = 1.0\times10^6\ \mathrm{A/m}\) |
| External PM distance from center | \(r = 0.25\ \mathrm{m}\) |
| Robot density | \(\rho = 7500\ \mathrm{kg/m^3}\) |
| Fluid viscosity | \(\mu = 0.001\ \mathrm{Pa\,s}\) |
| Surface drag correction | \(\alpha = 0.3\) |

---

## 2. Magnetic moments

For a cube:

\[
m = M L^3
\]

### External magnet

\[
m_s = 1.0\times10^6(0.02)^3
\]

\[
m_s = 1.0\times10^6(8\times10^{-6})
\]

\[
\boxed{m_s = 8.0\ \mathrm{A\,m^2}}
\]

### Microrobot

\[
m_r = 1.0\times10^6(0.0005)^3
\]

\[
m_r = 1.0\times10^6(1.25\times10^{-10})
\]

\[
\boxed{m_r = 1.25\times10^{-4}\ \mathrm{A\,m^2}}
\]

---

## 3. External PM force on one microrobot at workspace center

Using the simplified vertical-dipole planar force model:

\[
F =
\frac{3\mu_0}{4\pi}
\frac{m_s m_r}{r^4}
\]

Since

\[
\frac{\mu_0}{4\pi}
=
10^{-7}
\]

\[
F=
3\times10^{-7}
\frac{(8.0)(1.25\times10^{-4})}
     {(0.25)^4}
\]

\[
(0.25)^4
=
0.00390625
\]

\[
F=
3\times10^{-7}
\frac{0.001}
     {0.00390625}
\]

\[
F
=
7.68\times10^{-8}\ \mathrm{N}
\]

\[
\boxed{
F \approx 7.68\times10^{-8}\ \mathrm{N}
}
\]

\[
\boxed{
F \approx 76.8\ \mathrm{nN}
}
\]

---

## 4. Microrobot mass

\[
V_r = L_r^3 = (0.0005)^3
\]

\[
V_r = 1.25\times10^{-10}\ \mathrm{m^3}
\]

\[
m_{robot} = \rho V_r
\]

\[
m_{robot}=7500(1.25\times10^{-10})
\]

\[
\boxed{m_{robot}=9.38\times10^{-7}\ \mathrm{kg}}
\]

---

## 5. Acceleration without drag

\[
a
=
\frac{F}{m}
\]

\[
a
=
\frac
{7.68\times10^{-8}}
{9.38\times10^{-7}}
\]

\[
\boxed{
a
\approx
8.19\times10^{-2}
\ \mathrm{m/s^2}
}
\]

Thus

\[
\boxed{
a
\sim
10^{-1}
\ \mathrm{m/s^2}
}
\]

---

## 6. Drag-limited speed with my current drag model

Current drag model:

\[
c
=
\alpha
6\pi
\mu
R
\]

where

\[
R
=
0.00025\ \mathrm{m}
\]

\[
\alpha
=
0.3
\]

\[
\mu
=
0.001\ \mathrm{Pa\,s}
\]

\[
c
=
0.3
(6\pi)
(0.001)
(0.00025)
\]

\[
\boxed{
c
=
1.41\times10^{-6}
\ \mathrm{N\,s/m}
}
\]

Terminal speed:

\[
v
=
\frac{F}{c}
\]

\[
v
=
\frac
{7.68\times10^{-8}}
{1.41\times10^{-6}}
\]

\[
\boxed{
v
=
5.45\times10^{-2}
\ \mathrm{m/s}
}
\]

\[
\boxed{
v
\approx
5.45\ \mathrm{cm/s}
}
\]

This is likely high for real experiments because the drag model is light and assumes surface correction.

---

# 7. Inter-robot magnetic force

For two identical microrobots with vertical moments:

\[
F_{rr}
=
\frac{3\mu_0}{4\pi}
\frac{m_r^2}{d^4}
\]

Using

\[
m_r
=
1.25\times10^{-4}
\ \mathrm{A\,m^2}
\]

\[
F_{rr}
=
3\times10^{-7}
\frac
{(1.25\times10^{-4})^2}
{d^4}
\]

\[
F_{rr}
=
3\times10^{-7}
\frac
{1.5625\times10^{-8}}
{d^4}
\]

\[
F_{rr}
=
\frac
{4.6875\times10^{-15}}
{d^4}
\]

This is the same simplified vertical-dipole inter-robot force used in Dong’s model.

---

## 8. Inter-robot force table

| Center distance \(d\) | Force \(F_{rr}\) |
|---:|---:|
| 0.5 mm | \(7.50\times10^{-2}\ \mathrm{N}\) |
| 1 mm | \(4.69\times10^{-3}\ \mathrm{N}\) |
| 2 mm | \(2.93\times10^{-4}\ \mathrm{N}\) |
| 4 mm | \(1.83\times10^{-5}\ \mathrm{N}\) |
| 5 mm | \(7.50\times10^{-6}\ \mathrm{N}\) |
| 10 mm | \(4.69\times10^{-7}\ \mathrm{N}\) |
| 20 mm | \(2.93\times10^{-8}\ \mathrm{N}\) |
| 50 mm | \(7.50\times10^{-10}\ \mathrm{N}\) |

Important:

\[
F_{rr}
\propto
\frac{1}{d^4}
\]

Therefore very small changes in distance generate extremely large changes in magnetic interaction force.

A cutoff/contact model is required near

\[
d
\approx
2R
=
0.5\ \mathrm{mm}
\]

to avoid unrealistic forces.

---

## 9. Python plot for inter-robot force vs distance

```python
import numpy as np
import matplotlib.pyplot as plt

mu0 = 4 * np.pi * 1e-7

M = 1000e3
L_robot = 0.0005

m_r = M * L_robot**3

d = np.linspace(0.0005, 0.05, 500)  # 0.5 mm to 50 mm

F_rr = (3 * mu0 / (4 * np.pi)) * (m_r**2) / d**4

plt.figure(figsize=(7, 5))
plt.loglog(d * 1000, F_rr, linewidth=2)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.xlabel("Center-to-center distance d [mm]")
plt.ylabel("Inter-robot magnetic force [N]")
plt.title("Inter-robot magnetic repulsion vs distance")
plt.show()
```











---

## 10. Payload mass

For the circular PLA payload:

| Parameter | Value |
|---|---:|
| Payload radius | \(R_p = 0.015\ \mathrm{m}\) |
| Payload height | \(h_p = 0.001\ \mathrm{m}\) |
| Payload density | \(\rho_p = 1240\ \mathrm{kg/m^3}\) |

The payload volume is:

\[
V_p=\pi R_p^2 h_p
\]

\[
V_p=\pi(0.015)^2(0.001)
\]

\[
V_p=7.07\times10^{-7}\ \mathrm{m^3}
\]

The payload mass is:

\[
m_p=\rho_p V_p
\]

\[
m_p=1240(7.07\times10^{-7})
\]

\[
\boxed{m_p\approx8.76\times10^{-4}\ \mathrm{kg}}
\]

\[
\boxed{m_p\approx0.876\ \mathrm{g}}
\]

---

## 11. Payload acceleration from one microrobot

If one microrobot can transmit approximately the characteristic magnetic actuation force to the payload, then:

\[
F_{robot}\approx5.79\times10^{-8}\ \mathrm{N}
\]

The ideal payload acceleration is:

\[
a_p=\frac{F_{robot}}{m_p}
\]

\[
a_p=
\frac{5.79\times10^{-8}}{8.76\times10^{-4}}
\]

\[
\boxed{
a_p\approx6.61\times10^{-5}\ \mathrm{m/s^2}
}
\]

Thus, a single microrobot produces only a very small acceleration on the payload:

\[
\boxed{
a_p\sim10^{-5}\ \mathrm{m/s^2}
}
\]

---

## 12. Payload acceleration from multiple microrobots

If \(N_c\) microrobots are attached to the payload and their transmitted forces are approximately aligned, then:

\[
F_{payload}\approx N_c F_{robot}
\]

and:

\[
a_p\approx
\frac{N_cF_{robot}}{m_p}
\]

For example:

| Number of attached robots \(N_c\) | Force on payload | Payload acceleration |
|---:|---:|---:|
| 1 | \(5.79\times10^{-8}\ \mathrm{N}\) | \(6.61\times10^{-5}\ \mathrm{m/s^2}\) |
| 2 | \(1.16\times10^{-7}\ \mathrm{N}\) | \(1.32\times10^{-4}\ \mathrm{m/s^2}\) |
| 3 | \(1.74\times10^{-7}\ \mathrm{N}\) | \(1.98\times10^{-4}\ \mathrm{m/s^2}\) |
| 5 | \(2.90\times10^{-7}\ \mathrm{N}\) | \(3.31\times10^{-4}\ \mathrm{m/s^2}\) |

Therefore, even with several attached microrobots, the payload acceleration remains on the order of:

\[
\boxed{
a_p\sim10^{-4}\ \mathrm{m/s^2}
}
\]

for the present payload size and mass.

---

## 13. Payload speed estimate with drag

The current payload drag coefficient in the simulation is:

\[
c_p = 2000c_r
\]

where the robot drag coefficient is:

\[
c_r=1.41\times10^{-6}\ \mathrm{N\,s/m}
\]

Thus:

\[
c_p=2000(1.41\times10^{-6})
\]

\[
\boxed{
c_p\approx2.83\times10^{-3}\ \mathrm{N\,s/m}
}
\]

The drag-limited payload speed from one microrobot is:

\[
v_p=\frac{F_{robot}}{c_p}
\]

\[
v_p=
\frac{5.79\times10^{-8}}{2.83\times10^{-3}}
\]

\[
\boxed{
v_p\approx2.05\times10^{-5}\ \mathrm{m/s}
}
\]

\[
\boxed{
v_p\approx20.5\ \mu\mathrm{m/s}
}
\]

For multiple attached microrobots:

| Number of attached robots \(N_c\) | Estimated payload speed |
|---:|---:|
| 1 | \(20.5\ \mu\mathrm{m/s}\) |
| 2 | \(41.0\ \mu\mathrm{m/s}\) |
| 3 | \(61.5\ \mu\mathrm{m/s}\) |
| 5 | \(102.5\ \mu\mathrm{m/s}\) |

---

## 14. Interpretation

The calculation shows that the payload is much harder to move than a single microrobot. Although one external permanent magnet can exert a force of approximately:

\[
F\approx58\ \mathrm{nN}
\]

on one microrobot, the resulting acceleration of the \(15\ \mathrm{mm}\)-radius PLA payload is only:

\[
a_p\approx6.6\times10^{-5}\ \mathrm{m/s^2}.
\]

Moreover, with the current payload drag coefficient, the corresponding drag-limited speed is only:

\[
v_p\approx20\ \mu\mathrm{m/s}
\]

for one attached microrobot. Therefore, visible payload transport requires either:

1. several microrobots attached to the payload,
2. lower payload drag,
3. lower payload mass,
4. stronger magnetic actuation,
5. or stronger robot-payload tangential coupling.

The payload will not move significantly if the robot-payload interaction is only normal contact without sufficient tangential adhesion/friction, because attached robots must be able to transmit tangential magnetic force into the payload.