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

## 3. Comparison with Khalesi parameters

Khalesi uses:

| Parameter | Khalesi |
|---|---:|
| PM magnetic moment | \(8\ \mathrm{A\,m^2}\) |
| MR magnetic moment | \(2\times10^{-6}\ \mathrm{A\,m^2}\) |
| MR radius | \(0.250\ \mathrm{mm}\) |

My current simulation:

| Parameter | This simulation |
|---|---:|
| PM magnetic moment | \(8\ \mathrm{A\,m^2}\) |
| MR magnetic moment | \(1.25\times10^{-4}\ \mathrm{A\,m^2}\) |
| MR side length | \(0.500\ \mathrm{mm}\) |

Moment ratio:

\[
\frac{1.25\times10^{-4}}
     {2\times10^{-6}}
=
62.5
\]

\[
\boxed{
m_r(\text{mine})
\approx 62.5\times
m_r(\text{Khalesi})
}
\]

Thus the PMs are essentially identical, but the microrobot magnetic moment in the current simulation is approximately two orders of magnitude larger.

---

## 4. External PM force on one microrobot at workspace center

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

## 5. Microrobot mass

Robot volume:

\[
V_r
=
L_r^3
=
(0.0005)^3
\]

\[
V_r
=
1.25\times10^{-10}\ \mathrm{m^3}
\]

Robot mass:

\[
m_{robot}
=
\rho V_r
\]

\[
m_{robot}
=
7500
(1.25\times10^{-10})
\]

\[
\boxed{
m_{robot}
=
9.38\times10^{-7}\ \mathrm{kg}
}
\]

---

## 6. Acceleration without drag

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

## 7. Drag-limited speed with current drag model

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

---

# 8. Inter-robot magnetic force

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

---

## 9. Inter-robot force table

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

## 10. Summary

### Magnetic moments

| Quantity | Value |
|---|---:|
| PM magnetic moment | \(8.0\ \mathrm{A\,m^2}\) |
| MR magnetic moment | \(1.25\times10^{-4}\ \mathrm{A\,m^2}\) |

### Force scales

| Quantity | Value |
|---|---:|
| PM \(\rightarrow\) MR force | \(76.8\ \mathrm{nN}\) |
| MR acceleration | \(8.19\times10^{-2}\ \mathrm{m/s^2}\) |
| Drag-limited MR speed | \(5.45\ \mathrm{cm/s}\) |

### Comparison with Khalesi

| Quantity | Khalesi | This simulation |
|---|---:|---:|
| PM moment | \(8\ \mathrm{A\,m^2}\) | \(8\ \mathrm{A\,m^2}\) |
| MR moment | \(2\times10^{-6}\) | \(1.25\times10^{-4}\) |
| Ratio | 1 | 62.5 |

Thus the dominant difference between the two models is not the permanent magnet but the microrobot magnetic moment, which is approximately:

\[
\boxed{
62.5\times
\text{larger than Khalesi}
}
\]

in the current simulation.















# Updated Comparison Table with Corrected Equilibrium Distances
Common external PM parameters:

| Parameter | Value |
|---|---|
| PM side length | \(L_s=0.02\ \mathrm{m}\) |
| PM magnetization | \(M_s=1.0\times10^6\ \mathrm{A/m}\) |
| PM magnetic moment | \(m_s=8.0\ \mathrm{A\,m^2}\) |
| PM distance from center | \(r=0.25\ \mathrm{m}\) |
| Fluid viscosity | \(\mu=0.001\ \mathrm{Pa\,s}\) |
| Drag correction | \(\alpha=0.3\) |

---

## Main comparison

| Case | Robot model | Robot moment \(m_r\) | Robot mass | PM force \(F_{PM}\) | Acceleration \(a\) | Drag \(c\) | Speed \(v=F/c\) | Contact distance | Estimated \(d_{eq}\) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current | 0.5 mm NdFeB cube | \(1.25\times10^{-4}\) | \(9.38\times10^{-7}\) kg | \(7.68\times10^{-8}\) N | \(8.19\times10^{-2}\) m/s² | \(1.41\times10^{-6}\) Ns/m | \(5.43\times10^{-2}\) m/s | 0.50 mm | 15.7 mm |
| Yousefi | 250 µm disk, 250 µm thick N42 | \(1.29\times10^{-5}\) | \(1.23\times10^{-7}\) kg | \(7.91\times10^{-9}\) N | \(6.43\times10^{-2}\) m/s² | \(7.07\times10^{-7}\) Ns/m | \(1.12\times10^{-2}\) m/s | 0.25 mm | 9.3 mm |
| Dong | 350 µm disk, 100 µm thick, 20% NdFeB | \(1.50\times10^{-5}\) | \(1.78\times10^{-8}\) kg | \(9.20\times10^{-10}\) N | \(5.17\times10^{-2}\) m/s² | \(9.91\times10^{-7}\) Ns/m | \(9.28\times10^{-4}\) m/s | 0.35 mm | 3.9 mm |
| PDMS + SPION | 250 µm disk, 250 µm thick, 20% SPION | \(3.12\times10^{-6}\) | \(1.42\times10^{-8}\) kg | \(2.62\times10^{-10}\) N | \(1.84\times10^{-2}\) m/s² | \(7.07\times10^{-7}\) Ns/m | \(3.70\times10^{-4}\) m/s | 0.25 mm | 3.9 mm |

---

## Inter-robot magnetic force

\[
F_{rr}
=
\frac{3\mu_0}{4\pi}
\frac{m_r^2}{d^4}
\]

| Center distance \(d\) | Mine | Yousefi | Dong | PDMS + SPION |
|---:|---:|---:|---:|---:|
| 0.25 mm | \(1.20\ \mathrm{N}\) | \(2.50\times10^{-4}\ \mathrm{N}\) | \(1.20\times10^{-4}\ \mathrm{N}\) | \(3.64\times10^{-5}\ \mathrm{N}\) |
| 0.35 mm | \(3.12\times10^{-1}\ \mathrm{N}\) | \(1.02\times10^{-4}\ \mathrm{N}\) | \(5.50\times10^{-5}\ \mathrm{N}\) | \(1.52\times10^{-5}\ \mathrm{N}\) |
| 0.50 mm | \(7.50\times10^{-2}\ \mathrm{N}\) | \(4.10\times10^{-5}\ \mathrm{N}\) | \(2.75\times10^{-5}\ \mathrm{N}\) | \(6.10\times10^{-6}\ \mathrm{N}\) |
| 1.00 mm | \(4.69\times10^{-3}\ \mathrm{N}\) | \(2.56\times10^{-6}\ \mathrm{N}\) | \(1.70\times10^{-6}\ \mathrm{N}\) | \(3.81\times10^{-7}\ \mathrm{N}\) |
| 5.00 mm | \(7.50\times10^{-6}\ \mathrm{N}\) | \(4.10\times10^{-9}\ \mathrm{N}\) | \(2.73\times10^{-9}\ \mathrm{N}\) | \(6.10\times10^{-10}\ \mathrm{N}\) |
| 10.00 mm | \(4.69\times10^{-7}\ \mathrm{N}\) | \(2.56\times10^{-10}\ \mathrm{N}\) | \(1.71\times10^{-10}\ \mathrm{N}\) | \(3.81\times10^{-11}\ \mathrm{N}\) |

---

## Equilibrium spacing estimate

| Case | Estimated \(d_{eq}\) | Meaning |
|---|---:|---|
| Mine | 15.7 mm | MR magnetic moment very large → strong repulsion |
| Yousefi | 9.3 mm | Corrected using PM force and disk m_r from N42 250 µm |
| Dong | 3.9 mm | 350 µm disk, 20% NdFeB, effective M |
| PDMS + SPION | 3.9 mm | 250 µm disk, 250 µm thick, 20% SPION |