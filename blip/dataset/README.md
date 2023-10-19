# MSSM Datasets

Data is ordered in csv format, where one line corresponds to one point in the
SUSY parameter space. 

The mapping from indices to defined values is provided below. Note that indexing
begins from 0 in this list. If using software that indexes from 1 (Mathematica), 
you must remember to add 1 to the tabled index to return the variable of interest.

The first entries are GUT scale parameters of SUSY theories, and will depend on 
which subspace is sampled. The number of parameters of the theory will be 
defined as D.

After entry D, both datasets are indexed in the same way. 

Information between entries may be degenerate in some cases. Latex notation 
for masses excludes the tilde conventionally associated with sparticles. 
Table with particles names found in lib/micromegas_5.2.6/MSSM/work/models.

cMSSM:
0: Universal sfermion mass (M_0)
1: Oniversal gaugino mass (M_{1/2})
2: Oniversal trilinear coupling (A_0)
3: tan beta
4 / D-1: sgn(mu)

pMSSM:
0: GUT scale Bino mass (M_1)
1: GUT scale Wino mass (M_2)
2: GUT scale Gluino mass (M_3)
3: GUT Scale Bilinear Higgs mass (\mu)
4: GUT scale Pseudoscalar Higgs mass (M_A)
5: GUT scale Trilinear bottom coupling (A_b)
6: GUT scale Trilinear top coupling (A_t)
7: GUT scale Trilinear tau coupling (A_\tau)
8: GUT scale Left handed 1st and 2nd gen slepton mass (m_L_{1/2})
9: GUT scale Left handed 3rd gen slepton mass (m_L_3)
10: GUT scale Right handed 1st and 2nd gen slepton mass (m_e_R)
11: GUT scale Right handed 3rd gen slepton mass (m_\tau_R)
12: GUT scale Left handed 1st and 2nd gen squark mass (m_Q_{1/2})
13: GUT scale Left handed 3rd gen squark mass (m_Q_3)
14: GUT scale Right handed 1st and 2nd gen up-type squark mass (m_u_R)
15: GUT scale Right handed 3rd gen up-type squark mass (m_t_R)
16: GUT scale Right handed 1st and 2nd gen down-type squark mass (m_d_R)
17: GUT scale Right handed 3rd gen down-type squark mass (m_b_R)
18 / D-1: tan beta

*(Weak scale (soft) parameters, Q=M_Z, DRbar renormalization)
D: Up trilinear coupling (A_u)*
D+1: Charm trilinear coupling (A_c)*
D+2: Top trilinear coupling (A_t)*
D+3: Down trilinear coupling (A_d)*
D+4: Strange trilinear coupling (A_s)*
D+5: Bottom trilinear coupling (A_b)*
D+6: Electron trilinear coupling (A_e)*
D+7: Muon trilinear coupling (A_mu)*
D+8: Tau trilinear coupling (A_tau)*
D+9: Tan Beta*
D+10: U(1) gauge coupling, GUT normalization (g', g_1 = sqrt(5/3)g')*
D+11: SU(2) gauge coupling (g or g_2)*
D+12: SU(3) gauge coupling (g_3)*
D+13: Bino mass (M_1)*
D+14: Wino mass (M_2)*
D+15: Gluino mass (M_3)*
D+16: Squared pseudoscalar Higgs mass (M_A^2)*
D+17: Mu parameter (\mu)*
D+18: Down type Higgs squared mass (M_{H_1}^2)*
D+19: Up type Higgs squared mass (M_{H_2}^2)*
D+20: Right handed up squark mass (m_u_R)*
D+21: Right handed charm squark mass (m_c_R)*
D+22: Right handed top squark mass (m_t_R)*
D+23: Right handed down squark mass (m_d_R)*
D+24: Right handed strange squark mass (m_s_R)*
D+25: Right handed bottom squark mass (m_b_R)*
D+26: Right handed selectron mass (m_e_R)*
D+27: Right handed smuon mass (m_\mu_R)*
D+28: Right handed stau mass (m_\tau_R)* 
D+29: Left handed 1st gen. squark mass (m_Q_1)*
D+30: Left handed 2nd gen. squark mass (m_Q_2)*
D+31: Left handed 3rd gen. squark mass (m_Q_3)*
D+32: Left handed 1st gen. slepton mass (m_L_1)*
D+33: Left handed 2nd gen. slepton mass (m_L_2)*
D+34: Left handed 3rd gen. slepton mass (m_L_3)*
D+35: Higgs vev (<v>)*
D+36: Top Yukawa Coupling (Y_t)*
D+37: Bottom Yukawa Coupling (Y_b)*
D+38: Tau Yukawa Coupling (Y_\tau)*

**(Weak scale eigenmasses and LSP components, Q=M_Z, mass ordered when applicable.)
D+39: W mass (M_W)**
D+40: SM/Light CP even Higgs mass (m_h)**
D+41: Heavy CP even Higgs mass (m_H)**
D+42: Pseudoscalar Higgs mass (m_A)**
D+43: Charged Higgs mass (m_{H^+}, m_{H^-})**
D+44: Gluino Mass (M_3)**
D+45: Lightest neutralino mass (\chi^0_1)**
D+46: 2nd neutralino mass (\chi^0_2)**
D+47: 3rd neutralino mass (\chi^0_3)**
D+48: 4th neutralino mass (\chi^0_4)**
D+49: 1st chargino mass (\chi^\pm_1)**
D+50: 2nd chargino mass (\chi^\pm_2)**
D+51: Left handed down squark mass (m_d_L)**
D+52: Left handed up squark mass (m_u_L)**
D+53: Left handed strange squark mass (m_s_L)**
D+54: Left handed charm squark mass (m_c_L)**
D+55: Light bottom sqark mass (m_b_1)**
D+56: Light top sqark mass (m_t_1)**
D+57: Left handed selectron mass (m_e_L)**
D+58: Left handed electron sneutrino mass (m_{\nu_e}_L)**
D+59: Left handed smuon mass (m_\mu_L)**
D+60: Left handed muon sneutrino mass (m_{\nu_\mu}_L)**
D+61: Light stau mass (m_\tau_1)**
D+62: Left handed tau sneutrino mass (m_{\nu_\tau}_L)**
D+63: Right handed down squark mass (m_d_R)**
D+64: Right handed up squark mass (m_u_R)**
D+65: Right handed strange squark mass (m_s_R)**
D+66: Right handed charm squark mass (m_c_R)**
D+67: Heavy bottom squark mass (m_b_2)**
D+68: Heavy top squark mass (m_t_2)**
D+69: Right handed selectron mass (m_e_R)**
D+70: Right handed smuon mass (m_\mu_R)**
D+71: Heavy stau mass (m_\tau_2) **
D+72: Neutralino mixing matrix element (1,1) (LSP -i Bino component) (N_{1,1})**
D+73: Neutralino mixing matrix element (1,2) (LSP -i wino component) (N_{1,2})**
D+74: Neutralino mixing matrix element (1,3) (LSP higgsino 1 component) (N_{1,3})**
D+75: Neutralino mixing matrix element (1,4) (LSP higgsino 2 component) (N_{1,4})**

***(Weak scale measurements and computations)
D+76: Dark matter Relic Density (\Omega_{DM}h^2)
D+77: Anomolous muon magnetic moment (g_\mu - 2)
D+78: b -> s \gamma branching ratio
D+79: b -> s \gamma SM contribution (I believe this is bugged in micromegas)
D+80: B+ -> \tau \nu MSSM / SM branching fraction
D+81: Bs -> mu mu branching ratio
D+82: Ds -> tau nu branching ratio
D+83: Ds -> mu nu branching ratio
D+84: Delta rho parameter
D+85: Rl23 MSSM / SM ratio
D+86: LSP mass
D+87: <\sigma v>
D+88: CDM-proton spin-independant cross-section
D+89: CDM-proton spin-dependant cross-section
D+90: CDM-neutron spin-independant cross-section
D+91: CDM-neutron spin-dependant cross-section

****(Ordered DM channels)
D+92: Channel 1 weight
D+93: Channel 1 particle 1
D+94: Channel 1 particle 2
D+95: Channel 1 particle 3
D+96: Channel 1 particle 4
D+97: Channel 2 weight
D+98: Channel 2 particle 1
D+99: Channel 2 particle 2
D+100: Channel 2 particle 3
D+101: Channel 2 particle 4
D+102: Channel 3 weight
D+103: Channel 3 particle 1
D+104: Channel 3 particle 2
D+105: Channel 3 particle 3
D+106: Channel 3 particle 4
D+107: Channel 4 weight
D+108: Channel 4 particle 1
D+109: Channel 4 particle 2
D+110: Channel 4 particle 3
D+111: Channel 4 particle 4
D+112: Channel 5 weight
D+113: Channel 5 particle 1
D+114: Channel 5 particle 2
D+115: Channel 5 particle 3
D+116: Channel 5 particle 4
D+117: Channel 6 weight
D+118: Channel 6 particle 1
D+119: Channel 6 particle 2
D+120: Channel 6 particle 3
D+121: Channel 6 particle 4
D+122: Channel 7 weight
D+123: Channel 7 particle 1
D+124: Channel 7 particle 2
D+125: Channel 7 particle 3
D+126: Channel 7 particle 4
D+127: Channel 8 weight
D+128: Channel 8 particle 1
D+129: Channel 8 particle 2
D+130: Channel 8 particle 3
D+131: Channel 8 particle 4
D+132: Channel 9 weight
D+133: Channel 9 particle 1
D+134: Channel 9 particle 2
D+135: Channel 9 particle 3
D+136: Channel 9 particle 4
D+137: Channel 10 weight
D+138: Channel 10 particle 1
D+139: Channel 10 particle 2
D+140: Channel 10 particle 3
D+141: Channel 10 particle 4

***** GUT gauge couplings (1st order interpolation of SoftSUSY g_i(Q) @ M_{GUT})
D+142: GUT energy (g1=g2) (M_{GUT})
D+143: GUT scale g_1, U(1) gauge coupling
D+144: GUT scale g_2, SU(2) gauge coupling
D+145: GUT scale g_3, SU(3) gauge coupling.