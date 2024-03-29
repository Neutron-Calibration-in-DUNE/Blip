"""
"""
from enum import Enum
import numpy as np

blip_datasets = [
    'wire_view', 'wire_view_cluster', 'wire_views',
    'edep', 'edep_cluster',
    'tpc', 'tpc_cluster',
    'segment', 'segment_cluster',
]

mssm_datasets = [
    'cmssm', 'pmssm'
]


class ParticleLabel(Enum):
    Undefined = -1
    Noise = 0
    Electron = 11
    Positron = -11
    ElectronNeutrino = 12
    AntiElectronNeutrino = -12
    Muon = 13
    AntiMuon = -13
    MuonNeutrino = 14
    AntiMuonNeutrino = -14
    Tauon = 15
    AntiTauon = -15
    TauonNeutrino = 16
    AntiTauonNeutrino = -16
    Gamma = 22
    Pion0 = 111
    PionPlus = 211
    PionMinus = -211
    Kaon0 = 311
    KaonPlus = 321
    KaonMinus = -321
    Neutron = 2112
    AntiNeutron = -2112
    Proton = 2212
    AntiProton = -2212
    Deuteron = 1000010020
    Triton = 1000010030
    Alpha = 1000020040
    Sulfur32 = 1000160320
    Sulfur33 = 1000160330
    Sulfur34 = 1000160340
    Sulfur35 = 1000160350
    Sulfur36 = 1000160360
    Chlorine35 = 1000170350
    Chlorine36 = 1000170360
    Chlorine37 = 1000170370
    Chlorine38 = 1000170380
    Chlorine39 = 1000170390
    Chlorine40 = 1000170400
    Argon36 = 1000180360
    Argon37 = 1000180370
    Argon38 = 1000180380
    Argon39 = 1000180390
    Argon40 = 1000180400
    Argon41 = 1000180410
    Ion = 1000000000


class TopologyLabel(Enum):
    """
    High-level description of the shape of certain
    event types.
    """

    Undefined = -1
    Noise = 0
    Blip = 1
    Track = 2
    Shower = 3


class PhysicsMicroLabel(Enum):
    """
    micro-level descriptions of topological types.
    This further breaks down the blip/track/shower
    topology labels into micro-level physics.
    """

    Undefined = -1
    Noise = 0
    MIPIonization = 1
    HIPIonization = 2
    ElectronIonization = 3
    Bremsstrahlung = 4
    Annihilation = 5
    PhotoElectric = 6
    GammaCompton = 7
    GammaConversion = 8
    HadronElastic = 9
    HadronInelastic = 10


class PhysicsMesoLabel(Enum):
    """
    meso-level descriptions of topological types.
    """

    Undefined = -1
    Noise = 0
    MIP = 1
    HIP = 2
    DeltaElectron = 3
    MichelElectron = 4
    ElectronShower = 5
    PositronShower = 6
    PhotonShower = 7
    LowEnergyIonization = 8
    NeutronCaptureGamma474 = 9
    NeutronCaptureGamma336 = 10
    NeutronCaptureGamma256 = 11
    NeutronCaptureGamma118 = 12
    NeutronCaptureGamma083 = 13
    NeutronCaptureGamma051 = 14
    NeutronCaptureGamma016 = 15
    NeutronCaptureGammaOther = 16
    Pi0Decay = 17
    AlphaDecay = 18
    BetaDecay = 19
    GammaDecay = 20
    NuclearRecoil = 21
    ElectronRecoil = 22


class PhysicsMacroLabel(Enum):
    """
    macro-level descriptions of topological types
    """

    Undefined = -1
    Noise = 0

    # Neutrino interactions
    CCNue = 1
    CCNuMu = 2
    NC = 3

    Cosmics = 4

    # Radiological interactions
    Ar39 = 5
    Ar42 = 6
    K42 = 7
    Kr85 = 8
    Rn222 = 9
    Po218a = 10
    Po218b = 11
    At218a = 12
    At218b = 13
    Rn218 = 14
    Pb214 = 15
    Bi214a = 16
    Bi214b = 17
    Po214 = 18
    Tl210 = 19
    Pb210a = 20
    Pb210b = 21
    Bi210a = 22
    Bi210b = 23
    Po210 = 24


classification_labels = {
    "particle": {
        -1: "undefined",
        0: "noise",
        11: "electron",
        -11: "positron",
        12: "electron_neutrino",
        -12: "anti-electron_neutrino",
        13: "muon",
        -13: "anti-muon",
        14: "muon_neutrino",
        -14: "anti-muon_neutrino",
        15: "tauon",
        -15: "anti-tauon",
        16: "tauon_neutrino",
        -16: "anti-tauon_neutrino",
        22: "gamma",
        111: "pion0",
        211: "pion_plus",
        -211: "pion_minus",
        311: "kaon0",
        321: "kaon_plus",
        -321: "kaon_minus",
        2112: "neutron",
        -2112: "anti-neutron",
        2212: "proton",
        -2212: "anti-proton",
        1000010020: "deuteron",
        1000010030: "triton",
        1000020040: "alpha",
        1000160320: "sulfur_32",
        1000160330: "sulfur_33",
        1000160340: "sulfur_34",
        1000160350: "sulfur_35",
        1000160360: "sulfur_36",
        1000170350: "chlorine_35",
        1000170360: "chlorine_36",
        1000170370: "chlorine_37",
        1000170380: "chlorine_38",
        1000170390: "chlorine_39",
        1000170400: "chlorine_40",
        1000180360: "argon_36",
        1000180370: "argon_37",
        1000180380: "argon_38",
        1000180390: "argon_39",
        1000180400: "argon_40",
        1000180410: "argon_41",
        1000000000: "ion",
    },
    "topology": {
        -1: "undefined",
        0: "noise",
        1: "blip",
        2: "track",
        3: "shower",
    },
    "physics_micro": {
        -1: "undefined",
        0: "noise",
        1: "mip_ionization",
        2: "hip_ionization",
        3: "electron_ionization",
        4: "bremsstrahlung",
        5: "annihilation",
        6: "photo_electric",
        7: "gamma_compton",
        8: "gamma_conversion",
        9: "hadron_elastic",
        10: "hadron_inelastic",
    },
    "physics_meso": {
        -1: "undefined",
        0: "noise",
        1: "mip",
        2: "hip",
        3: "delta_electron",
        4: "michel_electron",
        5: "electron_shower",
        6: "positron_shower",
        7: "photon_shower",
        8: "low_energy_ionization",
        9: "neutron_capture_gamma_474",
        10: "neutron_capture_gamma_336",
        11: "neutron_capture_gamma_256",
        12: "neutron_capture_gamma_118",
        13: "neutron_capture_gamma_083",
        14: "neutron_capture_gamma_051",
        15: "neutron_capture_gamma_016",
        16: "neutron_capture_gamma_other",
        17: "pi0_decay",
        18: "alpha_decay",
        19: "beta_decay",
        20: "gamma_decay",
        21: "nuclear_recoil",
        22: "electron_recoil"
    },
    "physics_macro": {
        -1: "undefined",
        0: "noise",
        1: "cc_nu_e",
        2: "cc_nu_mu",
        3: "nc",
        4: "cosmics",
        5: "ar39",
        6: "ar42",
        7: "k42",
        8: "kr85",
        9: "rn222",
        10: "po218a",
        11: "po218b",
        12: "at218a",
        13: "at218b",
        14: "rn218",
        15: "pb214",
        16: "bi214a",
        17: "bi214b",
        18: "po214",
        19: "tl210",
        20: "pb210a",
        21: "pb210b",
        22: "bi210a",
        23: "bi210b",
        24: "po210",
    },
    "hit": {
        0: "induction",
        1: "hit",
    },
}

projection_types = [
    'signmu_mgaugino',
    'signmu_msfermion',
    'signA0_msfermion',
    'signmu_tanbeta',
    'A0_msfermion'
]

# the two available subspaces are the
# cMSSM, and pMSSM.
subspaces = {
    'cmssm':    5,
    'pmssm':    19,
}
# experimental constraints for the
# measured higgs mass and DM relic density.
# current FANL+BNL limits on muon g-2 is:
#   1.16592061(41) × 10−11
base_constraints = {
    "higgs_mass":       125.09,
    "higgs_mass_sigma": 3.0,
    "dm_relic_density": .11,
    "dm_relic_density_sigma": .03,
    "muon_magnetic_moment": 0.0011659209,
    "muon_magnetic_moment_sigma": 0.0000000006
}
# experimental bounds for various
# MSSM parameter values.
cmssm_constraints = {
    "m1":   {
        "bounds": [[0, -50.0], [50.0, 4000.0]],
        "prior": "uniform",
        "type": "continuous"
    },
    "m2":   {
        "bounds": [[-4000.0, -100.0], [100.0, 4000.0]],
        "prior": "uniform",
        "type": "continuous"
    },
    "m3":   {
        "bounds": [400.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "mmu":  {
        "bounds": [[-4000.0, -100.0], [100.0, 4000.0]],
        "prior": "uniform",
        "type": "continuous"
    },
    "mA":   {
        "bounds": [100.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
}
pmssm_constraints = {
    "m1":   {
        "bounds": [[-4000.0, -50.0], [50.0, 4000.0]],
        "prior": "uniform",
        "type": "continuous"
    },
    "m2":   {
        "bounds": [[-4000.0, -100.0], [100.0, 4000.0]],
        "prior": "uniform",
        "type": "continuous"
    },
    "m3":   {
        "bounds": [400.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "mmu":  {
        "bounds": [[-4000.0, -100.0], [100.0, 4000.0]],
        "prior": "uniform",
        "type": "continuous"
    },
    "mA":   {
        "bounds": [100.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "At":   {
        "bounds": [-4000.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "Ab":   {
        "bounds": [-4000.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "Atau": {
        "bounds": [-4000.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "mL12": {
        "bounds": [100.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
    },
    "mL3":  {
        "bounds": [100.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "me12": {
        "bounds": [100.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "me3":  {
        "bounds": [100.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "mQ12": {
        "bounds": [400.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "mQ3":  {
        "bounds": [200.0, 4000.0],
        "prior": "uniform",
        "type": "continuous",
        },
    "mu12": {
        "bounds": [400.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "mu3":  {
        "bounds": [200.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "md12": {
        "bounds": [400.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "md3":  {
        "bounds": [200.0, 4000.0],
        "prior": "uniform",
        "type": "continuous"
        },
    "tanb": {
        "bounds": [1.0, 60.0],
        "prior": "uniform",
        "type": "continuous"
    }
}
# column names for the simple
# two-dimensional MSSM model
simple_columns = [
    'gut_m0',
    'gut_m12',
    'sign_mu'
]
# parameter names for the
# cMSSM model.
cmssm_columns = [
    'gut_m0',
    'gut_m12',
    'gut_A0',
    'gut_tanb',
    'sign_mu'
]
# parameter names for the
# pMSSM model
pmssm_columns = [
    'gut_m1', 'gut_m2',
    'gut_m3', 'gut_mmu',
    'gut_mA', 'gut_At',
    'gut_Ab', 'gut_Atau',
    'gut_mL1', 'gut_mL3',
    'gut_me1', 'gut_mtau1',
    'gut_mQ1', 'gut_mQ3',
    'gut_mu1', 'gut_mu3',
    'gut_md1', 'gut_md3',
    'gut_tanb'
]
# column names corresponding to
# weak parameters.
weak_soft_columns = [
    'weak_Au', 'weak_Ac', 'weak_At',
    'weak_Ad', 'weak_As', 'weak_Ab',
    'weak_Ae', 'weak_Amu', 'weak_Atau',
    'weak_tanb', 'weak_gprime', 'weak_g2',
    'weak_g3', 'weak_m1', 'weak_m2',
    'weak_m3', 'weak_mA2', 'weak_mmu',
    'weak_mH12', 'weak_mH22', 'weak_muR',
    'weak_mcR', 'weak_mtR', 'weak_mdR',
    'weak_msR', 'weak_mbR', 'weak_eR',
    'weak_mmuR', 'weak_mtauR', 'weak_mQ1',
    'weak_mQ2', 'weak_mQ3', 'weak_mL1',
    'weak_mL2', 'weak_mL3', 'weak_higgsvev',
    'weak_Yt', 'weak_Yb', 'weak_Ytau'
]
# column names corresponding to
# mass variables.
weak_mass_columns = [
    'weakm_mW', 'weakm_mh', 'weakm_mH',
    'weakm_mA', 'weakm_mHpm', 'weakm_m3',
    'weakm_mneut1', 'weakm_mneut2', 'weakm_mneut3',
    'weakm_mneut4', 'weakm_mcharg1', 'weakm_mcharg2',
    'weakm_mdL', 'weakm_muL', 'weakm_msL',
    'weakm_mcL', 'weakm_mb1', 'weakm_mt1',
    'weakm_meL', 'weakm_mesneuL', 'weakm_mmuL',
    'weakm_mmusneuL', 'weakm_mtau1', 'weakm_mtausneuL',
    'weakm_mdR', 'weakm_muR', 'weakm_msR',
    'weakm_mcR', 'weakm_mb2', 'weakm_mt2',
    'weakm_meR', 'weakm_mmuR', 'weakm_mtau2',
    'weakm_neutmix11', 'weakm_neutmix12',
    'weakm_neutmix13', 'weakm_neutmix14'
]
# column names corresponding to
# computed experimental values.
weak_measurements_columns = [
    'omegah2', 'g-2', 'b->sgamma',
    'b->sgammaSM', 'B+->taunu', 'Bs->mumu',
    'Ds->taunu', 'Ds->munu', 'deltarho',
    'RL23', 'lspmass', 'sigmav',
    'cdmpSI', 'cdmpSD', 'cdmnSI', 'cdmnSD'
]
# column names for micromegas
# dm channel outputs.
dm_channels_columns = [
    'chan1weight', 'chan1part1', 'chan1part2', 'chan1part3', 'chan1part4',
    'chan2weight', 'chan2part1', 'chan2part2', 'chan2part3', 'chan2part4',
    'chan3weight', 'chan3part1', 'chan3part2', 'chan3part3', 'chan3part4',
    'chan4weight', 'chan4part1', 'chan4part2', 'chan4part3', 'chan4part4',
    'chan5weight', 'chan5part1', 'chan5part2', 'chan5part3', 'chan5part4',
    'chan6weight', 'chan6part1', 'chan6part2', 'chan6part3', 'chan6part4',
    'chan7weight', 'chan7part1', 'chan7part2', 'chan7part3', 'chan7part4',
    'chan8weight', 'chan8part1', 'chan8part2', 'chan8part3', 'chan8part4',
    'chan9weight', 'chan9part1', 'chan9part2', 'chan9part3', 'chan9part4',
    'chan10weight', 'chan10part1', 'chan10part2', 'chan10part3', 'chan10part4'
]
# column names for GUT scale
# gauge variables.
gut_gauge_columns = [
    'gut_energscale',
    'gut_gprime',
    'gut_g2',
    'gut_g3'
]
# columns common to both cMSSM and pMSSM
# models.
common_columns = weak_soft_columns + weak_mass_columns \
               + weak_measurements_columns + dm_channels_columns \
               + gut_gauge_columns

# here we have the specification of
# different softsusy output values
# and their corresponding block
softsusy_physical_parameters = {
    "Au":   "au",
    "Ac":   "au",
    "At":   "au",
    "Ad":   "ad",
    "As":   "ad",
    "Ab":   "ad",
    "Ae":   "ae",
    "Amu":  "ae",
    "Atau": "ae",
    "tan":  "hmix",
    "g":    "gauge",
    "g'":   "gauge",
    "g3":   "gauge",
    "M_1":  "msoft",
    "M_2":  "msoft",
    "M_3":  "msoft",
    "mA^2": "hmix",
    "mu":   "hmix",
    "mH1^2": "msoft",
    "mH2^2": "msoft",
    "muR":  "msoft",
    "mcR":  "msoft",
    "mtR":  "msoft",
    "mdR":  "msoft",
    "msR":  "msoft",
    "mbR":  "msoft",
    "meR":  "msoft",
    "mmuR": "msoft",
    "mtauR": "msoft",
    "mqL1": "msoft",
    "mqL2": "msoft",
    "mqL3": "msoft",
    "meL":  "msoft",
    "mmuL": "msoft",
    "mtauL": "msoft",
    "higgs": "hmix",
    "Yt":   "yu",
    "Yb":   "yd",
    "Ytau": "ye",
}
# incidentally, all the weak
# parameters are in the MASS block
softsusy_weak_parameters = {
    "MW":   "MASS",
    "h0":   "MASS",
    "H0":   "MASS",
    "A0":   "MASS",
    "H+":   "MASS",
    "~g":   "MASS",
    "~neutralino(1)":   "MASS",
    "~neutralino(2)":   "MASS",
    "~neutralino(3)":   "MASS",
    "~neutralino(4)":   "MASS",
    "~chargino(1)":     "MASS",
    "~chargino(2)":     "MASS",
    "~d_L":  "MASS",
    "~u_L":  "MASS",
    "~s_L":  "MASS",
    "~c_L":  "MASS",
    "~b_1":  "MASS",
    "~t_1":  "MASS",
    "~e_L":  "MASS",
    "~nue_L": "MASS",
    "~mu_L": "MASS",
    "~numu_L":   "MASS",
    "~stau_1":   "MASS",
    "~nu_tau_L":  "MASS",
    "~d_R":  "MASS",
    "~u_R":  "MASS",
    "~s_R":  "MASS",
    "~c_R":  "MASS",
    "~b_2":  "MASS",
    "~t_2":  "MASS",
    "~e_R":  "MASS",
    "~mu_R": "MASS",
    "~stau_2":   "MASS",
    "N_{1,1}":   "nmix",
    "N_{1,2}":   "nmix",
    "N_{1,3}":   "nmix",
    "N_{1,4}":   "nmix",
}


class DragonTransform:

    def __init__(
        self,
        projections
    ):
        self.projections = projections
        for projection in projections:
            if projection not in projection_types:
                pass

    def get_params(self):
        pass

    def set_params(self):
        pass

    def fit_transform(
        self,
        X
    ):
        results = []
        for projection in self.projections:
            if projection == 'signmu_mgaugino':
                results.append(X[:, 1] * np.sign(X[:, 4]))
            elif projection == 'signmu_msfermion':
                results.append(X[:, 0] * np.sign(X[:, 4]))
            elif projection == 'signA0_msfermion':
                results.append(X[:, 0] * np.sign(X[:, 2]))
            elif projection == 'signmu_tanbeta':
                results.append(X[:, 3] * np.sign(X[:, 4]))
            elif projection == 'A0_msfermion':
                results.append(X[:, 2]/X[:, 0])
        results = np.reshape(results, (-1, len(results)))
        return results
