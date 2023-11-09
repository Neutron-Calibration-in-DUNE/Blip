"""
"""
from enum import Enum
import numpy as np

class TopologyLabel(Enum):
    Undefined = -1
    Noise = 0
    Blip = 1
    Track = 2
    Shower = 3

class ParticleLabel(Enum):
    Undefined = -1
    Noise = 0
    Electron = 11
    Positron = -11
    ElectronNeutrino = 12,
    AntiElectronNeutrino = -12,
    Muon = 13,
    AntiMuon = -13,
    MuonNeutrino = 14,
    AntiMuonNeutrino = -14,
    Tauon = 15,
    AntiTauon = -15,
    TauonNeutrino = 16,
    AntiTauonNeutrino = -16,
    Gamma = 22,
    Pion0 = 111,
    PionPlus = 211,
    PionMinus = -211,
    Kaon0 = 311,
    KaonPlus = 321,
    KaonMinus = -321,
    Neutron = 2112,
    AntiNeutron = -2112,
    Proton = 2212,
    AntiProton = -2212,
    Deuteron = 1000010020,
    Triton = 1000010030,
    Alpha = 1000020040

class PhysicsLabel(Enum):
    Undefined = -1,
    Noise = 0,
    MIPIonization = 1,
    HIPIonization = 2,
    DeltaElectron = 3,
    MichelElectron = 4,
    ElectronShower = 5,
    PositronShower = 6,
    PhotonShower = 7,
    NeutronCaptureGamma474 = 8,
    NeutronCaptureGamma336 = 9,
    NeutronCaptureGamma256 = 10,
    NeutronCaptureGamma118 = 11,
    NeutronCaptureGamma083 = 12,
    NeutronCaptureGamma051 = 13,
    NeutronCaptureGamma016 = 14,
    NeutronCaptureGammaOther = 15,
    Ar39 = 16,
    Ar42 = 17,
    K42 = 18,
    Kr85 = 19,
    Rn222 = 20,
    Po218a = 21,
    Po218b = 22,
    At218a = 23,
    At218b = 24,
    Rn218 = 25,
    Pb214 = 26,
    Bi214a = 27,
    Bi214b = 28,
    Po214 = 29,
    Tl210 = 30,
    Pb210a = 31,
    Pb210b = 32,
    Bi210a = 33,
    Bi210b = 34,
    Po210 = 35,
    NuclearRecoil = 36,
    ElectronRecoil = 37

classification_labels = {
    "source":  {
        -1: "undefined",
        0:  "noise",
        1:  "cosmics",
        2:  "beam",
        3:  "radiological",
        4:  "pns",
        5:  "hepevt",
    },

    "topology": {
        -1: "undefined",
        0:  "noise",
        1:  "blip",
        2:  "track",
        3:  "shower",
    },

    "particle": {
        -1:     "undefined",
        0:      "noise",
        11:     "electron",
        -11:    "positron",
        12:     "electron_neutrino",
        -12:    "anti-electron_neutrino",
        13:     "muon",
        -13:    "anti-muon",
        14:     "muon_neutrino",
        -14:    "anti-muon_neutrino",
        15:     "tauon",
        -15:    "anti-tauon",
        16:     "tauon_neutrino",
        -16:    "anti-tauon_neutrino",
        22:     "gamma",
        111:    "pion0",
        211:    "pion_plus",
        -211:   "pion_minus",
        311:    "kaon0",
        321:    "kaon_plus",
        -321:   "kaon_minus",
        2112:   "neutron",
        -2112:  "anti-neutron",
        2212:   "proton",
        -2212:  "anti-proton",
        1000010020: "deuteron",
        1000010030: "triton",
        1000020040: "alpha"
    },

    "physics": {
        -1: "undefined",
        0:  "noise",
        1:  "mip_ionization",
        2:  "hip_ionization",
        3:  "delta_electron",
        4:  "michel_electron",
        5:  "electron_shower",
        6:  "positron_shower",
        7:  "photon_shower",
        8:  "neutron_capture_gamma_474",
        9:  "neutron_capture_gamma_336",
        10: "neutron_capture_gamma_256",
        11: "neutron_capture_gamma_118",
        12: "neutron_capture_gamma_083",
        13: "neutron_capture_gamma_051",
        14: "neutron_capture_gamma_016",
        15: "neutron_capture_gamma_other",
        16: "ar39",
        17: "ar42",
        18: "k42",
        19: "kr85",
        20: "rn222",
        21: "po218a",
        22: "po218b",
        23: "at218a",
        24: "at218b",
        25: "rn218",
        26: "pb214",
        27: "bi214a",
        28: "bi214b",
        29: "po214",
        30: "tl210",
        31: "pb210a",
        32: "pb210b",
        33: "bi210a",
        34: "bi210b",
        35: "po210",
        36: "nuclear_recoil",
        37: "electron_recoil"
    },

    "hit": {
        0:  "induction",
        1:  "hit",
    }
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
        "bounds": [[0, -50.0],[50.0, 4000.0]], 
        "prior": "uniform", 
        "type": "continuous"
    },
    "m2":   {
        "bounds": [[-4000.0, -100.0],[100.0, 4000.0]], 
        "prior": "uniform", 
        "type": "continuous"
    },
    "m3":   {
        "bounds": [400.0, 4000.0], 
        "prior": "uniform", 
        "type": "continuous"
    },
    "mmu":  {
        "bounds": [[-4000.0, -100.0],[100.0, 4000.0]], 
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
        "bounds": [[-4000.0, -50.0],[50.0, 4000.0]], 
        "prior": "uniform", 
        "type": "continuous"
    },
    "m2":   {
        "bounds": [[-4000.0, -100.0],[100.0, 4000.0]], 
        "prior": "uniform", 
        "type": "continuous"
    },
    "m3":   {
        "bounds": [400.0, 4000.0], 
        "prior": "uniform", 
        "type": "continuous"
    },
    "mmu":  {
        "bounds": [[-4000.0, -100.0],[100.0, 4000.0]], 
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
        "type": "continuous"
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
    'gut_mL1','gut_mL3', 
    'gut_me1','gut_mtau1', 
    'gut_mQ1','gut_mQ3', 
    'gut_mu1','gut_mu3', 
    'gut_md1','gut_md3', 
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
    'weakm_mW', 'weakm_mh','weakm_mH',
    'weakm_mA', 'weakm_mHpm','weakm_m3',
    'weakm_mneut1', 'weakm_mneut2','weakm_mneut3',
    'weakm_mneut4','weakm_mcharg1','weakm_mcharg2',
    'weakm_mdL','weakm_muL','weakm_msL',
    'weakm_mcL','weakm_mb1','weakm_mt1',
    'weakm_meL','weakm_mesneuL','weakm_mmuL',
    'weakm_mmusneuL', 'weakm_mtau1','weakm_mtausneuL',
    'weakm_mdR', 'weakm_muR','weakm_msR',
    'weakm_mcR', 'weakm_mb2','weakm_mt2',
    'weakm_meR', 'weakm_mmuR','weakm_mtau2',
    'weakm_neutmix11','weakm_neutmix12', 
    'weakm_neutmix13','weakm_neutmix14'
]
# column names corresponding to
# computed experimental values.
weak_measurements_columns = [
    'omegah2', 'g-2', 'b->sgamma',
    'b->sgammaSM', 'B+->taunu','Bs->mumu',
    'Ds->taunu','Ds->munu', 'deltarho', 
    'RL23', 'lspmass', 'sigmav',
    'cdmpSI','cdmpSD','cdmnSI','cdmnSD'
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
    'chan10weight','chan10part1','chan10part2','chan10part3','chan10part4'
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
    #parameter: block
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
    "mH1^2":"msoft", 
    "mH2^2":"msoft",
    "muR":  "msoft", 
    "mcR":  "msoft", 
    "mtR":  "msoft", 
    "mdR":  "msoft", 
    "msR":  "msoft",
    "mbR":  "msoft", 
    "meR":  "msoft", 
    "mmuR": "msoft", 
    "mtauR":"msoft", 
    "mqL1": "msoft",
    "mqL2": "msoft", 
    "mqL3": "msoft", 
    "meL":  "msoft", 
    "mmuL": "msoft", 
    "mtauL":"msoft",
    "higgs":"hmix", 
    "Yt":   "yu", 
    "Yb":   "yd", 
    "Ytau": "ye",
}
# incidentally, all the weak
# parameters are in the MASS block
softsusy_weak_parameters = {
    #parameter: block
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
    "~nue_L":"MASS", 
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

    def __init__(self,
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

    def fit_transform(self,
        X
    ):
        results = []
        for projection in self.projections:
            if projection == 'signmu_mgaugino':
                results.append(X[:,1] * np.sign(X[:,4]))
            elif projection == 'signmu_msfermion':
                results.append(X[:,0] * np.sign(X[:,4]))
            elif projection == 'signA0_msfermion':
                results.append(X[:,0] * np.sign(X[:,2]))
            elif projection == 'signmu_tanbeta':
                results.append(X[:,3] * np.sign(X[:,4]))
            elif projection == 'A0_msfermion':
                results.append(X[:,2]/X[:,0])
        results = np.reshape(results, (-1,len(results)))
        return results

