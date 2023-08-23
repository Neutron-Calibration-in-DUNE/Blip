"""
"""
from enum import Enum

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
    Kr85 = 18,
    Rn222 = 19,
    NuclearRecoil = 20,
    ElectronRecoil = 21

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
        18: "kr85",
        19: "rn222",
        20: "nuclear_recoil",
        21: "electron_recoil"
    }
}