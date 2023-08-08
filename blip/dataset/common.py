"""
"""

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