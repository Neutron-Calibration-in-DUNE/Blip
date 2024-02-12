
"""
nArInelastic analyzer code.
"""
from blip.analysis.generic_analyzer import GenericAnalyzer

generic_config = {
    "no_params":    "no_values"
}


class nArInelasticAnalyzer(GenericAnalyzer):
    """
    This analyzer does several things:
        1. Generate nAr-inelastic plots from larnd-sim.
            a.
            b.
            c.
            d. 
        2. Perform the rest of the analysis chain, after
           segmentation and other reconstruction tasks.
           a.
           b.
           c.
           d.
    """
    def __init__(
        self,
        name:   str,
        config: dict = {},
        meta:   dict = {},
    ):
        super(nArInelasticAnalyzer, self).__init__(
            name, config, meta
        )

        self.mc_data = {
            "lar_fv":   [],
            "p_vtx":    [],
            "nu_vtx":   [],
            "proton_total_energy":  [],
            "proton_vis_energy":    [],
            "proton_length":        [],
            "proton_start_momentum":    [],
            "proton_end_momentum":      [],
            "parent_total_energy":      [],
            "parent_length":    [],
            "parent_start_momentum":    [],
            "parent_end_momentum":      [],
            "nu_proton_dt": [],
            "nu_proton_distance":   [],
            "parent_pdg":   [],
            "grandparent_pdg":  [],
            "primary_pdg":  [],
            "primary_length":   [],
        }

        self.process_config()

    def process_config(self):
        if "analyzer_modes" in self.config:
            self.analyzer_modes = self.config["analyzer_modes"]

    def set_device(
        self,
        device
    ):
        self.device = device

    def analyze_event(
        self,
        event
    ):
        for analyzer_mode in self.analyzer_modes:
            if analyzer_mode == "mc_truth":
                self.analyze_event_mc_truth(event)
            elif analyzer_mode == "analysis_chain":
                self.analyze_event_analysis_chain(event)

    def analyze_event_mc_truth(
        self,
        event
    ):
        # copy what Brooke is doing in her "process_file" function
        data, truth = event
        particles = truth["particles"]

        proton_track_ids = particles["traj_id"][(particles["pdg_id"] == 2212)]
        print(proton_track_ids)

        self.logger.info("processing mc_truth")

    def analyze_event_analysis_chain(
        self,
        event
    ):
        """
        This will end with a plot of nAr-inelastic cross section
        as a function of energy.
        """
        pass

    def analyze_events(
        self
    ):
        self.logger.info("analyze_events")


"""
def process_file(sim_h5):
unique_spill = get_unique_spills(sim_h5)
d=dict()
for spill_id in unique_spill:
    ghdr, gstack, traj, vert, seg = get_spill_data(sim_h5, spill_id)
    traj_proton_mask = traj['pdg_id']==2212
    proton_traj = traj[traj_proton_mask]

    for pt in proton_traj:

        # REQUIRE proton contained in 2x2 active LAr
        proton_start=pt['xyz_start']
        if fiducialized_vertex(proton_start)==False: continue
        if fiducialized_vertex(pt['xyz_end'])==False: continue

        # is nu vertex contained in 2x2 active LAr?
        vert_mask = vert['vertex_id']==pt['vertex_id']
        nu_vert = vert[vert_mask]
        vert_loc = [nu_vert['x_vert'],nu_vert['y_vert'],nu_vert['z_vert']]
        vert_loc = np_array_of_array_to_flat_list(vert_loc)
        lar_fv = 1
        if fiducialized_vertex(vert_loc)==False: lar_fv = 0

        # Find proton parent PDG
        parent_mask = (traj['traj_id']==pt['parent_id'])
        if sum(parent_mask) == 0: continue
        parent_pdg=traj[parent_mask]['pdg_id']
        if pt['parent_id']==-1:
            ghdr_mask=ghdr['vertex_id']==pt['vertex_id']
            parent_pdg=ghdr[ghdr_mask]['nu_pdg']

        # Find proton grandparent PDG 
        grandparent_mask = (traj['traj_id']==traj[parent_mask]['parent_id'])
        grandparent_trackid = traj[grandparent_mask]['traj_id']
        grandparent_pdg = traj[grandparent_mask]['pdg_id']
        if grandparent_trackid.size>0:
            if grandparent_trackid==-1:
                ghdr_mask=ghdr['vertex_id']==pt['vertex_id']
                grandparent_pdg=ghdr[ghdr_mask]['nu_pdg']
        grandparent_pdg=list(grandparent_pdg)
        if len(grandparent_pdg)==0: grandparent_pdg=[0]
        
        if parent_pdg[0] not in [12,14,16,-12,-14,-16]:
            parent_total_energy = float(list(traj[parent_mask]['E_start'])[0])
            parent_length = float(total_edep_length(traj[parent_mask]['traj_id'], traj, seg))
            parent_start_momentum = float(three_momentum(traj[parent_mask]['pxyz_start'][0]))
            parent_end_momentum = float(three_momentum(traj[parent_mask]['pxyz_end'][0]))
        else:
            parent_total_energy = float(-1) 
            parent_length = float(-1)
            parent_start_momentum = float(-1)
            parent_end_momentum = float(-1)
            
        gstack_mask = gstack['vertex_id']==pt['vertex_id']
        gstack_traj_id = gstack[gstack_mask]['traj_id']
        primary_length=[]; primary_pdg=[]
        for gid in gstack_traj_id:
            primary_mask = traj['traj_id']==gid
            primary_start = traj[primary_mask]['xyz_start']
            primary_end = traj[primary_mask]['xyz_end']
            p_pdg = traj[primary_mask]['pdg_id']
            if len(p_pdg)==1:
                primary_pdg.append(int(p_pdg[0]))
                dis = np.sqrt( (primary_start[0][0]-primary_end[0][0])**2+
                                (primary_start[0][1]-primary_end[0][1])**2+
                                (primary_start[0][2]-primary_end[0][2])**2)
                primary_length.append(float(dis))
                    
        p_start = proton_start.tolist()
        p_vtx = []
        for i in p_start: p_vtx.append(float(i))
        
        nu_vtx=[]
        for i in vert_loc: nu_vtx.append(float(i))
        d[(spill_id, pt['vertex_id'], pt['traj_id'])]=dict(
            lar_fv=int(lar_fv),
            
            p_vtx=p_vtx,
            nu_vtx=nu_vtx,
            
            proton_total_energy = float(pt['E_start']),
            proton_vis_energy = float(total_edep_charged_e(pt['traj_id'], traj, seg)),
            proton_length = float(total_edep_length(pt['traj_id'], traj, seg)),
            proton_start_momentum = float(three_momentum(pt['pxyz_start'])),
            proton_end_momentum = float(three_momentum(pt['pxyz_end'])),
            
            parent_total_energy = parent_total_energy, 
            parent_length = parent_length, 
            parent_start_momentum = parent_start_momentum, 
            parent_end_momentum = parent_end_momentum, 
            
            nu_proton_dt = float(pt['t_start']) - float(nu_vert['t_vert'][0]),
            nu_proton_distance = float(np.sqrt( (proton_start[0]-vert_loc[0])**2+
                                        (proton_start[1]-vert_loc[1])**2+
                                        (proton_start[2]-vert_loc[2])**2 )),
            
            parent_pdg=int(parent_pdg[0]),
            grandparent_pdg=int(grandparent_pdg[0]),
            primary_pdg=primary_pdg,                
            primary_length=primary_length
        )
return d
"""