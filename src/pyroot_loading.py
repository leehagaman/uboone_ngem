import os
from tqdm import tqdm

from src.file_locations import tmp_dir

import ROOT

def get_rw_sys_weights_dic(
    file_path: str,
    tree_path: str = "nuselection/NeutrinoSelectionFilter",
    branch_name: str = "weights",
    max_entries: int = -1):
    """
    Gets systematic weights from Pandora Tree using PyROOT.
    Returns a list of dictionaries, one per event.
    
    Parameters:
    -----------
    file_path : str
        Path to the ROOT file
    tree_path : str
        Path to the tree within the ROOT file (default: "nuselection/NeutrinoSelectionFilter")
    branch_name : str
        Name of the branch containing weights (default: "weights")
    max_entries : int
        Maximum number of entries to process. If -1, process all entries (default: -1)
    
    Returns:
    --------
    list of dict
        List of dictionaries, one per event. Each dictionary maps systematic names to lists of weights.
    """

    ROOT.gSystem.SetBuildDir(tmp_dir, True)
    
    # Ensure ROOT dictionary is generated for the map type
    ROOT.gInterpreter.GenerateDictionary("map<string,vector<double>>", "map;string;vector")
    
    # Open the ROOT file
    root_file = ROOT.TFile.Open(os.path.abspath(file_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Cannot open file: {file_path}")
    
    # Get the tree
    tree = root_file.Get(tree_path)
    if not tree:
        raise RuntimeError(f"Tree not found at path: {tree_path}")
    # Check if it's actually a TTree by checking the class name
    if tree.ClassName() != "TTree":
        raise RuntimeError(f"Object at path {tree_path} is not a TTree (got {tree.ClassName()})")
    
    # Get the branch
    branch = tree.GetBranch(branch_name)
    if not branch:
        raise RuntimeError(f"Branch not found: {branch_name}")
    
    # Set up branch address - The branch stores a pointer to std::map<string, vector<double>>
    # In PyROOT, we create the map object and pass it to SetBranchAddress
    # PyROOT will handle the pointer conversion automatically
    weight_map = ROOT.std.map('string', ROOT.std.vector('double'))()
    tree.SetBranchAddress(branch_name, weight_map)
    
    # Determine number of entries to process
    nentries = tree.GetEntries()
    if max_entries >= 0 and max_entries < nentries:
        nentries = max_entries
    
    rows = []
    
    # Process each event
    for i in tqdm(range(nentries), total=nentries, desc="Loading systematic weights"):
        entry = tree.GetEntry(i)
        if entry <= 0:
            rows.append({})
            continue
        
        # The weight_map object is updated by GetEntry
        # Convert to Python dictionary
        event_dict = {}
        
        # Iterate over the map
        # In PyROOT, std::map can be iterated directly
        # If the map is empty (or pointer was null), this loop simply won't execute
        for key_pair in weight_map:
            key = key_pair.first
            weights_vec = key_pair.second
            
            # Convert vector<double> to Python list
            weights_list = []
            for j in range(weights_vec.size()):
                weights_list.append(weights_vec[j])
            
            event_dict[str(key)] = weights_list
        
        rows.append(event_dict)

    root_file.Close()
    
    return rows
    
