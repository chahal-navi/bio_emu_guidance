import torch
import pandas as pd
import os
import warnings
from Bio.PDB import PDBParser
from collections import defaultdict
import requests
import csv


# function for making a csv file compatible with downstream denoiser.py handling.
# def membrane_csv(opm_pdb_path):
#     parser = PDBParser(QUIET=True)
#     structure = parser.get_structure(opm_pdb_path)
#     model = structure[0]
#     chain = model[0]
#     for residue in chain:
#         pdb_res = chain[residue]
#         pdb_res.id[]
# ------------------------
# Single-letter to three-letter amino acid codes
ONE_TO_THREE = {
    'A': 'ALA',  # Alanine
    'R': 'ARG',  # Arginine
    'N': 'ASN',  # Asparagine
    'D': 'ASP',  # Aspartic acid
    'C': 'CYS',  # Cysteine
    'Q': 'GLN',  # Glutamine
    'E': 'GLU',  # Glutamic acid
    'G': 'GLY',  # Glycine
    'H': 'HIS',  # Histidine
    'I': 'ILE',  # Isoleucine
    'L': 'LEU',  # Leucine
    'K': 'LYS',  # Lysine
    'M': 'MET',  # Methionine
    'F': 'PHE',  # Phenylalanine
    'P': 'PRO',  # Proline
    'S': 'SER',  # Serine
    'T': 'THR',  # Threonine
    'W': 'TRP',  # Tryptophan
    'Y': 'TYR',  # Tyrosine
    'V': 'VAL',  # Valine
    'X': 'UNK',  # Unknown
}

# Three-letter to single-letter amino acid codes
THREE_TO_ONE = {
    'ALA': 'A',  # Alanine
    'ARG': 'R',  # Arginine
    'ASN': 'N',  # Asparagine
    'ASP': 'D',  # Aspartic acid
    'CYS': 'C',  # Cysteine
    'GLN': 'Q',  # Glutamine
    'GLU': 'E',  # Glutamic acid
    'GLY': 'G',  # Glycine
    'HIS': 'H',  # Histidine
    'ILE': 'I',  # Isoleucine
    'LEU': 'L',  # Leucine
    'LYS': 'K',  # Lysine
    'MET': 'M',  # Methionine
    'PHE': 'F',  # Phenylalanine
    'PRO': 'P',  # Proline
    'SER': 'S',  # Serine
    'THR': 'T',  # Threonine
    'TRP': 'W',  # Tryptophan
    'TYR': 'Y',  # Tyrosine
    'VAL': 'V',  # Valine
    'UNK': 'X',  # Unknown
}
# -------------------------

MODIFIED_TO_STANDARD = {
    # Methionine
    "MSE": "MET",  # selenomethionine
    "FME": "MET",  # N-formylmethionine
    "SME": "MET",  # methionine sulfoxide
    "MHO": "MET",  # methionine sulfone/oxide
    "MED": "MET",  # D-methionine
    # Serine
    "SEP": "SER",  # phosphoserine
    "SAC": "SER",  # N-acetylserine
    "OAS": "SER",  # O-acetylserine
    "SVA": "SER",  # serine vanadate
    "DSN": "SER",  # D-serine
    # Threonine
    "TPO": "THR",  # phosphothreonine
    "BMT": "THR",
    "ALO": "THR",  # allo-threonine
    "DTH": "THR",  # D-threonine
    # Tyrosine
    "PTR": "TYR",  # phosphotyrosine
    "TYS": "TYR",  # sulfotyrosine
    "TPQ": "TYR",  # topaquinone
    "DTY": "TYR",  # D-tyrosine
    # Cysteine
    "CSO": "CYS",  # S-hydroxycysteine
    "CSD": "CYS",  # 3-sulfinoalanine
    "OCS": "CYS",  # cysteic acid
    "CSX": "CYS",
    "CSS": "CYS",
    "CME": "CYS",  # S,S-(2-hydroxyethyl)thiocysteine
    "CAS": "CYS",
    "SCH": "CYS",
    "CYX": "CYS",  # disulfide-bonded (Amber naming)
    "CYM": "CYS",  # deprotonated (Amber naming)
    "DCY": "CYS",  # D-cysteine
    # Lysine
    "MLY": "LYS",  # N-dimethyllysine
    "MLZ": "LYS",  # N-methyllysine
    "M3L": "LYS",  # N-trimethyllysine
    "ALY": "LYS",  # N-acetyllysine
    "KCX": "LYS",  # carboxylysine
    "LLP": "LYS",  # PLP-linked lysine
    "LYZ": "LYS",
    "LYN": "LYS",  # neutral (Amber naming)
    "DLY": "LYS",  # D-lysine
    # Histidine
    "HIC": "HIS",  # 4-methylhistidine
    "NEP": "HIS",  # N1-phosphohistidine
    "HID": "HIS",  # protonation states (Amber naming)
    "HIE": "HIS",
    "HIP": "HIS",
    "DHI": "HIS",  # D-histidine
    # Proline
    "HYP": "PRO",  # hydroxyproline
    "DPR": "PRO",  # D-proline
    # Glutamate / glutamine
    "CGU": "GLU",  # gamma-carboxyglutamate
    "GLH": "GLU",  # protonated (Amber naming)
    "DGL": "GLU",  # D-glutamate
    "PCA": "GLN",  # pyroglutamic acid
    "DGN": "GLN",  # D-glutamine
    # Aspartate / asparagine
    "ASH": "ASP",  # protonated (Amber naming)
    "PHD": "ASP",  # phosphoaspartate
    "DAS": "ASP",  # D-aspartate
    "MEN": "ASN",  # N-methylasparagine
    "DSG": "ASN",  # D-asparagine
    # Arginine
    "AGM": "ARG",  # agmatine-like / methylated arginine
    "DAR": "ARG",  # D-arginine
    # Tryptophan
    "DTR": "TRP",  # D-tryptophan
    # Phenylalanine
    "DPN": "PHE",  # D-phenylalanine
    # Alanine / glycine
    "AIB": "ALA",  # alpha-aminoisobutyric acid
    "DAL": "ALA",  # D-alanine
    "SAR": "GLY",  # sarcosine
    "GL3": "GLY",  # thioglycine
    # Leucine / isoleucine / valine
    "NLE": "LEU",  # norleucine
    "MLE": "LEU",  # N-methylleucine
    "DLE": "LEU",  # D-leucine
    "DIL": "ILE",  # D-isoleucine
    "NVA": "VAL",  # norvaline
    "MVA": "VAL",  # N-methylvaline
    "DVA": "VAL",  # D-valine
}


def get_pdb_chain(full_id):
    # Parsing logic for each unique PDB_ID(csv['PDB_ID']) in the csv dataset file for obtaining.
    full_id = full_id.strip().upper() 
    pdb_code = full_id[:4]
    if len(full_id) > 4:
        chain = full_id[4:]
    else:
        return pdb_code, None
    return pdb_code, chain

def fetch_pdb(pdb_id, pdb_dir):
    
    candidates = [
        f"{pdb_id}_opm.pdb",
        f"{pdb_id}.pdb",
        f"{pdb_id.lower()}.pdb",
        f"{pdb_id.upper()}.pdb",
    ]
    for name in candidates:
        path = os.path.join(pdb_dir, name)
        if os.path.isfile(path):
            return path 
        else: 
            return None
    return None

def build_struct_dict(pdb_path, pdb_code):
    parser = PDBParser(PERMISSIVE=True, QUIET=True)
    structure = parser.get_structure(pdb_code, pdb_path)
    model = structure[0]
    target_chain = next(iter(model))
    # residue.id = (hetero_flag, sequence_number, insertion_code)
    chain_coords = {}
    for residue in target_chain:
        # if residue.id[0] != ' ':
        #     continue  --- this would lead to exclusion of modified residues before MODIFIED_TO_STANDARD could convert it to a standard one.
        # if residue.resname.upper().strip() not in THREE_TO_ONE:
        #     res_sym = MODIFIED_TO_STANDARD[residue.resname.upper().strip()]
        #     res_sym = THREE_TO_ONE[res_sym]
        # res_num = residue.id[1]
        # chain_coords[(res_num, res_sym)] = []
        # for atom in residue:
        #     if 'CA' not in residue:
        #         for i in range(3):
        #             chain_coords[(res_num, res_sym)][i] = "nan"   
            
        #     chain_coords[(res_num, res_sym)] = residue['CA'].get_coord()
        # Above: Alternate implementation of setting skipped residue coordinates to "nan". Commented because downstream PDB sequence only requires valid residues for kabsch alignment.

        res_name = residue.resname.upper().strip()
        if res_name in MODIFIED_TO_STANDARD:
            res_name = MODIFIED_TO_STANDARD[res_name]

        if res_name not in THREE_TO_ONE:
            continue

        if 'CA' not in residue:
            continue
        res_sym = THREE_TO_ONE[res_name]
        res_num = residue.id[1]
        chain_coords[(res_num, res_sym)] = residue['CA'].get_coord()

    pdb_seq = ''
    for (res_num, res_sym), coord_data in chain_coords.items():
        pdb_seq += res_sym
    return chain_coords, pdb_seq

# Output shape == {(res_num, "res_sym"):[x,y,z]} where [x,y,z] are the C-alpha coordinates.
# Making a function for converting this dictionary into a 2D dictionary for easier transmission to a csv file with shape : {(res_num, "res_sym"): membrane_distance}/
# Below: commented function because accessing membrane_distance for OPM files is trivial and does not need a separate function.

# def make_dist_dict(chain_coords):
#     dist_dict = defaultdict(dict)
#     for (res_num, res_sym), atom_coords in chain_coords.items():
#         z_coords = 0
#         for atom in atom_coords.items():
#             z_coords += atom_coords[atom][-1]
#         avg_z_coords = z_coords/len(atom_coords) 
#         memb_dist = avg_z_coords + 15.40
#         dist_dict[(res_num, res_sym)] = memb_dist

#     dist_dict = dict(dist_dict)
#     df = pd.DataFrame(dist_dict)
#     df.to_csv('membrane_distance.csv', index=False)
#     return dist_dict

# a function for extracting the SEQRES sequence for BioEmu input.
def get_seqres(opm_pdb_path):
    sequence = ''
    with open(opm_pdb_path, "r") as f:
        seqres_lines = [line.strip() for line in f if line.startswith("SEQRES")]
    for line in seqres_lines:
        sequence += line
    return sequence.strip().upper() 

# a function to fall back for the SEQRES sequence since OPM files don't usually have SEQRES
# def get_seqres_rcsb(pdb_id):
#     pdb_id = pdb_id.strip()
#     if len(pdb_id)>=4:
#         pdb_id, _ = get_pdb_chain(pdb_id)
#     pdb_id = pdb_id.lower()
#     # AI-generated -----
#     url = f"https://rcsb.org{pdb_id}"

#     response = requests.get(url)
#     seqres_sequence = ""
#     if response.status_code == 200:
#         data = response.json()
        
#         # 1. Locate the very first polymer entity in the structure's structural composition
#         first_entity_id = data['rcsb_entry_container_identifiers']['polymer_entity_ids'][0]
        
#         # 2. Query that specific entity to get the raw deposited SEQRES data
#         entity_url = f"https://rcsb.org{pdb_id}/{first_entity_id}"
#         entity_response = requests.get(entity_url)
        
#         if entity_response.status_code == 200:
#             entity_data = entity_response.json()
            
#             # This is the literal equivalent of the legacy SEQRES text record
#             seqres_sequence = entity_data['entity_poly']['pdbx_seq_one_letter_code_can']
            
#             # Find which author chain ID(s) this sequence applies to
#             author_chains = entity_data['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
#     # ------
#     return seqres_sequence

# AI-generated ---------------------
def get_seqres_rcsb(full_id):

    pdb_code, chain = get_pdb_chain(full_id)
    pdb_code = pdb_code.lower()

    # 1. Ask the entry which polymer entities it contains, e.g. ["1", "2"]
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}"
    response = requests.get(entry_url, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"RCSB entry request failed for {pdb_code}: HTTP {response.status_code}")
    entry_data = response.json()
    entity_ids = entry_data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]

    # 2. Go through the entities and find the one that contains our chain
    for entity_id in entity_ids:
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_code}/{entity_id}"
        response = requests.get(entity_url, timeout=30)
        if response.status_code != 200:
            raise ValueError(f"RCSB entity request failed for {pdb_code}/{entity_id}: HTTP {response.status_code}")
        entity_data = response.json()

        # Skip DNA / RNA entities
        if entity_data["entity_poly"]["type"] != "polypeptide(L)":
            continue

        chains_of_this_entity = entity_data["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]

        # If no chain was given, take the first protein entity
        if chain is None or chain in chains_of_this_entity:
            seqres_sequence = entity_data["entity_poly"]["pdbx_seq_one_letter_code_can"]
            seqres_sequence = seqres_sequence.replace("\n", "").strip().upper()
            return seqres_sequence

    raise ValueError(f"Chain {chain} not found among the protein entities of {pdb_code}.")
# -----------

import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer


from Bio import Align
from Bio.Align import substitution_matrices



# function to instantiate PairwiseAligner() class of BioPython. (AI-generated)
def make_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    # Do not penalise overhangs at the ends: the PDB usually misses termini
    # or the query is a sub-construct. Must be set AFTER the gap scores above.
    aligner.end_gap_score = 0
    return aligner
# ---------


# Function to get the alignment of the PDB sequence and the SEQRES sequence.
def align_seqres2pdb(seqres_sequence, pdb_seq):
    alignment = make_aligner().align(seqres_sequence, pdb_seq)[0]
    seqres_blocks, pdb_blocks = alignment.aligned
    map = {}
    for (s_start, s_end), (p_start, p_end) in zip(seqres_blocks, pdb_blocks):
        for s, p in zip(range(s_start, s_end), range(p_start, p_end)):
            map[(seqres_sequence[s], s)] = (pdb_seq[p], p)
    if len(map)<=len(seqres_sequence)/2:
        raise ValueError("seqres and pdb_sequence are very different")
    return map     # Shape: Eg. alignment of ACEGHJ and AEGHKJ - {(A,0): (A,0), (E,2): (E,1), (G,3): (G,2), ....}

# Function for writing the final csv with the following fields: 
# 1. resid - for (res_sym, res_id), pdb_res map.items(): res_id
# 2. resname - same as above
# 3. alignment_mask - list(seqres_sequence.strip().split()) - list((res_sym for (res_sym, res_id), pdb_res in map.items).strip().split())
# 4. pdb_resnum - (pdb_res, pdb_resnum) = map[(res_sym, res_id)]
# 5. x_coord - chain_coords[(pdb_res, pdb_resnum)][0]
# 6. y_coord - chain_coords[(pdb_res, pdb_resnum)][1]
# 7. z_coord of c_alpha - chain_coords[(pdb_res, pdb_resnum)][2]
# 8. membrane_interacting mask - if z_coord >= -20 && z_coord <=20

CSV_HEADER = [
    "resid", "resname", "has_ref", "pdb_resnum",
    "ref_x", "ref_y", "ref_z", "membrane_interacting",
]

def write_membrane_csv(map, chain_coords, seqres_sequence, out_csv, z_cutoff):
    pdb_keys = list(chain_coords.keys())

    rows = []
    for idx, res in enumerate(seqres_sequence):
        res_id = idx
        res_name = res.upper().strip()   # ONE_LETTER_CODE
        if (res_name, res_id) not in map.keys():
            has_ref = False
            pdb_resnum = None
            ref_x, ref_y, ref_z = "nan", "nan", "nan"
            membrane_interacting = False
            rows.append([res_id, res_name, has_ref, pdb_resnum, ref_x, ref_y, ref_z, membrane_interacting])
            continue
        
        has_ref = True
        pdb_res, pdb_resnum = map[(res_name, res_id)]
        pdb_key = pdb_keys[pdb_resnum]
        pdb_resnum = pdb_key[0]
        ref_x, ref_y, ref_z = chain_coords[pdb_key][0], chain_coords[pdb_key][1], chain_coords[pdb_key][2]
        ref_x, ref_y, ref_z = round(ref_x, 3), round(ref_y, 3), round(ref_z, 3) 
        membrane_interacting = (ref_z >= -z_cutoff) and (ref_z <= z_cutoff)
        rows.append([res_id, res_name, has_ref, pdb_resnum, ref_x, ref_y, ref_z, bool(membrane_interacting)])

    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    return out_csv

# Function to calculate the rotation and translation matrices for transforming to reference structure(AI_generated learning from the BioPython implementation - SVDSuperimposer.py) --------- 
def kabsch_rot_tran(mobile_A, reference_A):
    """
    Superimpose mobile_A onto reference_A (both (n, 3) numpy, same row order).
    Returns rot, tran, rms such that  mobile_A @ rot + tran  ~=  reference_A.
    (SVDSuperimposer uses right-multiplication.)
    """
    sup = SVDSuperimposer()
    sup.set(reference_A, mobile_A)
    sup.run()
    rot, tran = sup.get_rotran()
    return rot, tran, sup.get_rms()  # Usage in custom_potentials.py: x0_aligned = x0_hat_per_graph @ rot + tran 
# --------
