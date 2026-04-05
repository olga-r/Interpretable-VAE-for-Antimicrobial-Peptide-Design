import requests
import re
import math
import numpy as np
import pandas as pd
import time
from copy import deepcopy

# aa mass in Da
AA_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.18,
    "L": 131.18, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15
}
WATER_MASS = 18.015

def peptide_mw(seq: str) -> float:
    """
    Peptide molecular weight in kDa
    """
    seq = seq.strip().upper()
    masses = [AA_MASS[aa] for aa in seq if aa in AA_MASS]
    if len(masses) != len(seq):
        return np.nan
    return (sum(masses) - (len(seq) - 1) * WATER_MASS)/1000


#Chou–Fasman parameters
helix_propensity = {
    'A': 1.45, 'C': 0.77, 'D': 0.98, 'E': 1.53,
    'F': 1.12, 'G': 0.53, 'H': 1.00, 'I': 1.00,
    'K': 1.23, 'L': 1.34, 'M': 1.20, 'N': 0.73,
    'P': 0.59, 'Q': 1.17, 'R': 0.79, 'S': 0.79,
    'T': 0.82, 'V': 1.14, 'W': 1.14, 'Y': 0.61
}

def calc_helix_propensity(seq):
    values = [helix_propensity.get(aa, 0) for aa in seq]
    return sum(values) / len(values)

def geometric_mean(a: float, b: float) -> float:
    return math.sqrt(a * b)


######Functions for DBAASP parsing############
def parse_mic_string(s: str):
    """
    Returns a dict with:
    - low
    - high
    - mean
    - kind
    - usable_for_log
    """
    if pd.isna(s):
        return {"low": np.nan, "high": np.nan, "mean": np.nan,
                "kind": "missing", "usable_for_log": False}

    s = str(s).strip().replace(" ", "")

    # 5+/-1
    m = re.fullmatch(r'(\d*\.?\d+)\±([\d\.]+)', s)
    if m:
        center = float(m.group(1))
        err = float(m.group(2))
        low = max(center - err, 1e-12)
        high = center + err
        mean = geometric_mean(low, high)
        return {"low": low, "high": high, "mean": mean,
                "kind": "plusminus", "usable_for_log": True}

    # 1-10
    m = re.fullmatch(r'(\d*\.?\d+)-(\d*\.?\d+)', s)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        if low > high:
            low, high = high, low
        mean = geometric_mean(low, high)
        return {"low": low, "high": high, "mean": mean,
                "kind": "range", "usable_for_log": True}

    # =4
    m = re.fullmatch(r'=?(\\d*\\.?\\d+)', s)
    if m:
        val = float(m.group(1))
        return {"low": val, "high": val, "mean": val,
                "kind": "exact", "usable_for_log": True}

    # 4
    m = re.fullmatch(r'(\d*\.?\d+)', s)
    if m:
        val = float(m.group(1))
        return {"low": val, "high": val, "mean": val,
                "kind": "exact", "usable_for_log": True}

    # >=2 or >2
    m = re.fullmatch(r'(>=|>)(\d*\.?\d+)', s)
    if m:
        low = float(m.group(2))
        return {"low": low, "high": np.nan, "mean": np.nan,
                "kind": "lower_bound", "usable_for_log": False}

    # <=5 or <6
    m = re.fullmatch(r'(<=|<)(\d*\.?\d+)', s)
    if m:
        high = float(m.group(2))
        return {"low": np.nan, "high": high, "mean": np.nan,
                "kind": "upper_bound", "usable_for_log": False}

    return {"low": np.nan, "high": np.nan, "mean": np.nan,
            "kind": "unparsed", "usable_for_log": False}


def convert_to_uM(value, unit, seq=None):
    """
    Convert a concentration to uM.
    unit: 'uM' or 'ug/ml'
    """
    if pd.isna(value):
        return np.nan

    unit = str(unit).strip().lower().replace("μ", "u").replace("µ", "u")

    if unit == "um":
        return float(value)

    if unit == "ug/ml":
        mw = peptide_mw(seq)
        if pd.isna(mw) or mw <= 0:
            return np.nan
        return float(value) / mw

    return np.nan


def derive_binary_label(low, high, threshold):
    """
    Returns:
    1 if definitely <= threshold
    0 if definitely > threshold
    NaN otherwise
    """
    if not pd.isna(high) and high <= threshold:
        return 1
    if not pd.isna(low) and low > threshold:
        return 0
    return np.nan


def derive_binary_label_lt(low, high, threshold):
    """
    For condition MIC < threshold.
    Returns:
    1 if definitely < threshold
    0 if definitely >= threshold
    NaN otherwise
    """
    if not pd.isna(high) and high < threshold:
        return 1
    if not pd.isna(low) and low >= threshold:
        return 0
    return np.nan


def process_activity(activity, seq, unit):
    mic = activity.get('concentration')
    parsed = parse_mic_string(mic)

    low_uM = convert_to_uM(parsed["low"], unit, seq)
    high_uM = convert_to_uM(parsed["high"], unit, seq)
    mean_uM = convert_to_uM(parsed["mean"], unit, seq)

    log10_mic = np.log10(mean_uM) if (not pd.isna(mean_uM) and mean_uM > 0) else np.nan

    strong_active_25 = derive_binary_label(low_uM, high_uM, 25.0)
    active_100 = derive_binary_label_lt(low_uM, high_uM, 100.0)

    return {
        "mic_kind": parsed["kind"],
        "mic_low_raw": parsed["low"],
        "mic_high_raw": parsed["high"],
        "mic_mean_raw": parsed["mean"],
        "mic_low_uM": low_uM,
        "mic_high_uM": high_uM,
        "mic_mean_uM": mean_uM,
        "log10_mic_uM": log10_mic,
        "strong_active_25": strong_active_25,
        "active_100": active_100,
    }

def assign_bond_class(t, c):
    """
    t - type name
    c - cycletype name
    """

    if t == "DSB":
        return "disulfide"

    if t == "AMD":
        if c == "DKP":
            return "dkp"
        return "amide_cycle"

    if c in ["LAN", "MeLAN"]:
        return "lanthionine"

    if t == "TIE":
        return "thioether"

    return "exotic"
######################################

URL = "https://dbaasp.org/peptides"

all_peptides = []
limit = 1000
offset = 0

headers = {
    'Accept': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
    'User-Agent': 'Mozilla/5.0'
}

while True:
    params = {
        'offset': offset,
        'limit': limit,
        'targetSpecies.value': 'Escherichia coli',
        'complexity.value': 'monomer',
    }

    print(f"Загрузка: offset {offset}...")
    response = requests.get(URL, params=params, headers=headers)

    if response.status_code != 200:
        break

    data = response.json()
    batch = data.get('data', [])

    if not batch:
        break

    all_peptides.extend(batch)
    offset += limit

    time.sleep(0.5)

print(f"All peptides: {len(all_peptides)}")


final_dataset = []
for p in all_peptides[0:]:
    p_id = p.get('id')
    detail_url = f"https://dbaasp.org/peptides/{p_id}"

    try:
        detail_resp = requests.get(detail_url)
        if detail_resp.status_code == 200:
            full_data = detail_resp.json()

            sequence = full_data.get('sequence').upper()
            seq_len = full_data.get('sequenceLength')
            nterminus, cterminus = None, None
            if full_data.get('nTerminus'):
                nterminus = full_data.get('nTerminus')["name"]
            if full_data.get('cTerminus'):
                cterminus = full_data.get('cTerminus')["name"]
            if ("X" in sequence) or seq_len > 30:
                continue
            if cterminus and cterminus != "AMD":
                continue
            if nterminus and nterminus != "ACT":
                continue
            if full_data['interchainBonds']:
                continue

            bond_class = set()
            if full_data['intrachainBonds']:
                for entry in full_data['intrachainBonds']:
                    t = entry["type"]["name"]
                    c = entry["cycleType"]["name"] if entry["cycleType"] else None
                    bond_class.add(assign_bond_class(t, c))
            skip_bonds = set(["exotic", "lanthionine", "thioether", "dkp"])

            if bond_class.intersection(skip_bonds):
                continue
            if len(bond_class)> 1:
                bond_class = "mixed"
            elif len(bond_class) == 1:
                bond_class = bond_class.pop()
            else:
                bond_class = "none"


            activities = full_data.get('targetActivities', [])

            n_total_records = 0
            n_generic_records = 0
            has_strain_specific = False
            mic_generic = []
            all_dict = None
            for act in activities:
                species = act.get('targetSpecies', {}).get('name', '')
                value = act.get('activityMeasureValue', "")
                if not act.get('unit'):
                    continue
                unit =  act.get('unit').get('name')

                if species == "Escherichia coli"  and value == "MIC":
                    n_generic_records += 1
                    n_total_records += 1
                    mic_dict = process_activity(act, sequence, unit)
                    mic_generic.append(mic_dict)
                    pdb = ""
                    if  full_data["pdbs"]:
                        pdb = full_data.get('pdbs')[0].get('name')

                    all_dict = {
                                'id': p_id,
                                'sequence': sequence,
                                'sequenceLength': seq_len,
                                'nTerminus': nterminus,
                                'cTerminus': cterminus,
                                'pdb': pdb,
                                'target': species,
                                'bond_class': bond_class
                            }
                    properties = full_data['physicoChemicalProperties']
                    if properties:
                        for prop in properties:
                            if prop["name"] == "Net Charge":
                                all_dict[prop["name"]] = float(prop["value"])
                                all_dict["Normalized charge"] = float(prop["value"]) /seq_len
                            elif prop["name"] != "ID":
                                all_dict[prop["name"]] = float(prop["value"])

                    all_dict["helix_propensity_normalized"] = calc_helix_propensity(sequence)

                elif  species.startswith("Escherichia coli"):
                    n_total_records += 1
                    has_strain_specific = True
            #agregation
            if all_dict:
                all_dict["n_generic_records"] = n_generic_records
                all_dict["n_total_records"] = n_total_records
                all_dict["has_strain_specific"] = has_strain_specific

                for act in mic_generic:
                    temp_dict = deepcopy(all_dict)
                    temp_dict.update(act)
                    final_dataset.append(temp_dict)

            time.sleep(0.1)

    except Exception as e:
        print(f"Error while downloading ID {p_id}: {e}")


print(f"Selected peptides: {len(final_dataset)}")

df = pd.DataFrame(final_dataset)
df.to_csv("dbaasp_full_ecoli.csv", index=False)

