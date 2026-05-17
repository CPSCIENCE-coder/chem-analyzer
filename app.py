from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.inchi import MolToInchiKey
from concurrent.futures import ThreadPoolExecutor
import requests
import urllib.parse

app = Flask(__name__)

BASE_PKA_SMARTS = [
    ("NC(=N)N",                      12.5, "Guanidine"),
    ("NC(=N)",                       11.6, "Amidine"),
    ("N1CCCC1",                      11.3, "Pyrrolidine"),
    ("N1CCCCC1",                     11.0, "Piperidine"),
    ("[NH2][CX4;!$(C=O)]",           10.6, "Primary aliphatic amine"),
    ("[NH;X3]([CX4])[CX4;!$(C=O)]", 10.3, "Secondary aliphatic amine"),
    ("[NX3;H0]([CX4])([CX4])[CX4]",  9.7, "Tertiary aliphatic amine"),
    ("N1CCNCC1",                      9.8, "Piperazine"),
    ("N1CCOCC1",                      8.4, "Morpholine"),
    ("[nH]1ccnc1",                    7.0, "Imidazole"),
    ("[nH]1ccc2ccccc21",              5.5, "Benzimidazole"),
    ("n1ccccc1",                      5.2, "Pyridine"),
    ("[NH2]c",                        4.5, "Aromatic amine"),
    ("n1ccncc1",                      3.5, "Pyrimidine"),
    ("[NH]c",                         3.5, "Secondary aromatic amine"),
]

ACIDIC_PKA_SMARTS = [
    ("S(=O)(=O)[OH]",  1.5,  "Sulfonic acid"),
    ("C(=O)[OH]",      4.2,  "Carboxylic acid"),
    ("n1[nH]nnn1",     4.9,  "Tetrazole"),
    ("c[OH]",          9.9,  "Phenol"),
    ("[SH]",          10.5,  "Thiol"),
    ("S(=O)(=O)[NH]", 10.1,  "Sulfonamide N-H"),
]


# ── Chemistry helpers ──────────────────────────────────────────────────────

def mol_to_svg(mol, highlight_atoms=None, size=(480, 300)):
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    drawer.drawOptions().padding = 0.13
    if highlight_atoms:
        atom_set = set(highlight_atoms)
        a_colors = {i: (1.0, 0.50, 0.0) for i in atom_set}
        a_radii  = {i: 0.50 for i in atom_set}
        b_idxs, b_colors = [], {}
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if u in atom_set and v in atom_set:
                bid = bond.GetIdx()
                b_idxs.append(bid)
                b_colors[bid] = (1.0, 0.50, 0.0)
        drawer.DrawMolecule(mol,
                            highlightAtoms=list(atom_set),
                            highlightAtomColors=a_colors,
                            highlightBonds=b_idxs,
                            highlightBondColors=b_colors,
                            highlightAtomRadii=a_radii)
    else:
        drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def calculate_logp(mol):
    return round(MolLogP(mol), 2)


def predict_base_pka(mol):
    claimed, hits = set(), []
    for smarts, pka, name in BASE_PKA_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat):
            key = match[0]
            if key not in claimed:
                claimed.add(key)
                hits.append({"pka": pka, "group": name, "atoms": list(match)})
    if not hits:
        return None, None, [], []
    hits.sort(key=lambda x: x["pka"], reverse=True)
    top = hits[0]
    return top["pka"], top["group"], [{"pka": h["pka"], "group": h["group"]} for h in hits], top["atoms"]


def predict_acidic_pka(mol):
    claimed, hits = set(), []
    for smarts, pka, name in ACIDIC_PKA_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat):
            key = match[0]
            if key not in claimed:
                claimed.add(key)
                hits.append({"pka": pka, "group": name})
    hits.sort(key=lambda x: x["pka"])
    return hits


# ── PubChem ────────────────────────────────────────────────────────────────

def get_all_pubchem(smiles):
    try:
        encoded = urllib.parse.quote(smiles)
        cid_r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/cids/JSON",
            timeout=10)
        if cid_r.status_code != 200:
            return {}
        cid = cid_r.json()["IdentifierList"]["CID"][0]
    except Exception:
        return {}

    props, exp_pka = None, None
    try:
        p = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
            f"/property/IUPACName,MolecularFormula,MolecularWeight,XLogP/JSON",
            timeout=10)
        if p.status_code == 200:
            props = p.json()["PropertyTable"]["Properties"][0]
    except Exception:
        pass
    try:
        pka_r = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}"
            f"/JSON/?heading=Dissociation+Constants",
            timeout=10)
        if pka_r.status_code == 200:
            values = []
            for sec in pka_r.json().get("Record", {}).get("Section", []):
                for sub in sec.get("Section", []):
                    for prop in sub.get("Section", []):
                        if "Dissociation" in prop.get("TOCHeading", ""):
                            for info in prop.get("Information", []):
                                for item in info.get("Value", {}).get("StringWithMarkup", []):
                                    s = item.get("String", "").strip()
                                    if s:
                                        values.append(s)
            exp_pka = values if values else None
    except Exception:
        pass

    return {
        "cid": cid,
        "iupac_name":        props.get("IUPACName") if props else None,
        "molecular_formula": props.get("MolecularFormula") if props else None,
        "molecular_weight":  props.get("MolecularWeight") if props else None,
        "xlogp":             props.get("XLogP") if props else None,
        "experimental_pka":  exp_pka,
        "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
    }


# ── ChEMBL ─────────────────────────────────────────────────────────────────

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


def _cget(url, timeout=12):
    try:
        r = requests.get(url, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_chembl_data(smiles):
    # Use InChI key for reliable exact-match lookup
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        return None
    try:
        inchikey = MolToInchiKey(rdmol)
    except Exception:
        inchikey = None

    mol = None
    if inchikey:
        data = _cget(f"{CHEMBL}/molecule/{inchikey}.json")
        if data and data.get("molecule_chembl_id"):
            mol = data

    # Fall back to 85 % similarity search
    if mol is None:
        encoded = urllib.parse.quote(smiles)
        sim = _cget(f"{CHEMBL}/similarity/{encoded}/85.json?limit=1")
        mols = (sim or {}).get("molecules", [])
        if mols:
            mol = mols[0]

    if mol is None:
        return None

    cid = mol["molecule_chembl_id"]

    # 2 — parallel sub-queries
    urls = {
        "ind":   f"{CHEMBL}/drug_indication.json?molecule_chembl_id={cid}&limit=25",
        "ic50":  f"{CHEMBL}/activity.json?molecule_chembl_id={cid}&standard_type=IC50&limit=30",
        "hl":    f"{CHEMBL}/activity.json?molecule_chembl_id={cid}&standard_type=Half+life&limit=15",
        "hl2":   f"{CHEMBL}/activity.json?molecule_chembl_id={cid}&standard_type=t1%2F2&limit=15",
        "mech":  f"{CHEMBL}/mechanism.json?molecule_chembl_id={cid}&limit=10",
    }
    res = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(_cget, u): k for k, u in urls.items()}
        for f in futs:
            res[futs[f]] = f.result()

    # Indications
    indications = []
    for ind in (res.get("ind") or {}).get("drug_indications", []):
        if ind.get("mesh_heading"):
            indications.append({
                "name":  ind["mesh_heading"],
                "phase": ind.get("max_phase_for_ind"),
            })

    # IC50 — deduplicate by target, sort ascending (most potent first)
    ic50_list = []
    seen = set()
    raw_acts = (res.get("ic50") or {}).get("activities", [])
    raw_acts.sort(key=lambda a: float(a["standard_value"]) if a.get("standard_value") else 1e9)
    for act in raw_acts:
        target = act.get("target_pref_name") or act.get("target_chembl_id")
        val    = act.get("standard_value")
        if not target or not val:
            continue
        try:
            fval = float(val)
        except ValueError:
            continue
        if fval > 1e6 or target in seen:
            continue
        seen.add(target)
        ic50_list.append({
            "target":   target,
            "value":    fval,
            "units":    act.get("standard_units", "nM"),
            "organism": act.get("target_organism", ""),
        })
        if len(ic50_list) >= 6:
            break

    # Half-life (merge both standard_type queries)
    hl_list = []
    for key in ("hl", "hl2"):
        for act in (res.get(key) or {}).get("activities", []):
            val = act.get("standard_value")
            if not val:
                continue
            try:
                fval = float(val)
            except ValueError:
                continue
            org = (act.get("target_organism") or act.get("assay_organism") or "").strip()
            desc = (act.get("assay_description") or "")[:120]
            hl_list.append({
                "value":       fval,
                "units":       act.get("standard_units") or "h",
                "organism":    org,
                "description": desc,
            })
    hl_list = hl_list[:6]

    # Mechanisms of action
    mechanisms = []
    for m in (res.get("mech") or {}).get("mechanisms", []):
        if m.get("mechanism_of_action"):
            mechanisms.append({
                "action": m["mechanism_of_action"],
                "target": m.get("target_name", ""),
                "type":   m.get("action_type", ""),
            })

    # Synonyms (skip duplicates of pref_name)
    pref = (mol.get("pref_name") or "").upper()
    synonyms = list({
        s["molecule_synonym"]
        for s in mol.get("molecule_synonyms", [])
        if s.get("molecule_synonym") and s["molecule_synonym"].upper() != pref
    })[:8]

    return {
        "chembl_id":         cid,
        "name":              mol.get("pref_name"),
        "max_phase":         mol.get("max_phase"),
        "withdrawn":         mol.get("withdrawn_flag"),
        "withdrawn_reason":  mol.get("withdrawn_reason"),
        "withdrawn_class":   mol.get("withdrawn_class"),
        "withdrawn_year":    mol.get("withdrawn_year"),
        "withdrawn_country": mol.get("withdrawn_country"),
        "indication_class":  mol.get("indication_class"),
        "synonyms":          synonyms,
        "indications":       indications,
        "ic50":              ic50_list,
        "halflife":          hl_list,
        "mechanisms":        mechanisms,
        "url": f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/",
    }


# ── Flask routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body   = request.get_json(force=True)
    smiles = (body.get("smiles") or "").strip()

    if not smiles:
        return jsonify({"error": "No SMILES string provided."}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string — please check your input."}), 400

    canonical = Chem.MolToSmiles(mol)

    logp = calculate_logp(mol)
    pred_pka, pka_group, all_basic, highlight_atoms = predict_base_pka(mol)
    acidic_groups = predict_acidic_pka(mol)
    svg = mol_to_svg(mol, highlight_atoms or None)

    mw        = round(rdMolDescriptors.CalcExactMolWt(mol), 3)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    tpsa      = round(rdMolDescriptors.CalcTPSA(mol), 2)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings     = rdMolDescriptors.CalcNumRings(mol)

    # PubChem + ChEMBL in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_pc  = ex.submit(get_all_pubchem, canonical)
        f_che = ex.submit(get_chembl_data, canonical)
        pubchem = f_pc.result()
        chembl  = f_che.result()

    return jsonify({
        "smiles":               canonical,
        "svg":                  svg,
        "logp":                 logp,
        "predicted_pka":        pred_pka,
        "pka_functional_group": pka_group,
        "all_basic_groups":     all_basic,
        "acidic_groups":        acidic_groups,
        "molecular_weight":     mw,
        "hbd": hbd, "hba": hba, "tpsa": tpsa,
        "rotatable_bonds":      rot_bonds,
        "rings":                rings,
        "pubchem":              pubchem,
        "chembl":               chembl,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
