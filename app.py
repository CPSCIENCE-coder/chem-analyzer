from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import requests
import urllib.parse

app = Flask(__name__)

# Basic groups — pKa_BH+ (conjugate acid), order matters: more specific first
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

# Acidic groups — pKa_AH (proton donor), most acidic first
ACIDIC_PKA_SMARTS = [
    ("S(=O)(=O)[OH]",  1.5,  "Sulfonic acid"),
    ("C(=O)[OH]",      4.2,  "Carboxylic acid"),
    ("n1[nH]nnn1",     4.9,  "Tetrazole"),
    ("c[OH]",          9.9,  "Phenol"),
    ("[SH]",          10.5,  "Thiol"),
    ("S(=O)(=O)[NH]", 10.1,  "Sulfonamide N-H"),
]


def mol_to_svg(mol, highlight_atoms=None, size=(480, 300)):
    """Return an SVG string with optional orange highlighting on given atom indices."""
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
    """Return (top_pka, top_group, all_groups_list, highlight_atom_indices).
    Deduplicates by tracking which nitrogen atoms have been claimed."""
    claimed = set()
    hits = []

    for smarts, pka, name in BASE_PKA_SMARTS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is None:
            continue
        for match in mol.GetSubstructMatches(pat):
            # The first atom in the match is treated as the key ionisable atom
            key = match[0]
            if key not in claimed:
                claimed.add(key)
                hits.append({"pka": pka, "group": name, "atoms": list(match)})

    if not hits:
        return None, None, [], []

    hits.sort(key=lambda x: x["pka"], reverse=True)
    top = hits[0]
    all_groups = [{"pka": h["pka"], "group": h["group"]} for h in hits]
    return top["pka"], top["group"], all_groups, top["atoms"]


def predict_acidic_pka(mol):
    """Return list of {pka, group} for each distinct acidic site found."""
    claimed = set()
    hits = []

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


def get_pubchem_cid(smiles):
    try:
        encoded = urllib.parse.quote(smiles)
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
               f"{encoded}/cids/JSON")
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()["IdentifierList"]["CID"][0]
    except Exception:
        pass
    return None


def get_pubchem_properties(cid):
    try:
        props = "IUPACName,MolecularFormula,MolecularWeight,XLogP"
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
               f"{cid}/property/{props}/JSON")
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()["PropertyTable"]["Properties"][0]
    except Exception:
        pass
    return None


def get_pubchem_experimental_pka(cid):
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/"
               f"{cid}/JSON/?heading=Dissociation+Constants")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        values = []
        for section in data.get("Record", {}).get("Section", []):
            for subsec in section.get("Section", []):
                for prop in subsec.get("Section", []):
                    if "Dissociation" in prop.get("TOCHeading", ""):
                        for info in prop.get("Information", []):
                            for item in info.get("Value", {}).get("StringWithMarkup", []):
                                s = item.get("String", "").strip()
                                if s:
                                    values.append(s)
        return values if values else None
    except Exception:
        pass
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
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

    # SVG with top basic site circled in orange
    svg = mol_to_svg(mol, highlight_atoms if highlight_atoms else None)

    mw        = round(rdMolDescriptors.CalcExactMolWt(mol), 3)
    hbd       = rdMolDescriptors.CalcNumHBD(mol)
    hba       = rdMolDescriptors.CalcNumHBA(mol)
    tpsa      = round(rdMolDescriptors.CalcTPSA(mol), 2)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings     = rdMolDescriptors.CalcNumRings(mol)

    pubchem = {}
    cid = get_pubchem_cid(canonical)
    if cid:
        props   = get_pubchem_properties(cid)
        exp_pka = get_pubchem_experimental_pka(cid)
        pubchem = {
            "cid": cid,
            "iupac_name":       props.get("IUPACName") if props else None,
            "molecular_formula":props.get("MolecularFormula") if props else None,
            "molecular_weight": props.get("MolecularWeight") if props else None,
            "xlogp":            props.get("XLogP") if props else None,
            "experimental_pka": exp_pka,
            "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        }

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
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
