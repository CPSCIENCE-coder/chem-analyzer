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
import math

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


def calculate_logd(logp, all_basic_groups, acidic_groups, ph=7.4):
    """LogD at given pH: logP corrected for ionization of all predicted sites.
    Assumes only the fully neutral species partitions into octanol."""
    log_f = 0.0
    for g in (all_basic_groups or []):
        # basic site: neutral fraction = 1 / (1 + 10^(pKa - pH))
        log_f -= math.log10(1 + 10 ** (g["pka"] - ph))
    for g in (acidic_groups or []):
        # acidic site: neutral fraction = 1 / (1 + 10^(pH - pKa))
        log_f -= math.log10(1 + 10 ** (ph - g["pka"]))
    return round(logp + log_f, 2)


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
        "dose":  f"{CHEMBL}/activity.json?molecule_chembl_id={cid}&standard_type=Dose&limit=15",
    }
    res = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_cget, u): k for k, u in urls.items()}
        for f in futs:
            res[futs[f]] = f.result()

    # Indications — include ClinicalTrials search URL per indication
    indications = []
    for ind in (res.get("ind") or {}).get("drug_indications", []):
        if ind.get("mesh_heading"):
            name_q = urllib.parse.quote(f"{mol.get('pref_name','')} {ind['mesh_heading']}")
            indications.append({
                "name":  ind["mesh_heading"],
                "phase": ind.get("max_phase_for_ind"),
                "trials_url": f"https://clinicaltrials.gov/search?term={name_q}",
            })

    # IC50 — include assay + target ChEMBL reference links
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
        assay_id  = act.get("assay_chembl_id")
        target_id = act.get("target_chembl_id")
        ic50_list.append({
            "target":      target,
            "value":       fval,
            "units":       act.get("standard_units", "nM"),
            "organism":    act.get("target_organism", ""),
            "assay_url":   f"https://www.ebi.ac.uk/chembl/assay_report_card/{assay_id}/" if assay_id else None,
            "target_url":  f"https://www.ebi.ac.uk/chembl/target_report_card/{target_id}/" if target_id else None,
        })
        if len(ic50_list) >= 6:
            break

    # Half-life
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
            org  = (act.get("target_organism") or act.get("assay_organism") or "").strip()
            desc = (act.get("assay_description") or "")[:120]
            assay_id = act.get("assay_chembl_id")
            hl_list.append({
                "value":     fval,
                "units":     act.get("standard_units") or "h",
                "organism":  org,
                "description": desc,
                "assay_url": f"https://www.ebi.ac.uk/chembl/assay_report_card/{assay_id}/" if assay_id else None,
            })
    hl_list = hl_list[:6]

    # Clinical Dose
    dose_list = []
    for act in (res.get("dose") or {}).get("activities", []):
        val = act.get("standard_value")
        if not val:
            continue
        try:
            fval = float(val)
        except ValueError:
            continue
        org   = (act.get("target_organism") or act.get("assay_organism") or "").strip()
        units = (act.get("standard_units") or "mg").strip()
        desc  = (act.get("assay_description") or "")[:140]
        assay_id = act.get("assay_chembl_id")
        dose_list.append({
            "value":     fval,
            "units":     units,
            "organism":  org,
            "description": desc,
            "assay_url": f"https://www.ebi.ac.uk/chembl/assay_report_card/{assay_id}/" if assay_id else None,
        })
    dose_list = dose_list[:6]

    # Mechanisms — include ChEMBL target URL
    mechanisms = []
    for m in (res.get("mech") or {}).get("mechanisms", []):
        if m.get("mechanism_of_action"):
            target_id = m.get("target_chembl_id")
            mechanisms.append({
                "action":     m["mechanism_of_action"],
                "target":     m.get("target_name", ""),
                "type":       m.get("action_type", ""),
                "target_url": f"https://www.ebi.ac.uk/chembl/target_report_card/{target_id}/" if target_id else None,
            })

    # Synonyms
    pref = (mol.get("pref_name") or "").upper()
    synonyms = list({
        s["molecule_synonym"]
        for s in mol.get("molecule_synonyms", [])
        if s.get("molecule_synonym") and s["molecule_synonym"].upper() != pref
    })[:8]

    name = mol.get("pref_name") or ""
    return {
        "chembl_id":         cid,
        "name":              name,
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
        "clinical_dose":     dose_list,
        "mechanisms":        mechanisms,
        "url":               f"https://www.ebi.ac.uk/chembl/compound_report_card/{cid}/",
        "trials_url":        f"https://clinicaltrials.gov/search?term={urllib.parse.quote(name)}",
        "pubmed_url":        f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote(name)}&sort=date",
        "drugbank_url":      f"https://go.drugbank.com/unearth/q?query={urllib.parse.quote(name)}&searcher=drugs",
    }


# ── PubMed recent publications ─────────────────────────────────────────────

def get_pubmed_news(name, max_results=6):
    if not name:
        return []
    try:
        q = urllib.parse.quote(f"{name}[Title/Abstract]")
        search = _cget(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=pubmed&term={q}&sort=date&retmax={max_results}"
            f"&datetype=pdat&reldate=1095&retmode=json",   # last 3 years
            timeout=12)
        ids = (search or {}).get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        ids_str = ",".join(ids)
        summary = _cget(
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=pubmed&id={ids_str}&retmode=json",
            timeout=12)
        result = (summary or {}).get("result", {})
        articles = []
        for pmid in ids:
            art = result.get(pmid, {})
            if not art or art.get("error"):
                continue
            authors = [a.get("name", "") for a in art.get("authors", [])[:3]]
            articles.append({
                "title":   art.get("title", "").rstrip("."),
                "journal": art.get("source", ""),
                "date":    art.get("pubdate", ""),
                "authors": authors,
                "pmid":    pmid,
                "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return articles
    except Exception:
        return []


# ── ClinicalTrials.gov ──────────────────────────────────────────────────────

def get_clinical_trials(name, max_results=5):
    if not name:
        return []
    try:
        q = urllib.parse.quote(name)
        data = _cget(
            f"https://clinicaltrials.gov/api/v2/studies"
            f"?query.term={q}&sort=LastUpdatePostDate:desc&pageSize={max_results}"
            f"&fields=NCTId,BriefTitle,OverallStatus,Phase,StartDate",
            timeout=12)
        studies = (data or {}).get("studies", [])
        trials = []
        for s in studies:
            proto  = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            st_mod = proto.get("statusModule", {})
            dg_mod = proto.get("designModule", {})
            nct    = id_mod.get("nctId")
            trials.append({
                "nct_id":  nct,
                "title":   id_mod.get("briefTitle", ""),
                "status":  st_mod.get("overallStatus", ""),
                "phases":  dg_mod.get("phases", []),
                "start":   (st_mod.get("startDateStruct") or {}).get("date", ""),
                "url":     f"https://clinicaltrials.gov/study/{nct}" if nct else "#",
            })
        return trials
    except Exception:
        return []


# ── Liposomal encapsulation suitability ───────────────────────────────────

def evaluate_liposomal_suitability(logd, pred_pka, all_basic, acidic_groups, mw, tpsa, hbd, chembl_data):
    score      = 0
    rationale  = []
    concerns   = []
    load_mode  = None   # "remote" | "passive"

    # ── logD7.4 ──
    if 1.0 <= logd <= 3.5:
        score += 30; load_mode = "passive"
        rationale.append(f"logD₇.₄ {logd:.2f}: Optimal (1–3.5) — good bilayer affinity with sufficient aqueous solubility at physiological pH; high passive encapsulation efficiency expected.")
    elif 3.5 < logd <= 5.0:
        score += 22; load_mode = "passive"
        rationale.append(f"logD₇.₄ {logd:.2f}: Lipophilic (3.5–5) — passive loading feasible; monitor for bilayer integration rather than aqueous-core retention.")
        concerns.append("Elevated logD₇.₄ may cause drug to embed in the lipid bilayer membrane, reducing encapsulation efficiency and altering release kinetics.")
    elif 0.0 <= logd < 1.0:
        score += 15
        rationale.append(f"logD₇.₄ {logd:.2f}: Hydrophilic at pH 7.4 — passive encapsulation efficiency will be low; remote loading is strongly preferred if an ionizable amine is present.")
    elif logd < 0:
        score += 5
        concerns.append(f"logD₇.₄ {logd:.2f}: Highly hydrophilic at physiological pH — passive encapsulation unlikely. Remote pH-gradient loading required (requires ionizable amine with pKa 7.5–10.5).")
    else:  # > 5
        score += 10
        concerns.append(f"logD₇.₄ {logd:.2f}: Highly lipophilic at pH 7.4 — likely integrates into the lipid bilayer rather than the aqueous core. Consider nanostructured lipid carriers (NLC) or lipid-drug conjugate strategies.")

    # ── Basic pKa ──
    if pred_pka is not None:
        if 7.5 <= pred_pka <= 10.5:
            score += 35; load_mode = "remote"
            rationale.append(f"Basic pKa {pred_pka:.1f}: Ideal for remote (active) loading — ionizable amine enables transmembrane pH-gradient trapping via ammonium sulfate or citrate buffer method. Encapsulation efficiency typically >80%.")
        elif 10.5 < pred_pka <= 12.5:
            score += 20; load_mode = load_mode or "remote"
            rationale.append(f"Basic pKa {pred_pka:.1f}: Strongly basic amine — remote loading feasible but ionization at endosomal pH (~5.5) may be incomplete, potentially reducing triggered release.")
            concerns.append(f"pKa {pred_pka:.1f} is very high; the amine may remain unionized at endosomal/lysosomal pH, reducing pH-triggered release efficiency.")
        elif 5.0 <= pred_pka < 7.5:
            score += 10
            concerns.append(f"Basic pKa {pred_pka:.1f}: Weakly basic — partially ionized at physiological pH; transmembrane pH gradient may be insufficient for high encapsulation efficiency via remote loading.")
        else:
            score += 5
            concerns.append(f"Basic pKa {pred_pka:.1f}: Very weak base — remote pH-gradient loading not effective at practical pH differentials.")
    else:
        concerns.append("No ionizable basic group detected — remote (pH-gradient) loading not applicable. Passive loading or pH-sensitive lipid formulations (e.g., DOPE/CHEMS) should be considered.")

    # ── Acidic groups ──
    if acidic_groups:
        lowest = min(g["pka"] for g in acidic_groups)
        if lowest < 4.5:
            concerns.append(f"Acidic group (pKa {lowest:.1f}) — drug is anionic at physiological pH, which may cause electrostatic repulsion with negatively charged DSPE-PEG liposomes. Use cationic lipids (DOTAP) or neutral PEGylated formulations.")
        else:
            concerns.append(f"Weakly acidic group (pKa {lowest:.1f}) detected — monitor potential drug–lipid interactions at different pH conditions.")

    # ── MW ──
    if mw < 500:
        score += 15
        rationale.append(f"MW {mw:.0f} Da: Small molecule — excellent candidate for liposomal encapsulation.")
    elif mw < 1000:
        score += 8
        rationale.append(f"MW {mw:.0f} Da: Medium-sized molecule — acceptable for encapsulation; larger liposomes (≥150 nm) may improve loading.")
    else:
        score += 2
        concerns.append(f"MW {mw:.0f} Da: Large molecule — may reduce encapsulation efficiency and diffusion across the lipid bilayer.")

    # ── TPSA ──
    if tpsa < 60:
        score += 10
        rationale.append(f"TPSA {tpsa:.0f} Å²: Low polar surface area — favorable bilayer interaction and membrane partitioning.")
    elif tpsa < 120:
        score += 7
    else:
        score += 3
        concerns.append(f"TPSA {tpsa:.0f} Å²: High polar surface area — reduced bilayer permeability; drug may remain poorly associated with lipid membrane.")

    # ── HBD ──
    if hbd <= 2:
        score += 10
        rationale.append(f"H-bond donors ({hbd}): Low count — favorable for membrane partitioning and passive permeation.")
    elif hbd <= 5:
        score += 6
    else:
        score += 2
        concerns.append(f"H-bond donors ({hbd}): High count — hydrogen bonding with aqueous phase may reduce membrane affinity and encapsulation.")

    # ── Off-target toxicity from ChEMBL ──
    if chembl_data:
        if chembl_data.get("withdrawn"):
            reason = chembl_data.get("withdrawn_reason") or "unknown reason"
            tox_kw = ["tox", "cardiac", "hepat", "renal", "adverse", "safety", "carcinogen", "arrhythmia", "QT", "mutagenic"]
            if any(k.lower() in reason.lower() for k in tox_kw):
                concerns.append(f"Withdrawn due to systemic toxicity ({reason}) — liposomal encapsulation with targeted delivery could reduce systemic exposure and potentially rehabilitate the compound.")
            else:
                concerns.append(f"Drug withdrawn/discontinued ({reason}) — evaluate whether reformulation as a liposome addresses the underlying withdrawal issue.")
        off = [r for r in (chembl_data.get("ic50") or []) if r.get("value") is not None and r["value"] < 1000]
        if len(off) > 3:
            concerns.append(f"{len(off)} off-target activities with IC₅₀ < 1 µM detected — liposomal targeted delivery may improve therapeutic index by limiting systemic exposure.")

    # ── Loading method label ──
    if load_mode == "remote":
        loading_label = "Remote loading (pH gradient — ammonium sulfate / citrate buffer method)"
    elif load_mode == "passive":
        loading_label = "Passive loading (thin-film hydration or solvent injection)"
    else:
        loading_label = "No standard method applicable — evaluate pH-sensitive lipid formulations (DOPE/CHEMS)"

    # ── Verdict ──
    if   score >= 75: overall, color = "Highly Suitable", "green"
    elif score >= 55: overall, color = "Suitable",        "green"
    elif score >= 35: overall, color = "Potentially Suitable", "warn"
    else:             overall, color = "Limited Suitability",  "error"

    return {
        "overall":       overall,
        "verdict_color": color,
        "score":         score,
        "loading_method": loading_label,
        "rationale":     rationale,
        "concerns":      concerns,
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
    logd_74 = calculate_logd(logp, all_basic, acidic_groups, ph=7.4)
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

    liposomal = evaluate_liposomal_suitability(logd_74, pred_pka, all_basic, acidic_groups, mw, tpsa, hbd, chembl)

    # PubMed news + ClinicalTrials in parallel (use compound name if found)
    compound_name = (chembl or {}).get("name") or (pubchem or {}).get("iupac_name") or ""
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_news   = ex.submit(get_pubmed_news,     compound_name)
        f_trials = ex.submit(get_clinical_trials, compound_name)
        news   = f_news.result()
        trials = f_trials.result()

    return jsonify({
        "smiles":               canonical,
        "svg":                  svg,
        "logp":                 logp,
        "logd":                 logd_74,
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
        "liposomal":            liposomal,
        "news":                 news,
        "trials":               trials,
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host="0.0.0.0", port=port)
