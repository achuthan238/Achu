import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

# RDKit
from rdkit import Chem
from rdkit.Chem import (
    Draw,
    AllChem,
    Descriptors,
    Crippen,
    Lipinski,
    rdMolDescriptors,
    DataStructs
)

# 3D Visualization
import py3Dmol
import streamlit.components.v1 as components

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    Preformatted
)
from reportlab.lib.units import inch
import tempfile
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cheminformatics Virtual Laboratory",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.section-title {
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SHARED MOLECULE INPUT STATE
# ============================================================

EXAMPLE_MOLECULES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1"
}

if "active_smiles" not in st.session_state:
    st.session_state.active_smiles = "CCO"

if "selected_example" not in st.session_state:
    st.session_state.selected_example = "Ethanol"

def load_shared_example():
    """Load the selected example into the shared SMILES field."""
    st.session_state.active_smiles = EXAMPLE_MOLECULES[
        st.session_state.selected_example
    ]

def get_molecule(smiles):
    """Convert SMILES into an RDKit molecule."""
    if not smiles or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol
    except Exception:
        return None


# ============================================================
# CALCULATE SIGMA AND PI BONDS
# ============================================================

def calculate_bond_information(mol):
    """Calculate heavy-atom bonds, sigma bonds, and pi bonds."""
    heavy_atom_bonds = mol.GetNumBonds()
    kekule_mol = Chem.Mol(mol)

    try:
        Chem.Kekulize(kekule_mol, clearAromaticFlags=True)
    except Exception:
        pass

    pi_bonds = 0
    for bond in kekule_mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == Chem.BondType.DOUBLE:
            pi_bonds += 1
        elif bond_type == Chem.BondType.TRIPLE:
            pi_bonds += 2
        elif bond_type == Chem.BondType.QUADRUPLE:
            pi_bonds += 3
        elif bond.GetIsAromatic():
            pi_bonds += 0.5

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    sigma_bonds = mol_with_h.GetNumBonds()
    total_bonds = sigma_bonds + pi_bonds

    return {
        "Heavy Atom Bonds": heavy_atom_bonds,
        "Sigma Bonds": sigma_bonds,
        "Pi Bonds": pi_bonds,
        "Total Bonds": total_bonds,
        "Total Atoms Including H": mol_with_h.GetNumAtoms(),
    }


# ============================================================
# DRUG-LIKENESS FILTERS (Lipinski & Veber)
# ============================================================

def evaluate_drug_likeness(properties):
    """Evaluate Lipinski's Rule of 5 and Veber's rules."""
    mw = properties["Molecular Weight"]
    logp = properties["LogP"]
    hbd = properties["Hydrogen Bond Donors"]
    hba = properties["Hydrogen Bond Acceptors"]
    rot_bonds = properties["Rotatable Bonds"]
    tpsa = properties["TPSA"]

    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1

    veber_pass = (rot_bonds <= 10) and (tpsa <= 140)

    return {
        "Lipinski Violations": lipinski_violations,
        "Lipinski Pass": lipinski_violations <= 1,
        "Veber Pass": veber_pass
    }


# ============================================================
# CALCULATE COMPLETE MOLECULAR PROPERTIES
# ============================================================

def calculate_properties(mol):
    bond_info = calculate_bond_information(mol)
    molecular_formula = rdMolDescriptors.CalcMolFormula(mol)

    properties = {
        "Molecular Formula": molecular_formula,
        "Molecular Weight": round(Descriptors.MolWt(mol), 3),
        "Exact Molecular Weight": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "Hydrogen Bond Donors": Lipinski.NumHDonors(mol),
        "Hydrogen Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Aromatic Rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Aliphatic Rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "Number of Atoms": mol.GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Heavy Atom Bonds": bond_info["Heavy Atom Bonds"],
        "Sigma (σ) Bonds": bond_info["Sigma Bonds"],
        "Pi (π) Bonds": bond_info["Pi Bonds"],
        "Total Bonds (σ + π)": bond_info["Total Bonds"],
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        "Molar Refractivity": round(Crippen.MolMR(mol), 3)
    }
    return properties


# ============================================================
# GENERATE 2D IMAGE WITH HIGHLIGHTS
# ============================================================

def generate_2d_image(mol, highlight_atoms=None):
    if highlight_atoms:
        return Draw.MolToImage(mol, size=(700, 500), highlightAtoms=highlight_atoms)
    return Draw.MolToImage(mol, size=(700, 500))


# ============================================================
# GENERATE 3D MOLECULE
# ============================================================

def generate_3d_molecule(mol):
    try:
        mol3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol3d, params)
        if status != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol3d)
            except Exception:
                pass
        return mol3d
    except Exception:
        return None


# ============================================================
# DISPLAY 3D STRUCTURE
# ============================================================

def display_3d_structure(mol3d, style, show_surface=False):
    mol_block = Chem.MolToMolBlock(mol3d)
    view = py3Dmol.view(width=900, height=600)
    view.addModel(mol_block, "mol")

    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.30}})
    elif style == "Stick":
        view.setStyle({"stick": {"radius": 0.20}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {"scale": 1.0}})
    elif style == "Wireframe":
        view.setStyle({"line": {"linewidth": 2}})
    elif style == "Cartoon":
        view.setStyle({"stick": {}, "sphere": {"scale": 0.20}})

    if show_surface:
        view.addSurface(py3Dmol.VDW, {"opacity": 0.7})

    view.setBackgroundColor("white")
    view.zoomTo()
    html = view._make_html()
    components.html(html, height=620)


# ============================================================
# REPORT IMAGE HELPERS
# ============================================================

def save_2d_structure_png(mol):
    img = Draw.MolToImage(mol, size=(700, 500))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    img.save(tmp.name)
    return tmp.name


def save_3d_structure_png(mol):
    mol3d = generate_3d_molecule(mol)
    if mol3d is None:
        return None

    conf = mol3d.GetConformer()
    n_atoms = mol3d.GetNumAtoms()
    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(n_atoms)], dtype=float)

    centered = coords - coords.mean(axis=0)
    if n_atoms >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ vh[:2].T
    else:
        projected = centered[:, :2]

    cpk = {"H": "#FFFFFF", "C": "#3F3F3F", "N": "#2F5BFF", "O": "#E53935", "F": "#65C466", "S": "#F4D03F"}
    sizes = {"H": 70, "C": 240, "N": 270, "O": 285, "S": 310}

    fig, ax = plt.subplots(figsize=(8.8, 6.4), facecolor="white")
    ax.set_facecolor("white")

    for bond in mol3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = projected[i]
        x2, y2 = projected[j]
        ax.plot([x1, x2], [y1, y2], color="#707070", linewidth=3.0, zorder=1)

    for i, atom in enumerate(mol3d.GetAtoms()):
        symbol = atom.GetSymbol()
        x, y = projected[i]
        ax.scatter(x, y, s=sizes.get(symbol, 250), c=cpk.get(symbol, "#A0A0A0"), edgecolors="#222222", linewidths=1.0, zorder=3)
        if symbol not in ("H", "C"):
            ax.text(x, y, symbol, ha="center", va="center", fontsize=8, fontweight="bold", color="black", zorder=4)

    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.close()
    fig.savefig(tmp.name, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


# ============================================================
# PROPERTY INTERPRETATION HELPERS
# ============================================================

def get_property_interpretation(property_name, value, properties):
    if property_name == "Molecular Formula":
        return "Shows the elemental composition of the molecule."
    if property_name == "Molecular Weight":
        return "Affects diffusion, formulation and physicochemical behaviour."
    if property_name == "LogP":
        return "Measures lipophilicity and membrane permeability."
    if property_name == "TPSA":
        return "Topological Polar Surface Area; correlates with passive molecular transport."
    if property_name == "Hydrogen Bond Donors":
        return f"Molecule has {value} donor site(s)."
    if property_name == "Hydrogen Bond Acceptors":
        return f"Molecule has {value} acceptor site(s)."
    return "Provides additional quantitative insight into the structure."


def build_key_interpretation_points(properties):
    points = [
        f"<b>• Molecular identity:</b> Formula <b>{properties['Molecular Formula']}</b>, MW = <b>{properties['Molecular Weight']}</b>.",
        f"<b>• Lipophilicity:</b> LogP = <b>{properties['LogP']}</b>.",
        f"<b>• Molecular polarity:</b> TPSA = <b>{properties['TPSA']}</b> Å².",
        f"<b>• Flexibility:</b> {properties['Rotatable Bonds']} rotatable bond(s)."
    ]
    return points


# ============================================================
# GENERATE PDF REPORT
# ============================================================

def generate_final_report(student_name, registration_number, input_smiles, mol, properties):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=38, rightMargin=38, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, textColor=HexColor("#17365D"), alignment=TA_CENTER)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading1"], fontSize=12, textColor=HexColor("#17365D"), spaceBefore=6, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12, alignment=TA_JUSTIFY)

    story = [
        Paragraph("CHEMINFORMATICS VIRTUAL LABORATORY REPORT", title_style),
        Spacer(1, 0.1 * inch),
        Paragraph(f"<b>Student Name:</b> {student_name or 'N/A'} &nbsp;&nbsp;&nbsp; <b>Reg No:</b> {registration_number or 'N/A'}", body_style),
        Spacer(1, 0.15 * inch)
    ]

    img2d = save_2d_structure_png(mol)
    if img2d:
        story.append(Image(img2d, width=3.5 * inch, height=2.5 * inch))
        story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Key Molecular Properties", heading_style))
    prop_data = [["Property", "Value"]]
    for k, v in properties.items():
        prop_data.append([str(k), str(v)])

    t = Table(prop_data, colWidths=[3.0 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#2F5597")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 4)
    ]))
    story.append(t)

    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    if img2d and os.path.exists(img2d):
        os.remove(img2d)
    return pdf_data


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 Virtual Lab")
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 Theory",
        "🧬 Molecular Visualization",
        "📊 Molecular Properties Report",
        "📈 Structure–Property & Similarity",
        "📝 Assessment",
        "📄 Final Project Report"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student_name = st.sidebar.text_input("Student Name", key="student_name")
registration_number = st.sidebar.text_input("Registration Number", key="registration_number")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":
    st.markdown('<div class="main-title">🧪 Cheminformatics Virtual Laboratory</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align:center; font-size:20px; font-weight:700; color:#1f4e79;">
        SRM Institute of Science and Technology, Tiruchirappalli
        </div>
        <div style="text-align:center; font-size:16px; font-weight:600; margin-top:5px;">
        School of Sciences | Division of Chemistry
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.write("Welcome to the advanced interactive platform for computational chemistry and cheminformatics education.")
    
    st.markdown("### Example Molecules")
    example_df = pd.DataFrame(list(EXAMPLE_MOLECULES.items()), columns=["Molecule", "SMILES"])
    st.dataframe(example_df, use_container_width=True, hide_index=True)


# ============================================================
# THEORY PAGE
# ============================================================

elif page == "📖 Theory":
    st.title("📖 Cheminformatics Theory & Guidelines")
    st.markdown("""
    ### Core Concepts
    - **SMILES Notation**: Compact text strings representing molecular topology.
    - **Drug-Likeness (Lipinski's Rule of 5)**: Evaluates oral bioavailability parameters ($MW \le 500$, $LogP \le 5$, $HBD \le 5$, $HBA \le 10$).
    - **Molecular Descriptors**: Quantitative metrics reflecting polar surface area, refractivity, and bonding topologies.
    """)


# ============================================================
# MOLECULAR VISUALIZATION & SUBSTRUCTURE HIGHLIGHTER
# ============================================================

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization & Substructure Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input("Enter SMILES", key="active_smiles")
    with col2:
        st.selectbox("Choose Example Molecule", list(EXAMPLE_MOLECULES.keys()), key="selected_example")
        st.button("Load Example", on_click=load_shared_example)

    mol = get_molecule(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES notation.")
    else:
        st.success("✅ Valid Molecular Structure!")
        
        # Substructure highlighting option
        substruct_query = st.text_input("Highlight Substructure (SMARTS, e.g., 'O' for oxygen or 'c1ccccc1' for phenyl)", "")
        highlight_atoms = []
        if substruct_query:
            query_mol = Chem.MolFromSmarts(substruct_query)
            if query_mol:
                matches = mol.GetSubstructMatches(query_mol)
                highlight_atoms = [atom for match in matches for atom in match]
            else:
                st.warning("Invalid SMARTS query pattern.")

        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.subheader("2D Molecular Structure")
            st.image(generate_2d_image(mol, highlight_atoms))
        with col2:
            st.subheader("Bond Analytics")
            bond_info = calculate_bond_information(mol)
            st.write("**Heavy Atoms:**", mol.GetNumHeavyAtoms())
            st.write("**Sigma (σ) Bonds:**", bond_info["Sigma Bonds"])
            st.write("**Pi (π) Bonds:**", bond_info["Pi Bonds"])

        st.divider()
        st.subheader("🌐 Interactive 3D Molecular Structure")
        style = st.selectbox("Select 3D Model", ["Ball and Stick", "Stick", "Space Filling", "Wireframe"])
        if st.button("Generate 3D Structure"):
            mol3d = generate_3d_molecule(mol)
            if mol3d:
                display_3d_structure(mol3d, style)
            else:
                st.error("Could not embed 3D coordinates.")


# ============================================================
# COMPLETE MOLECULAR PROPERTIES REPORT
# ============================================================

elif page == "📊 Molecular Properties Report":
    st.title("📊 Complete Molecular Properties & Drug-Likeness")

    col1, col2 = st.columns([2, 1])
    with col1:
        smiles = st.text_input("Enter SMILES", key="active_smiles")
    with col2:
        st.selectbox("Choose Example Molecule", list(EXAMPLE_MOLECULES.keys()), key="selected_example")
        st.button("Load Example", on_click=load_shared_example)

    if smiles:
        mol = get_molecule(smiles)
        if mol:
            properties = calculate_properties(mol)
            drug_filters = evaluate_drug_likeness(properties)

            st.success("✅ Valid Molecular Structure Detected!")
            
            # Drug-Likeness Metrics Banner
            st.info(f"💊 **Lipinski's Rule of 5:** {'Passes (≤1 violation)' if drug_filters['Lipinski Pass'] else 'Fails (>1 violation)'} | **Violations:** {drug_filters['Lipinski Violations']}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Molecular Weight", f"{properties['Molecular Weight']} g/mol")
            col2.metric("LogP", properties['LogP'])
            col3.metric("TPSA", f"{properties['TPSA']} Å²")
            col4.metric("H-Bond Donors", properties['Hydrogen Bond Donors'])

            report_df = pd.DataFrame(list(properties.items()), columns=["Property", "Value"])
            st.dataframe(report_df, use_container_width=True, hide_index=True)

            pdf = generate_final_report(student_name, registration_number, smiles, mol, properties)
            st.download_button("⬇️ Download PDF Report", data=pdf, file_name="Molecular_Report.pdf", mime="application/pdf")


# ============================================================
# STRUCTURE-PROPERTY & SIMILARITY ANALYSIS
# ============================================================

elif page == "📈 Structure–Property & Similarity":
    st.title("📈 Structure–Property & Fingerprint Similarity")

    st.write("Compare molecular descriptors and compute Tanimoto similarity based on Morgan fingerprints.")

    ref_smiles = st.text_input("Reference Molecule SMILES", "CCO")
    target_smiles_list = st.text_area("Target Molecules (one per line, e.g., CCO, CC(=O)O, Oc1ccccc1)", "CC(=O)O\nOc1ccccc1\nc1ccccc1")

    if st.button("Run Similarity Analysis"):
        ref_mol = get_molecule(ref_smiles)
        if not ref_mol:
            st.error("Invalid Reference SMILES.")
        else:
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
            results = []
            for line in target_smiles_list.splitlines():
                if not line.strip(): continue
                t_mol = get_molecule(line.strip())
                if t_mol:
                    t_fp = AllChem.GetMorganFingerprintAsBitVect(t_mol, 2, nBits=2048)
                    similarity = DataStructs.TanimotoSimilarity(ref_fp, t_fp)
                    results.append({
                        "SMILES": line.strip(),
                        "Formula": rdMolDescriptors.CalcMolFormula(t_mol),
                        "Tanimoto Similarity": round(similarity, 3)
                    })
            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)


# ============================================================
# ASSESSMENT
# ============================================================

elif page == "📝 Assessment":
    st.title("📝 Cheminformatics Assessment")
    questions = [
        {"question": "What does SMILES stand for?", "options": ["Simplified Molecular Input Line Entry System", "Standard Molecular Index Language", "Simple Molecular Identification System"], "answer": "Simplified Molecular Input Line Entry System"},
        {"question": "Which descriptor measures molecular lipophilicity?", "options": ["LogP", "TPSA", "Molecular Weight"], "answer": "LogP"}
    ]
    
    user_answers = [st.radio(q["question"], q["options"], index=None, key=f"q_{i}") for i, q in enumerate(questions)]
    if st.button("Submit Assessment"):
        score = sum(1 for i, q in enumerate(questions) if user_answers[i] == q["answer"])
        st.success(f"Your Score: {score}/{len(questions)}")


# ============================================================
# FINAL PROJECT REPORT
# ============================================================

elif page == "📄 Final Project Report":
    st.title("📄 Final Student Molecular Report Generation")
    smiles_input = st.text_area("Enter SMILES (one per line)", "CCO\nc1ccccc1")
    if st.button("Generate Final Report"):
        mol = get_molecule(smiles_input.splitlines()[0] if smiles_input else "CCO")
        if mol:
            pdf = generate_final_report(student_name, registration_number, smiles_input, mol, calculate_properties(mol))
            st.download_button("Download Final Report", data=pdf, file_name="Final_Project_Report.pdf", mime="application/pdf")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("🧪 Cheminformatics Virtual Laboratory | Built with Streamlit & RDKit")