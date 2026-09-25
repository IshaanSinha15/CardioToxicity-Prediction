from rdkit.Chem.Draw import rdMolDraw2D

def generate_svg(mol, highlight_atoms):
    """
    Generate an SVG image of a molecule with highlighted atoms.
    """

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms
    )

    drawer.FinishDrawing()

    return drawer.GetDrawingText()