export interface MoleculeProperties {
  smiles: string;

  molecular_weight?: number;

  formula?: string;

  logp?: number;

  tpsa?: number;

  h_bond_donors?: number;

  h_bond_acceptors?: number;

  rotatable_bonds?: number;

  aromatic_rings?: number;

  heavy_atoms?: number;
}

export interface MoleculeValidation {
  valid: boolean;
  smiles: string;
  canonical_smiles?: string;
  message?: string;
}

export interface MoleculeRenderResponse {
  svg?: string;
  png?: string;
}