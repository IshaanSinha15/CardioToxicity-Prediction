import api from "./api";

export interface MoleculeInfo {
  smiles: string;
  molecular_weight?: number;
  formula?: string;
  logp?: number;
  tpsa?: number;
  rotatable_bonds?: number;
}

class MoleculeService {
  async validateSmiles(smiles: string) {
    const response = await api.post(
      "/molecule/validate",
      {
        smiles,
      }
    );

    return response.data;
  }

  async getProperties(
    smiles: string
  ): Promise<MoleculeInfo> {
    const response = await api.post(
      "/molecule/properties",
      {
        smiles,
      }
    );

    return response.data;
  }

  async renderMolecule(smiles: string) {
    const response = await api.post(
      "/molecule/render",
      {
        smiles,
      }
    );

    return response.data;
  }
}

export default new MoleculeService();