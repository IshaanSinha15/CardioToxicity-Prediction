import { Atom } from "lucide-react";

interface MoleculeViewerProps {
  smiles: string;
}

export default function MoleculeViewer({
  smiles,
}: MoleculeViewerProps) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <div className="flex items-center gap-3 mb-5">

        <Atom className="text-blue-600" />

        <h2 className="text-xl font-semibold">
          Molecule
        </h2>

      </div>

      <div className="rounded-xl border-2 border-dashed border-slate-300 h-64 flex flex-col items-center justify-center bg-slate-50">

        <Atom
          size={72}
          className="text-slate-300 mb-4"
        />

        <p className="text-sm text-slate-500 text-center px-6 break-all">
          {smiles || "No molecule loaded"}
        </p>

        <p className="text-xs text-slate-400 mt-3">
          RDKit visualization can replace this placeholder.
        </p>

      </div>

    </div>
  );
}