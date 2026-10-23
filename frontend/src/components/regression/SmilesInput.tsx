
interface SmilesInputProps {
  value: string;
  onChange: (value: string) => void;
}

export default function SmilesInput({
  value,
  onChange,
}: SmilesInputProps) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-slate-700">
        SMILES String
      </label>

      <textarea
        rows={5}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Paste SMILES string here..."
        className="w-full rounded-xl border border-slate-300 p-4 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none resize-none"
      />

      <p className="text-xs text-slate-500">
        Example: CCOC(=O)c1ccccc1
      </p>
    </div>
  );
}