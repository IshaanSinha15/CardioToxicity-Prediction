
interface DoseInputProps {
  value: number | "";
  onChange: (value: number) => void;
}

export default function DoseInput({
  value,
  onChange,
}: DoseInputProps) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-slate-700">
        Dose (nM)
      </label>

      <input
        type="number"
        min={0}
        step="0.1"
        value={value}
        placeholder="1000"
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-xl border border-slate-300 px-4 py-3 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none"
      />
    </div>
  );
}