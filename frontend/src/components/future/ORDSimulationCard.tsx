import { Activity } from "lucide-react";

interface Props {
  apd90: number;
  apd50: number;
  triangulation: number;
  rmp: number;
}

export default function ORDSimulationCard({
  apd90,
  apd50,
  triangulation,
  rmp,
}: Props) {
  const rows = [
    ["Resting Membrane Potential", `${rmp.toFixed(2)} mV`],
    ["APD50", `${apd50.toFixed(2)} ms`],
    ["APD90", `${apd90.toFixed(2)} ms`],
    ["Triangulation", triangulation.toFixed(2)],
  ];

  return (
    <div className="bg-white rounded-2xl shadow-md p-6">
      <div className="flex items-center gap-3 mb-5">
        <Activity className="text-blue-600" />
        <h2 className="text-xl font-semibold">
          ORd Simulation
        </h2>
      </div>

      <div className="space-y-3">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="flex justify-between border-b pb-2"
          >
            <span>{label}</span>
            <span className="font-semibold">{value}</span>
          </div>
        ))}
      </div>

      <div className="mt-6 h-56 rounded-xl border-2 border-dashed border-slate-300 flex items-center justify-center bg-slate-50">
        <p className="text-slate-500">
          Action Potential Plot Placeholder
        </p>
      </div>
    </div>
  );
}