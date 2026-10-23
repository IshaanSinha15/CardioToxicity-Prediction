import { ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";

interface ClassificationCardProps {
  risk: "Low" | "Intermediate" | "High";
  confidence: number;
}

export default function ClassificationCard({
  risk,
  confidence,
}: ClassificationCardProps) {
  const config = {
    Low: {
      color: "bg-green-100 text-green-700",
      icon: ShieldCheck,
    },
    Intermediate: {
      color: "bg-yellow-100 text-yellow-700",
      icon: ShieldAlert,
    },
    High: {
      color: "bg-red-100 text-red-700",
      icon: ShieldX,
    },
  };

  const Icon = config[risk].icon;

  return (
    <div className="bg-white rounded-2xl shadow-md p-6">
      <h2 className="text-xl font-semibold mb-6">
        Cardiotoxicity Classification
      </h2>

      <div className={`rounded-xl p-6 flex items-center gap-4 ${config[risk].color}`}>
        <Icon size={42} />

        <div>
          <h3 className="text-2xl font-bold">{risk} Risk</h3>
          <p>Confidence: {confidence.toFixed(2)}%</p>
        </div>
      </div>
    </div>
  );
}