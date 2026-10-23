import { Atom } from "lucide-react";

interface ChemicalFeature {
  fragment: string;
  importance: number;
}

interface Props {
  features: ChemicalFeature[];
}

export default function ChemicalXAICard({
  features,
}: Props) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <div className="flex items-center gap-3 mb-5">
        <Atom className="text-blue-600" />
        <h2 className="text-xl font-semibold">
          Chemical XAI
        </h2>
      </div>

      <p className="text-slate-500 mb-5">
        Toxic molecular fragments contributing to prediction.
      </p>

      <div className="space-y-4">
        {features.map((feature, index) => (
          <div key={index}>
            <div className="flex justify-between mb-2">
              <span>{feature.fragment}</span>
              <span>{feature.importance.toFixed(2)}</span>
            </div>

            <div className="w-full h-3 rounded-full bg-slate-200">
              <div
                className="bg-red-500 h-3 rounded-full"
                style={{
                  width: `${Math.min(
                    feature.importance * 100,
                    100
                  )}%`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}