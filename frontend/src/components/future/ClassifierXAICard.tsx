import { BrainCircuit } from "lucide-react";

interface Feature {
  name: string;
  value: number;
}

interface Props {
  features: Feature[];
}

export default function ClassifierXAICard({
  features,
}: Props) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <div className="flex items-center gap-3 mb-5">
        <BrainCircuit className="text-purple-600" />
        <h2 className="text-xl font-semibold">
          Classifier SHAP Explanation
        </h2>
      </div>

      <p className="text-slate-500 mb-5">
        Most influential features affecting the predicted class.
      </p>

      <div className="space-y-4">
        {features.map((feature, index) => (
          <div key={index}>
            <div className="flex justify-between mb-2">
              <span>{feature.name}</span>
              <span>{feature.value.toFixed(3)}</span>
            </div>

            <div className="w-full bg-slate-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full ${
                  feature.value >= 0
                    ? "bg-blue-600"
                    : "bg-red-500"
                }`}
                style={{
                  width: `${Math.min(
                    Math.abs(feature.value) * 100,
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