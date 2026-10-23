import {
  FlaskConical,
  Activity,
  CircleCheck,
} from "lucide-react";

interface PredictionSummaryProps {
  smiles: string;
  dose: number;
  herg: number;
  nav: number;
  cav: number;
}

export default function PredictionSummary({
  smiles,
  dose,
  herg,
  nav,
  cav,
}: PredictionSummaryProps) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <h2 className="text-xl font-semibold mb-6">
        Prediction Summary
      </h2>

      <div className="space-y-4">

        <div className="flex items-center gap-3">
          <FlaskConical className="text-blue-600" />
          <span className="font-medium">
            Dose
          </span>

          <span className="ml-auto">
            {dose} nM
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Activity className="text-red-500" />

          <span>
            Predicted hERG IC50
          </span>

          <span className="ml-auto font-semibold">
            {herg.toFixed(2)} nM
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Activity className="text-blue-500" />

          <span>
            Predicted NaV IC50
          </span>

          <span className="ml-auto font-semibold">
            {nav.toFixed(2)} nM
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Activity className="text-teal-500" />

          <span>
            Predicted CaV IC50
          </span>

          <span className="ml-auto font-semibold">
            {cav.toFixed(2)} nM
          </span>
        </div>

        <div className="border-t pt-4">

          <CircleCheck className="inline mr-2 text-green-500" />

          <span className="font-medium text-slate-700">
            Regression Completed Successfully
          </span>

        </div>

        <div className="bg-slate-50 rounded-xl p-4 mt-3">

          <p className="text-xs text-slate-500 mb-2">
            SMILES
          </p>

          <p className="break-all text-sm">
            {smiles}
          </p>

        </div>

      </div>

    </div>
  );
}