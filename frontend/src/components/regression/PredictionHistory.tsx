import {
  History,
  Clock3,
} from "lucide-react";

interface PredictionItem {
  id: number;
  smiles: string;
  dose: number;
  date: string;
}

interface PredictionHistoryProps {
  history: PredictionItem[];
}

export default function PredictionHistory({
  history,
}: PredictionHistoryProps) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <div className="flex items-center gap-3 mb-6">

        <History className="text-blue-600" />

        <h2 className="text-xl font-semibold">
          Recent Predictions
        </h2>

      </div>

      {history.length === 0 ? (
        <div className="text-center py-10 text-slate-500">

          <History
            size={50}
            className="mx-auto mb-4 text-slate-300"
          />

          No prediction history available.

        </div>
      ) : (
        <div className="space-y-4">

          {history.map((item) => (
            <div
              key={item.id}
              className="rounded-xl border border-slate-200 p-4 hover:bg-slate-50 transition"
            >
              <div className="flex justify-between">

                <div>

                  <p className="text-xs text-slate-500">
                    SMILES
                  </p>

                  <p className="text-sm break-all">
                    {item.smiles}
                  </p>

                </div>

                <div className="text-right">

                  <p className="font-semibold">
                    {item.dose} nM
                  </p>

                  <div className="flex items-center gap-1 justify-end mt-2 text-xs text-slate-500">

                    <Clock3 size={14} />

                    {item.date}

                  </div>

                </div>

              </div>
            </div>
          ))}

        </div>
      )}

    </div>
  );
}