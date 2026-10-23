import { FlaskConical, LoaderCircle } from "lucide-react";

interface PredictButtonProps {
  loading?: boolean;
  onClick: () => void;
}

export default function PredictButton({
  loading = false,
  onClick,
}: PredictButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="w-full flex items-center justify-center gap-3 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-semibold py-3 transition"
    >
      {loading ? (
        <>
          <LoaderCircle className="animate-spin" size={20} />
          Predicting...
        </>
      ) : (
        <>
          <FlaskConical size={20} />
          Predict IC50
        </>
      )}
    </button>
  );
}