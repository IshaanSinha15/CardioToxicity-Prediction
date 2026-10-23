import PredictionHistory from "../components/regression/PredictionHistory";

export default function HistoryPage() {
  return (
    <div className="page-placeholder">
      <h1 className="text-3xl font-bold mb-8">Prediction History</h1>
      <PredictionHistory history={[]} />
    </div>
  );
}
