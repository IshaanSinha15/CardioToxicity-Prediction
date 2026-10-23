import { Link } from "react-router-dom";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-100 flex flex-col justify-center items-center">

      <h1 className="text-6xl font-bold text-blue-700">
        CardioTox AI
      </h1>

      <p className="mt-6 text-xl text-slate-600">
        AI-Powered Cardiotoxicity Prediction Platform
      </p>

      <Link
        to="/prediction"
        className="mt-10 bg-blue-600 text-white px-8 py-4 rounded-xl hover:bg-blue-700"
      >
        Start Prediction
      </Link>

    </div>
  );
}