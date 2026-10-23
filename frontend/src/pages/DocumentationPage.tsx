import MainLayout from "../layouts/MainLayout";

export default function DocumentationPage() {

  return (
    <MainLayout>

      <div className="max-w-5xl mx-auto py-16">

        <h1 className="text-4xl font-bold mb-8">
          Documentation
        </h1>

        <div className="bg-white rounded-2xl shadow p-8">

          <h2 className="text-xl font-semibold mb-4">
            Prediction Pipeline
          </h2>

          <pre className="bg-slate-100 rounded-xl p-6 overflow-x-auto">

{`SMILES
   ↓
ChemBERTa
   ↓
IC50 Prediction
   ↓
Hill Equation
   ↓
Dose Response
   ↓
Regression Output`}

          </pre>

        </div>

      </div>

    </MainLayout>
  );
}