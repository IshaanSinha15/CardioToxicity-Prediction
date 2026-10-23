import MainLayout from "../layouts/MainLayout";

export default function AboutPage() {

  return (
    <MainLayout>

      <div className="prose max-w-4xl mx-auto py-16">

        <h1>About CardioTox AI</h1>

        <p>

          CardioTox AI is an AI-powered cardiotoxicity prediction
          platform developed to estimate ion channel IC50 values
          using ChemBERTa and machine learning.

        </p>

        <h2>Technology</h2>

        <ul>

          <li>ChemBERTa Regression</li>

          <li>Hill Equation</li>

          <li>Machine Learning</li>

          <li>ORd Cardiac Model</li>

          <li>Explainable AI (SHAP)</li>

        </ul>

      </div>

    </MainLayout>
  );
}