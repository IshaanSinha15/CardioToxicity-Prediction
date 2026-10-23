import MainLayout from "../layouts/MainLayout";

import Button from "../components/common/Button";

export default function NotFound() {

  return (
    <MainLayout>

      <div className="flex flex-col justify-center items-center py-32">

        <h1 className="text-8xl font-bold text-blue-600">
          404
        </h1>

        <p className="text-slate-600 mt-4 mb-8">

          Page not found.

        </p>

        <Button
          onClick={() => (window.location.href = "/")}
        >
          Go Home
        </Button>

      </div>

    </MainLayout>
  );
}