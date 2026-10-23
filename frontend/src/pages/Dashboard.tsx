export default function Dashboard() {
  return (
    <div className="page-placeholder">
      <h1 className="text-3xl font-bold mb-8">Dashboard</h1>
      <div className="grid md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl shadow p-6">
          <h2 className="font-semibold">Total Predictions</h2>
          <p className="text-4xl font-bold mt-4">0</p>
        </div>
        <div className="bg-white rounded-2xl shadow p-6">
          <h2 className="font-semibold">Average IC50</h2>
          <p className="text-4xl font-bold mt-4">—</p>
        </div>
        <div className="bg-white rounded-2xl shadow p-6">
          <h2 className="font-semibold">API Status</h2>
          <p className="text-green-600 font-bold mt-4">Online</p>
        </div>
      </div>
    </div>
  );
}
