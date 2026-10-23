
interface IC50CardProps {
  herg: number;
  nav: number;
  cav: number;
}

export default function IC50Card({
  herg,
  nav,
  cav,
}: IC50CardProps) {
  const channels = [
    {
      name: "hERG (IKr)",
      value: herg,
      color: "text-red-600",
    },
    {
      name: "NaV1.5",
      value: nav,
      color: "text-blue-600",
    },
    {
      name: "CaV1.2",
      value: cav,
      color: "text-teal-600",
    },
  ];

  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <h2 className="text-xl font-semibold mb-6">
        Predicted IC50
      </h2>

      <div className="grid md:grid-cols-3 gap-5">

        {channels.map((channel) => (
          <div
            key={channel.name}
            className="rounded-xl border border-slate-200 p-5 text-center"
          >
            <p className="text-slate-500">
              {channel.name}
            </p>

            <h3
              className={`mt-3 text-2xl font-bold ${channel.color}`}
            >
              {channel.value.toFixed(2)}
            </h3>

            <span className="text-sm text-slate-500">
              nM
            </span>
          </div>
        ))}

      </div>

    </div>
  );
}