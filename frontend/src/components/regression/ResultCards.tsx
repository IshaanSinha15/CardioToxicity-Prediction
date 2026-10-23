
interface ResultCardProps {
  title: string;
  value: string | number;
  unit?: string;
  color?: string;
}

export default function ResultCards({
  title,
  value,
  unit = "",
  color = "blue",
}: ResultCardProps) {
  const colors = {
    blue: "border-blue-500 bg-blue-50 text-blue-700",
    green: "border-green-500 bg-green-50 text-green-700",
    orange: "border-orange-500 bg-orange-50 text-orange-700",
    red: "border-red-500 bg-red-50 text-red-700",
  };

  return (
    <div
      className={`rounded-2xl border-l-4 shadow-sm bg-white p-5 ${
        colors[color as keyof typeof colors]
      }`}
    >
      <p className="text-sm font-medium text-slate-500">{title}</p>

      <h2 className="mt-2 text-3xl font-bold">
        {value}
        <span className="text-base ml-1">{unit}</span>
      </h2>
    </div>
  );
}