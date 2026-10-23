import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";

interface Props {
  data: {
    dose: number;
    IKr: number;
    INa: number;
    ICaL: number;
  }[];
}

export default function DoseResponseGraph({ data }: Props) {
  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <h2 className="text-xl font-semibold mb-6">
        Dose Response Curve
      </h2>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />

          <XAxis
            dataKey="dose"
            label={{
              value: "Dose (nM)",
              position: "insideBottom",
              offset: -5,
            }}
          />

          <YAxis
            label={{
              value: "Channel Block (%)",
              angle: -90,
              position: "insideLeft",
            }}
          />

          <Tooltip />

          <Legend />

          <Line
            type="monotone"
            dataKey="IKr"
            stroke="#2563eb"
            strokeWidth={3}
          />

          <Line
            type="monotone"
            dataKey="INa"
            stroke="#14b8a6"
            strokeWidth={3}
          />

          <Line
            type="monotone"
            dataKey="ICaL"
            stroke="#ef4444"
            strokeWidth={3}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}