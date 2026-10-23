
interface DoseResponseCardProps {
  dose: number;
  ikr: number;
  ina: number;
  ical: number;
}

export default function DoseResponseCard({
  dose,
  ikr,
  ina,
  ical,
}: DoseResponseCardProps) {
  const blocks = [
    {
      name: "IKr",
      value: ikr,
      color: "bg-red-500",
    },
    {
      name: "INa",
      value: ina,
      color: "bg-blue-500",
    },
    {
      name: "ICaL",
      value: ical,
      color: "bg-teal-500",
    },
  ];

  return (
    <div className="bg-white rounded-2xl shadow-md p-6">

      <div className="flex justify-between items-center mb-6">

        <h2 className="text-xl font-semibold">
          Dose Response
        </h2>

        <span className="text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full">
          {dose} nM
        </span>

      </div>

      <div className="space-y-5">

        {blocks.map((block) => (
          <div key={block.name}>

            <div className="flex justify-between mb-2">

              <span>{block.name}</span>

              <span className="font-semibold">
                {block.value.toFixed(2)}%
              </span>

            </div>

            <div className="w-full bg-slate-200 rounded-full h-3">

              <div
                className={`${block.color} h-3 rounded-full transition-all duration-500`}
                style={{
                  width: `${Math.min(block.value, 100)}%`,
                }}
              />

            </div>

          </div>
        ))}

      </div>

    </div>
  );
}