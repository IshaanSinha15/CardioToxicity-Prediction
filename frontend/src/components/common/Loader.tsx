import { LoaderCircle } from "lucide-react";

interface LoaderProps {
  text?: string;
  size?: number;
}

export default function Loader({
  text = "Loading...",
  size = 40,
}: LoaderProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-8">
      <LoaderCircle
        size={size}
        className="animate-spin text-blue-600"
      />

      <p className="text-slate-600 text-sm">
        {text}
      </p>
    </div>
  );
}