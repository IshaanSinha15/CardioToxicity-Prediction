import {
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Info,
} from "lucide-react";

interface StatusBadgeProps {
  status:
    | "success"
    | "warning"
    | "error"
    | "info";

  text: string;
}

const styles = {
  success: {
    bg: "bg-green-100",
    text: "text-green-700",
    icon: CheckCircle2,
  },

  warning: {
    bg: "bg-yellow-100",
    text: "text-yellow-700",
    icon: AlertTriangle,
  },

  error: {
    bg: "bg-red-100",
    text: "text-red-700",
    icon: XCircle,
  },

  info: {
    bg: "bg-blue-100",
    text: "text-blue-700",
    icon: Info,
  },
};

export default function StatusBadge({
  status,
  text,
}: StatusBadgeProps) {
  const config = styles[status];
  const Icon = config.icon;

  return (
    <div
      className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${config.bg}`}
    >
      <Icon
        size={18}
        className={config.text}
      />

      <span
        className={`text-sm font-semibold ${config.text}`}
      >
        {text}
      </span>
    </div>
  );
}