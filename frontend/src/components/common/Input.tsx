import React from "react";

interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
}

export default function Input({
  label,
  className = "",
  ...props
}: InputProps) {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-sm font-medium text-slate-700">
          {label}
        </label>
      )}

      <input
        className={`w-full rounded-xl border border-slate-300 px-4 py-3
        focus:outline-none
        focus:ring-2
        focus:ring-blue-500
        focus:border-blue-500
        ${className}`}
        {...props}
      />
    </div>
  );
}