import React from "react";

interface TextAreaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
}

export default function TextArea({
  label,
  className = "",
  ...props
}: TextAreaProps) {
  return (
    <div className="flex flex-col gap-2">
      {label && (
        <label className="text-sm font-medium text-slate-700">
          {label}
        </label>
      )}

      <textarea
        className={`w-full min-h-[180px]
        rounded-xl
        border
        border-slate-300
        p-4
        resize-none
        focus:ring-2
        focus:ring-blue-500
        focus:border-blue-500
        focus:outline-none
        ${className}`}
        {...props}
      />
    </div>
  );
}