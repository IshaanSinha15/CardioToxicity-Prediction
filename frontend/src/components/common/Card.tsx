import React from "react";

interface CardProps {
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export default function Card({
  title,
  children,
  className = "",
}: CardProps) {
  return (
    <div
      className={`bg-white rounded-2xl shadow-md border border-slate-200 p-6 ${className}`}
    >
      {title && (
        <h2 className="text-lg font-semibold text-slate-800 mb-4">
          {title}
        </h2>
      )}

      {children}
    </div>
  );
}