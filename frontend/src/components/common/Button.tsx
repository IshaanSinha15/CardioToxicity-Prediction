import React from "react";

type Variant = "primary" | "secondary" | "outline" | "danger";

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: React.ReactNode;
}

const variants: Record<Variant, string> = {
  primary:
    "bg-blue-600 hover:bg-blue-700 text-white shadow-lg",

  secondary:
    "bg-teal-500 hover:bg-teal-600 text-white",

  outline:
    "border border-slate-300 bg-white hover:bg-slate-100 text-slate-800",

  danger:
    "bg-red-500 hover:bg-red-600 text-white",
};

export default function Button({
  variant = "primary",
  children,
  className = "",
  ...props
}: ButtonProps) {
  return (
    <button
      className={`px-5 py-3 rounded-xl font-semibold transition duration-200 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}