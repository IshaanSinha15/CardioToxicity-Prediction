import type { ReactNode } from "react";
import Navbar from "../components/common/Navbar";
import Footer from "../components/common/Footer";

interface Props {
  children: ReactNode;
}

export default function MainLayout({ children }: Props) {
  return (
    <div className="min-h-screen flex flex-col bg-slate-100">
      <Navbar />

      <main className="flex-1 p-8">
        {children}
      </main>

      <Footer />
    </div>
  );
}