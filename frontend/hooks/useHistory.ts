import { useState } from "react";

export interface HistoryItem {
  id: number;
  smiles: string;
  dose: number;
  date: string;
}

const STORAGE_KEY = "cardiotox_history";

export default function useHistory() {
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? (JSON.parse(saved) as HistoryItem[]) : [];
    } catch {
      return [];
    }
  });

  const savePrediction = (
    smiles: string,
    dose: number
  ) => {
    const item: HistoryItem = {
      id: Date.now(),
      smiles,
      dose,
      date: new Date().toLocaleString(),
    };

    const updated = [item, ...history];

    setHistory(updated);

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(updated)
    );
  };

  const clearHistory = () => {
    localStorage.removeItem(STORAGE_KEY);
    setHistory([]);
  };

  const removePrediction = (id: number) => {
    const updated = history.filter(
      (item) => item.id !== id
    );

    setHistory(updated);

    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(updated)
    );
  };

  return {
    history,
    savePrediction,
    removePrediction,
    clearHistory,
  };
}