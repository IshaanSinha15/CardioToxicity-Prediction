export function formatNumber(
  value: number,
  decimals = 2
): string {
  if (Number.isNaN(value)) return "-";
  return value.toFixed(decimals);
}

export function formatPercent(
  value: number,
  decimals = 2
): string {
  return `${value.toFixed(decimals)}%`;
}

export function formatDose(
  dose: number
): string {
  return `${dose.toFixed(2)} nM`;
}

export function formatIC50(
  value: number
): string {
  return `${value.toFixed(2)} nM`;
}

export function shortenSmiles(
  smiles: string,
  maxLength = 40
): string {
  if (!smiles) return "";

  if (smiles.length <= maxLength) {
    return smiles;
  }

  return (
    smiles.substring(0, maxLength) + "..."
  );
}

export function capitalize(
  text: string
): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function formatDate(
  date: Date | string
): string {
  return new Date(date).toLocaleString();
}