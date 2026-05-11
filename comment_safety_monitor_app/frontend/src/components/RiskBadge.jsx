function riskClass(level) {
  return String(level || "Low").toLowerCase().replace(/\s+/g, "-");
}

function riskLabel(level) {
  if (level === "Borderline") {
    return "Borderline";
  }
  return `${level || "Low"} Risk`;
}

export default function RiskBadge({ level }) {
  return <span className={`risk-badge risk-${riskClass(level)}`}>{riskLabel(level)}</span>;
}
