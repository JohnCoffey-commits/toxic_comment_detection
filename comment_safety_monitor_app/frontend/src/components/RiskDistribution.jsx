const DISTRIBUTION = [
  { key: "high_risk", label: "High", className: "high" },
  { key: "medium_risk", label: "Medium", className: "medium" },
  { key: "borderline", label: "Borderline", className: "borderline" },
  { key: "low_risk", label: "Low", className: "low" }
];

export default function RiskDistribution({ summary }) {
  const total = Number(summary?.total_comments || 0);

  return (
    <section className="panel distribution-panel">
      <div className="section-heading">
        <h2>Risk distribution</h2>
        <p>Comment volume by review priority band.</p>
      </div>

      <div className="distribution-list">
        {DISTRIBUTION.map((row) => {
          const count = Number(summary?.[row.key] || 0);
          const width = total ? `${Math.max((count / total) * 100, 2)}%` : "0%";
          return (
            <div className="distribution-row" key={row.key}>
              <div className="distribution-label">
                <span>{row.label}</span>
                <strong>{count}</strong>
              </div>
              <div className="distribution-track" aria-hidden="true">
                <span
                  className={`distribution-fill distribution-${row.className}`}
                  style={{ "--bar-width": width }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
