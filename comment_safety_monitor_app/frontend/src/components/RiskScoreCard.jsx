import RiskBadge from "./RiskBadge.jsx";

const EXPLANATIONS = {
  High: "This comment should be prioritised for human review.",
  Medium: "This comment may require review depending on moderation policy.",
  Borderline: "This comment shows some risk signals and may require context.",
  Low: "No immediate review priority detected."
};

function scoreWidth(result) {
  const score = Number(result?.risk_score ?? 0);
  return `${Math.min(Math.max(score * 100, 0), 100).toFixed(1)}%`;
}

export default function RiskScoreCard({ result, loading }) {
  if (loading) {
    return (
      <section className="panel result-panel loading-panel" aria-live="polite">
        <span className="spinner" />
        <div>
          <h2>Scanning comment...</h2>
          <p>Assessing safety risk and preparing review signals.</p>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel result-panel empty-state hover-lift">
        <span className="empty-icon" aria-hidden="true" />
        <h2>Ready to scan</h2>
        <p>
          Paste a comment and run a risk check. Results will appear here with
          review priority and detected language cues.
        </p>
      </section>
    );
  }

  const level = result.risk_level || "Low";

  return (
    <section className={`panel result-panel result-${String(level).toLowerCase()}`}>
      <div className="result-header">
        <RiskBadge level={level} />
        <span className="priority-label">Priority {result.review_priority}</span>
      </div>

      <div className="score-display">{result.risk_score_percent || "0.0%"}</div>
      <p className="score-label">Risk score</p>

      <div className="risk-meter" aria-hidden="true">
        <span
          className={`risk-meter-fill risk-fill-${String(level).toLowerCase()}`}
          style={{ "--risk-width": scoreWidth(result) }}
        />
      </div>

      <h2>{result.review_recommendation}</h2>
      <p>{EXPLANATIONS[level] || EXPLANATIONS.Low}</p>
    </section>
  );
}
