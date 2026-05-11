import CueChips from "./CueChips.jsx";
import HighlightedComment from "./HighlightedComment.jsx";
import RiskBadge from "./RiskBadge.jsx";

export default function ReviewQueueCard({ item, rank }) {
  const cues = item?.detected_cues || [];

  return (
    <article className="queue-card hover-lift">
      <div className="queue-rank">#{rank}</div>
      <div className="queue-content">
        <div className="queue-topline">
          <RiskBadge level={item.risk_level} />
          <span className="queue-score">{item.risk_score_percent}</span>
          <span className="queue-recommendation">{item.review_recommendation}</span>
        </div>
        <HighlightedComment text={item.original_text} cues={cues} />
        {cues.length ? (
          <CueChips cues={cues} />
        ) : (
          <p className="no-cues compact">
            No predefined language cues were detected. The risk score may still
            reflect broader sentence context.
          </p>
        )}
      </div>
    </article>
  );
}
