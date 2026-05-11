import { useState } from "react";
import { analyzeText } from "../api.js";
import AdvancedDetails from "../components/AdvancedDetails.jsx";
import CueChips from "../components/CueChips.jsx";
import HighlightedComment from "../components/HighlightedComment.jsx";
import RiskScoreCard from "../components/RiskScoreCard.jsx";
import { sampleComments } from "../data/sampleComments.js";

function AdvancedKeyValue({ result }) {
  const advanced = result?.advanced || {};
  const rows = [
    ["predicted_label", advanced.predicted_label],
    ["numeric_label", advanced.numeric_label],
    ["risk_score", result?.risk_score],
    ["toxic_probability", advanced.toxic_probability],
    ["non_toxic_probability", advanced.non_toxic_probability],
    ["confidence", advanced.confidence],
    ["threshold", advanced.threshold],
    ["cleaned_text", advanced.cleaned_text]
  ];

  return (
    <dl className="advanced-grid">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{String(value ?? "")}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function TextCheck() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleAnalyze(event) {
    event.preventDefault();
    setError("");

    if (!text.trim()) {
      setResult(null);
      setError("Please enter a comment before running the check.");
      return;
    }

    setLoading(true);
    try {
      const data = await analyzeText(text);
      setResult(data);
    } catch (apiError) {
      setResult(null);
      setError(apiError.message);
    } finally {
      setLoading(false);
    }
  }

  function handleUseSample() {
    const index = Math.floor(Math.random() * sampleComments.length);
    setText(sampleComments[index]);
    setResult(null);
    setError("");
  }

  const cues = result?.detected_cues || [];

  return (
    <div className="page-shell fade-in">
      <section className="page-intro">
        <p className="eyebrow">Text Check</p>
        <h1>Check a comment</h1>
        <p>Paste a comment to assess whether it should be prioritised for human review.</p>
      </section>

      <div className="text-check-grid">
        <form className="panel input-panel hover-lift" onSubmit={handleAnalyze}>
          <div className="input-label-row">
            <label htmlFor="comment-text">Comment text</label>
            <button
              className="sample-button"
              type="button"
              onClick={handleUseSample}
              disabled={loading}
            >
              Use sample
            </button>
          </div>
          <textarea
            id="comment-text"
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Paste a comment to assess potential safety risk..."
            rows={11}
          />
          <p className="helper-text">
            Empty and whitespace-only comments are blocked before analysis.
          </p>
          {error ? <p className="error-message">{error}</p> : null}
          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? (
              <>
                <span className="button-spinner" />
                Scanning comment...
              </>
            ) : (
              "Analyze risk"
            )}
          </button>
        </form>

        <RiskScoreCard result={result} loading={loading} />
      </div>

      {result ? (
        <>
          <section className="panel cues-panel slide-up">
            <div className="section-heading">
              <h2>Detected language cues</h2>
              <p>
                Highlighted cues are rule-based review hints and may help reviewers
                locate potentially problematic language.
              </p>
            </div>
            {cues.length ? (
              <>
                <HighlightedComment text={result.original_text} cues={cues} />
                <CueChips cues={cues} />
              </>
            ) : (
              <p className="no-cues">
                No predefined language cues were detected. The risk score may still
                reflect broader sentence context.
              </p>
            )}
          </section>

          <AdvancedDetails title="Advanced details">
            <AdvancedKeyValue result={result} />
          </AdvancedDetails>
        </>
      ) : null}
    </div>
  );
}
