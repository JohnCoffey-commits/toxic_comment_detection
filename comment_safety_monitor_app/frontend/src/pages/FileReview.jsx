import { useMemo, useState } from "react";
import { analyzeFile, inspectFile } from "../api.js";
import AdvancedDetails from "../components/AdvancedDetails.jsx";
import RiskDistribution from "../components/RiskDistribution.jsx";
import ReviewQueueCard from "../components/ReviewQueueCard.jsx";
import UploadPanel from "../components/UploadPanel.jsx";

const RISK_FILTERS = ["All", "High", "Medium", "Borderline", "Low"];
const LIMITS = [
  { label: "Top 10", value: "10" },
  { label: "Top 25", value: "25" },
  { label: "All", value: "all" }
];

function csvEscape(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

function cueSummary(cues) {
  if (!Array.isArray(cues) || cues.length === 0) {
    return "";
  }
  return cues
    .map((cue) => `${cue.keyword} (${cue.category}, ${cue.severity})${cue.count > 1 ? ` x${cue.count}` : ""}`)
    .join("; ");
}

function downloadReviewQueue(items) {
  const headers = [
    "row_id",
    "original_text",
    "prediction",
    "risk_level",
    "risk_score",
    "risk_score_percent",
    "review_recommendation",
    "review_priority",
    "keyword_cues",
    "toxic_probability",
    "non_toxic_probability",
    "confidence",
    "threshold",
    "cleaned_text"
  ];

  const rows = items.map((item) => {
    const advanced = item.advanced || {};
    return [
      item.row_id,
      item.original_text,
      item.prediction,
      item.risk_level,
      item.risk_score,
      item.risk_score_percent,
      item.review_recommendation,
      item.review_priority,
      cueSummary(item.detected_cues),
      advanced.toxic_probability,
      advanced.non_toxic_probability,
      advanced.confidence,
      advanced.threshold,
      advanced.cleaned_text
    ].map(csvEscape).join(",");
  });

  const blob = new Blob([[headers.join(","), ...rows].join("\n")], {
    type: "text/csv;charset=utf-8"
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "comment_safety_review_queue.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function SummaryCards({ summary }) {
  const cards = [
    ["Total comments", summary.total_comments],
    ["Needs review", summary.needs_review],
    ["High risk", summary.high_risk],
    ["Average risk score", summary.average_risk_score_percent]
  ];

  return (
    <section className="summary-grid" aria-label="File analysis summary">
      {cards.map(([label, value]) => (
        <div className="summary-card hover-lift" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}

function AdvancedOutputTable({ items }) {
  return (
    <div className="table-wrap">
      <table className="advanced-table">
        <thead>
          <tr>
            <th>row_id</th>
            <th>prediction</th>
            <th>risk_level</th>
            <th>risk_score</th>
            <th>toxic_probability</th>
            <th>non_toxic_probability</th>
            <th>confidence</th>
            <th>threshold</th>
            <th>cleaned_text</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => {
            const advanced = item.advanced || {};
            return (
              <tr key={`${item.row_id}-${item.risk_score}`}>
                <td>{item.row_id}</td>
                <td>{item.prediction}</td>
                <td>{item.risk_level}</td>
                <td>{item.risk_score}</td>
                <td>{advanced.toxic_probability}</td>
                <td>{advanced.non_toxic_probability}</td>
                <td>{advanced.confidence}</td>
                <td>{advanced.threshold}</td>
                <td>{advanced.cleaned_text}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function FileReview() {
  const [file, setFile] = useState(null);
  const [inspectInfo, setInspectInfo] = useState(null);
  const [textColumn, setTextColumn] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [inspecting, setInspecting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [riskFilter, setRiskFilter] = useState("All");
  const [displayLimit, setDisplayLimit] = useState("10");

  async function handleFileSelect(selectedFile) {
    setFile(selectedFile);
    setInspectInfo(null);
    setTextColumn("");
    setResult(null);
    setError("");
    setInspecting(true);

    try {
      const data = await inspectFile(selectedFile);
      setInspectInfo(data);
      if (data.default_text_column) {
        setTextColumn(data.default_text_column);
      } else if (data.file_type === "csv" && data.columns?.length) {
        setTextColumn(data.columns[0]);
      }
    } catch (apiError) {
      setError(apiError.message);
    } finally {
      setInspecting(false);
    }
  }

  async function handleAnalyze() {
    setError("");
    if (!file) {
      setError("Please upload a CSV or TXT file before analysis.");
      return;
    }
    if (inspectInfo?.file_type === "csv" && inspectInfo.columns?.length > 1 && !textColumn) {
      setError("Select the comment text column before analysis.");
      return;
    }

    setLoading(true);
    try {
      const data = await analyzeFile(file, inspectInfo?.file_type === "csv" ? textColumn : undefined);
      setResult(data);
      setRiskFilter("All");
      setDisplayLimit("10");
    } catch (apiError) {
      setResult(null);
      setError(apiError.message);
    } finally {
      setLoading(false);
    }
  }

  const filteredItems = useMemo(() => {
    const items = result?.items || [];
    const filtered =
      riskFilter === "All" ? items : items.filter((item) => item.risk_level === riskFilter);
    return displayLimit === "all" ? filtered : filtered.slice(0, Number(displayLimit));
  }, [result, riskFilter, displayLimit]);

  return (
    <div className="page-shell fade-in">
      <section className="page-intro">
        <p className="eyebrow">File Review</p>
        <h1>Review uploaded comments</h1>
        <p>Upload a CSV or TXT file to prioritise comments for human review.</p>
      </section>

      <div className="file-workflow-grid">
        <UploadPanel
          file={file}
          inspectInfo={inspectInfo}
          inspecting={inspecting}
          onFileSelect={handleFileSelect}
        />

        <section className="panel workflow-panel">
          <div className="section-heading">
            <h2>Prepare review queue</h2>
            <p>Inspect the upload, select the text source, then analyze the file.</p>
          </div>

          {inspectInfo?.file_type === "csv" && inspectInfo.columns?.length > 1 ? (
            <label className="select-label" htmlFor="text-column">
              Select comment text column
              <select
                id="text-column"
                value={textColumn}
                onChange={(event) => setTextColumn(event.target.value)}
              >
                {inspectInfo.columns.map((column) => (
                  <option key={column} value={column}>
                    {column}
                  </option>
                ))}
              </select>
            </label>
          ) : inspectInfo?.file_type === "csv" ? (
            <p className="helper-pill">Using CSV column: {textColumn}</p>
          ) : inspectInfo?.file_type === "txt" ? (
            <p className="helper-pill">Each non-empty line will be treated as one comment.</p>
          ) : (
            <p className="helper-text">Upload a file to inspect available text fields.</p>
          )}

          {error ? <p className="error-message">{error}</p> : null}

          <button
            type="button"
            className="primary-button"
            onClick={handleAnalyze}
            disabled={!file || inspecting || loading}
          >
            {loading ? (
              <>
                <span className="button-spinner" />
                Analyzing comments...
              </>
            ) : (
              "Analyze file"
            )}
          </button>
        </section>
      </div>

      {result ? (
        <>
          <SummaryCards summary={result.summary} />
          <RiskDistribution summary={result.summary} />

          <section className="panel queue-panel slide-up">
            <div className="queue-heading-row">
              <div className="section-heading">
                <h2>Prioritised review queue</h2>
                <p>Sorted by highest risk score</p>
              </div>
              <button
                type="button"
                className="secondary-button download-button"
                onClick={() => downloadReviewQueue(result.items || [])}
              >
                Download review queue CSV
              </button>
            </div>

            <div className="queue-controls">
              <div className="segmented-control" aria-label="Filter by risk level">
                {RISK_FILTERS.map((filter) => (
                  <button
                    type="button"
                    key={filter}
                    className={riskFilter === filter ? "active" : ""}
                    onClick={() => setRiskFilter(filter)}
                  >
                    {filter}
                  </button>
                ))}
              </div>
              <div className="segmented-control" aria-label="Display limit">
                {LIMITS.map((limit) => (
                  <button
                    type="button"
                    key={limit.value}
                    className={displayLimit === limit.value ? "active" : ""}
                    onClick={() => setDisplayLimit(limit.value)}
                  >
                    {limit.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="queue-list">
              {filteredItems.length ? (
                filteredItems.map((item, index) => (
                  <ReviewQueueCard item={item} rank={index + 1} key={`${item.row_id}-${index}`} />
                ))
              ) : (
                <p className="no-cues">No comments match the selected filter.</p>
              )}
            </div>
          </section>

          <AdvancedDetails title="Advanced output table">
            <AdvancedOutputTable items={result.items || []} />
          </AdvancedDetails>
        </>
      ) : null}
    </div>
  );
}
