function clamp(value, min, max) {
  return Math.min(Math.max(Number(value), min), max);
}

function collectCueSpans(text, cues) {
  const spans = [];
  const length = text.length;

  for (const cue of cues || []) {
    const positions = Array.isArray(cue.positions)
      ? cue.positions
      : [{ start: cue.start, end: cue.end }];

    for (const position of positions) {
      const start = clamp(position.start, 0, length);
      const end = clamp(position.end, 0, length);
      if (end > start) {
        spans.push({
          start,
          end,
          keyword: cue.keyword,
          category: cue.category,
          severity: cue.severity
        });
      }
    }
  }

  spans.sort((a, b) => a.start - b.start || b.end - a.end);

  const filtered = [];
  let cursor = 0;
  for (const span of spans) {
    if (span.start >= cursor) {
      filtered.push(span);
      cursor = span.end;
    }
  }
  return filtered;
}

export default function HighlightedComment({ text, cues }) {
  const source = String(text || "");
  const spans = collectCueSpans(source, cues);

  if (!source) {
    return <div className="highlighted-comment muted">No comment text available.</div>;
  }

  if (!spans.length) {
    return <div className="highlighted-comment">{source}</div>;
  }

  const parts = [];
  let cursor = 0;

  spans.forEach((span, index) => {
    if (span.start > cursor) {
      parts.push(source.slice(cursor, span.start));
    }
    parts.push(
      <mark
        className="cue-highlight"
        key={`${span.start}-${span.end}-${index}`}
        title={`${span.category || "Review signal"} - ${span.severity || "Cue"}`}
      >
        {source.slice(span.start, span.end)}
      </mark>
    );
    cursor = span.end;
  });

  if (cursor < source.length) {
    parts.push(source.slice(cursor));
  }

  return <div className="highlighted-comment">{parts}</div>;
}
