export default function CueChips({ cues }) {
  if (!Array.isArray(cues) || cues.length === 0) {
    return null;
  }

  return (
    <div className="cue-chip-list" aria-label="Detected language cues">
      {cues.map((cue) => (
        <span className="cue-chip" key={`${cue.keyword}-${cue.start}-${cue.end}`}>
          <strong>{cue.keyword}</strong>
          <span>{cue.category}</span>
          <em>{cue.severity}</em>
          {cue.count > 1 ? <small>x{cue.count}</small> : null}
        </span>
      ))}
    </div>
  );
}
