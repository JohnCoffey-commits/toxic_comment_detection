export default function AdvancedDetails({ title = "Advanced details", children }) {
  return (
    <details className="advanced-details">
      <summary>{title}</summary>
      <div className="advanced-body">{children}</div>
    </details>
  );
}
