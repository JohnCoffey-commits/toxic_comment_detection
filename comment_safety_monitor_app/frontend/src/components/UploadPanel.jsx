import { useRef, useState } from "react";

export default function UploadPanel({ file, inspectInfo, inspecting, onFileSelect }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files) {
    const selected = files?.[0];
    if (selected) {
      onFileSelect(selected);
    }
  }

  return (
    <section className="panel upload-panel hover-lift">
      <div className="section-heading">
        <h2>Upload comments</h2>
        <p>Upload a file to generate a prioritised human review queue.</p>
      </div>

      <div
        className={`drop-zone ${dragging ? "dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          handleFiles(event.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          className="file-input"
          type="file"
          accept=".csv,.txt,text/csv,text/plain"
          onChange={(event) => handleFiles(event.target.files)}
        />
        <span className="upload-icon">CSV</span>
        <h3>Drag a CSV or TXT file here</h3>
        <p>Supported formats: CSV, TXT</p>
        <button type="button" className="secondary-button" onClick={() => inputRef.current?.click()}>
          Choose file
        </button>
      </div>

      {file ? (
        <div className="file-meta">
          <div>
            <strong>{file.name}</strong>
            <span>{(file.size / 1024).toFixed(1)} KB</span>
          </div>
          {inspecting ? (
            <span className="inline-loading">
              <span className="mini-dot" />
              Inspecting file...
            </span>
          ) : inspectInfo ? (
            <span>{inspectInfo.valid_comment_count} valid comments detected</span>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
