import peaceLogo from "../../../peace.png";

export default function Layout({ page, onNavigate, children }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="brand-button"
          type="button"
          onClick={() => onNavigate("text")}
          aria-label="Open Text Check"
        >
          <span className="brand-mark">
            <img src={peaceLogo} alt="" aria-hidden="true" />
          </span>
          <span>
            <span className="brand-name">Comment Safety Monitor</span>
            <span className="brand-subtitle">
              AI-assisted comment review and risk prioritisation
            </span>
          </span>
        </button>

        <nav className="nav-links" aria-label="Primary navigation">
          <button
            type="button"
            className={`nav-link ${page === "text" ? "active" : ""}`}
            onClick={() => onNavigate("text")}
          >
            Text Check
          </button>
          <button
            type="button"
            className={`nav-link ${page === "file" ? "active" : ""}`}
            onClick={() => onNavigate("file")}
          >
            File Review
          </button>
        </nav>

        <div className="status-pill" aria-label="System ready">
          <span className="status-dot" />
          System ready
        </div>
      </header>

      <main>{children}</main>

      <footer className="footer-note">
        Designed for human review support. This tool does not make automatic
        enforcement decisions.
      </footer>
    </div>
  );
}
