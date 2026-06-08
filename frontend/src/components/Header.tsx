import type { HealthStatus } from "../types";

interface HeaderProps {
  health: HealthStatus | null;
  healthError: boolean;
  onTryOcr: () => void;
}

export function Header({ health, healthError, onTryOcr }: HeaderProps) {
  return (
    <header className="site-header">
      <div className="container header-inner">
        <a className="brand" href="#top" aria-label="Sinhala OCR Studio home">
          <div className="brand-mark" aria-hidden="true">
            සි
          </div>
          <div>
            <strong className="brand-title">Sinhala OCR Studio</strong>
            <span className="brand-subtitle">Production Sinhala text recognition</span>
          </div>
        </a>

        <nav className="header-nav" aria-label="Primary">
          <a href="#features">Features</a>
          <a href="#studio">Studio</a>
          <a href="#how-it-works">How it works</a>
        </nav>

        <div className="header-actions">
          <div className="status-group" aria-live="polite">
            {healthError && <span className="pill warn">API offline</span>}
            {health && (
              <>
                <span className="pill ok">{health.device}</span>
                <span className={`pill ${health.stage1_ready ? "ok" : "warn"}`}>
                  Stage 1 {health.stage1_ready ? "ready" : "missing"}
                </span>
                <span className={`pill ${health.stage2_ready ? "ok" : "warn"}`}>
                  Stage 2 {health.stage2_ready ? "ready" : "missing"}
                </span>
              </>
            )}
            {!health && !healthError && <span className="pill">Checking models…</span>}
          </div>
          <button type="button" className="btn btn-primary btn-sm" onClick={onTryOcr}>
            Try OCR
          </button>
        </div>
      </div>
    </header>
  );
}
