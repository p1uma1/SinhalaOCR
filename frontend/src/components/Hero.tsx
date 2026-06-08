interface HeroProps {
  onTryOcr: () => void;
}

export function Hero({ onTryOcr }: HeroProps) {
  return (
    <section className="hero" id="top">
      <div className="container hero-grid">
        <div className="hero-copy">
          <p className="eyebrow">AI-powered · Sinhala-first · Production ready</p>
          <h1>
            Turn Sinhala handwriting and documents into editable text
          </h1>
          <p className="hero-lead">
            A two-stage vision pipeline trained on Dataset454 characters and SinOCR
            handwritten &amp; printed line data — built for real-world Sinhala OCR.
          </p>
          <div className="hero-actions">
            <button type="button" className="btn btn-primary" onClick={onTryOcr}>
              Open OCR Studio
            </button>
            <a className="btn btn-secondary" href="#how-it-works">
              See how it works
            </a>
          </div>
          <div className="hero-stats" role="list">
            <div className="stat" role="listitem">
              <strong>454</strong>
              <span>Character classes</span>
            </div>
            <div className="stat" role="listitem">
              <strong>90K+</strong>
              <span>Line training samples</span>
            </div>
            <div className="stat" role="listitem">
              <strong>3</strong>
              <span>OCR workflows</span>
            </div>
          </div>
        </div>

        <div className="hero-card" aria-hidden="true">
          <div className="pipeline-card">
            <div className="pipeline-item">
              <span className="pipeline-badge">Stage 1</span>
              <h3>Character encoder</h3>
              <p>ViT learns 454 Sinhala glyphs from Dataset454</p>
            </div>
            <div className="pipeline-divider" />
            <div className="pipeline-item">
              <span className="pipeline-badge">Stage 2</span>
              <h3>Line decoder</h3>
              <p>TrOCR + SinBERT reads words and full lines</p>
            </div>
            <div className="pipeline-divider" />
            <div className="pipeline-item highlight">
              <span className="pipeline-badge">Output</span>
              <h3>සිංහල පෙළ</h3>
              <p>Editable Unicode Sinhala text you can copy or export</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
