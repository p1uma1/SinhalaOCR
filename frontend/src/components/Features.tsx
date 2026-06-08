const FEATURES = [
  {
    title: "Line OCR",
    body: "Upload a single line image — the same format used in SinOCR training — and get accurate Sinhala transcription.",
    icon: "—",
  },
  {
    title: "Document OCR",
    body: "Upload a full page. We detect lines, deskew, crop, and run OCR on every line automatically.",
    icon: "▤",
  },
  {
    title: "Character recognition",
    body: "Demo the Stage 1 classifier with top-5 predictions and confidence scores.",
    icon: "අ",
  },
  {
    title: "GPU accelerated",
    body: "Optimized for NVIDIA GPUs with mixed precision and production inference serving.",
    icon: "⚡",
  },
];

export function Features() {
  return (
    <section className="section" id="features">
      <div className="container">
        <div className="section-head">
          <p className="eyebrow">Product capabilities</p>
          <h2>Built for Sinhala, not adapted from English OCR</h2>
          <p>
            Every workflow maps to how the models were actually trained — line images,
            document pages, and isolated characters.
          </p>
        </div>
        <div className="feature-grid">
          {FEATURES.map((feature) => (
            <article key={feature.title} className="feature-card">
              <div className="feature-icon sinhala">{feature.icon}</div>
              <h3>{feature.title}</h3>
              <p>{feature.body}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
