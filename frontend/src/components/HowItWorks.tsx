const STEPS = [
  {
    step: "01",
    title: "Upload",
    text: "Choose line, document, or character mode and upload a PNG or JPEG.",
  },
  {
    step: "02",
    title: "Preprocess",
    text: "Documents are deskewed and split into individual line crops automatically.",
  },
  {
    step: "03",
    title: "Recognize",
    text: "Stage 1 or Stage 2 models run on GPU with production inference settings.",
  },
  {
    step: "04",
    title: "Copy & use",
    text: "Copy Unicode Sinhala text into your editor, CMS, or downstream workflow.",
  },
];

export function HowItWorks() {
  return (
    <section className="section section-muted" id="how-it-works">
      <div className="container">
        <div className="section-head">
          <p className="eyebrow">How it works</p>
          <h2>From image to සිංහල text in four steps</h2>
        </div>
        <ol className="steps-grid">
          {STEPS.map((item) => (
            <li key={item.step} className="step-card">
              <span className="step-number">{item.step}</span>
              <h3>{item.title}</h3>
              <p>{item.text}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
