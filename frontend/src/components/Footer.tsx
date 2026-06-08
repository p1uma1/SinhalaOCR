export function Footer() {
  return (
    <footer className="site-footer">
      <div className="container footer-grid">
        <div>
          <strong className="footer-brand">Sinhala OCR Studio</strong>
          <p>
            Open research product for Sinhala optical character recognition — character
            classification, line OCR, and document transcription.
          </p>
        </div>
        <div>
          <strong>Models</strong>
          <ul>
            <li>Stage 1 · Dataset454 · ViT classifier</li>
            <li>Stage 2 · SinOCR · TrOCR + SinBERT</li>
          </ul>
        </div>
        <div>
          <strong>Product</strong>
          <ul>
            <li>
              <a href="#studio">OCR Studio</a>
            </li>
            <li>
              <a href="/api/docs" target="_blank" rel="noreferrer">
                API docs
              </a>
            </li>
            <li>
              <a href="#features">Features</a>
            </li>
          </ul>
        </div>
      </div>
      <div className="container footer-bottom">
        <span>© {new Date().getFullYear()} Sinhala OCR Studio</span>
        <span>Built for Sinhala · Powered by PyTorch &amp; Transformers</span>
      </div>
    </footer>
  );
}
