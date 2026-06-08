import { useCallback, useEffect, useState } from "react";
import { fetchHealth } from "./api";
import { Features } from "./components/Features";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { HowItWorks } from "./components/HowItWorks";
import { OCRStudio } from "./components/OCRStudio";
import { Toast } from "./components/Toast";
import { useToast } from "./hooks/useToast";
import type { HealthStatus } from "./types";

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState(false);
  const { toast, showToast } = useToast();

  const scrollToStudio = useCallback(() => {
    document.getElementById("studio")?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealthError(true));
  }, []);

  return (
    <>
      <div className="bg-pattern" aria-hidden="true" />
      <Header health={health} healthError={healthError} onTryOcr={scrollToStudio} />
      <main>
        <Hero onTryOcr={scrollToStudio} />
        <Features />
        <OCRStudio health={health} showToast={showToast} />
        <HowItWorks />
      </main>
      <Footer />
      <Toast toast={toast} />
    </>
  );
}
