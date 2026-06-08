import type { ToastState } from "../hooks/useToast";

interface ToastProps {
  toast: ToastState | null;
}

export function Toast({ toast }: ToastProps) {
  if (!toast) return null;
  return (
    <div className={`toast toast-${toast.type}`} role="status" aria-live="polite">
      {toast.message}
    </div>
  );
}
