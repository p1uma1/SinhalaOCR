import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  id: string;
  label: string;
  hint: string;
  icon: string;
  accept?: string;
  maxSizeMb?: number;
  onFileSelect: (file: File) => void;
  onError?: (message: string) => void;
}

export function UploadZone({
  id,
  label,
  hint,
  icon,
  accept = "image/png,image/jpeg,image/jpg,image/webp",
  maxSizeMb = 10,
  onFileSelect,
  onError,
}: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const validateAndSet = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) {
        onError?.("Please upload a valid image file (PNG, JPG, or WEBP).");
        return;
      }
      if (file.size > maxSizeMb * 1024 * 1024) {
        onError?.(`File is too large. Maximum size is ${maxSizeMb} MB.`);
        return;
      }
      setFileName(file.name);
      onFileSelect(file);
      const reader = new FileReader();
      reader.onload = (event) => setPreview(event.target?.result as string);
      reader.readAsDataURL(file);
    },
    [maxSizeMb, onError, onFileSelect]
  );

  const clear = () => {
    setPreview(null);
    setFileName(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="upload-zone-wrap">
      <div
        className={`upload-zone ${dragOver ? "dragover" : ""} ${preview ? "has-preview" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          validateAndSet(e.dataTransfer.files[0]);
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
        aria-label={label}
      >
        <input
          ref={inputRef}
          id={id}
          type="file"
          accept={accept}
          hidden
          onChange={(e) => validateAndSet(e.target.files?.[0])}
        />
        {!preview ? (
          <div className="upload-placeholder">
            <div className="upload-icon sinhala">{icon}</div>
            <p className="upload-title">{label}</p>
            <p className="upload-hint">{hint}</p>
            <span className="upload-meta">PNG, JPG, WEBP · up to {maxSizeMb} MB</span>
          </div>
        ) : (
          <img className="upload-preview" src={preview} alt="Upload preview" />
        )}
      </div>
      {fileName && (
        <div className="upload-footer">
          <span className="file-name">{fileName}</span>
          <button type="button" className="btn btn-ghost btn-sm" onClick={clear}>
            Remove
          </button>
        </div>
      )}
    </div>
  );
}
