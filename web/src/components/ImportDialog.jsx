// ============================================================================
// ImportDialog — création d'un lot (Windows Fluent) : dépôt multiple, nom,
// langue, prétraitement, progression d'envoi.
// ============================================================================

import { useEffect, useMemo, useRef, useState } from "react";
import { isSupportedFile, LANGS } from "../api/client.js";
import { PlusIcon, UploadIcon, XIcon } from "./icons.jsx";

export default function ImportDialog({
  open,
  lang,
  preprocess,
  uploading,
  progress,
  onLangChange,
  onPreprocessChange,
  onCreate,
  onClose,
}) {
  const [name, setName] = useState("");
  const [files, setFiles] = useState([]);
  const inputRef = useRef(null);
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (open) {
      setName("");
      setFiles([]);
    }
  }, [open]);

  const accepted = useMemo(() => files.filter((f) => isSupportedFile(f.name)), [files]);
  const rejected = files.length - accepted.length;

  if (!open) return null;

  function addFiles(list) {
    const next = Array.from(list || []);
    if (next.length) setFiles((prev) => [...prev, ...next]);
  }

  function handleCreate() {
    if (!accepted.length || uploading) return;
    const label = name.trim() || `Lot du ${new Date().toLocaleTimeString("fr-FR", {
      hour: "2-digit",
      minute: "2-digit",
    })} (${accepted.length})`;
    onCreate(accepted, label);
  }

  return (
    <div className="dialog-overlay" role="presentation" onMouseDown={onClose}>
      <div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-label="Créer un lot"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="dialog-head">
          <div className="dialog-title-wrap">
            <span className="dialog-icon">
              <UploadIcon size={18} />
            </span>
            <div>
              <h3 className="dialog-title">Créer un lot de documents</h3>
              <p className="dialog-sub">
                Traitement OCR 100&nbsp;% local — les milliers de pages sont analysées en
                arrière-plan.
              </p>
            </div>
          </div>
          <button type="button" className="btn btn-icon" onClick={onClose} title="Fermer (Échap)">
            <XIcon />
          </button>
        </div>

        <div className="dialog-body">
          <div
            className={`import-drop${dragging ? " dragover" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
            }}
            onDragEnter={(event) => {
              event.preventDefault();
              dragDepth.current += 1;
              setDragging(true);
            }}
            onDragLeave={(event) => {
              event.preventDefault();
              dragDepth.current = Math.max(0, dragDepth.current - 1);
              if (dragDepth.current === 0) setDragging(false);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              dragDepth.current = 0;
              setDragging(false);
              addFiles(event.dataTransfer.files);
            }}
          >
            <span className="import-drop-icon">
              <PlusIcon size={22} />
            </span>
            <span className="import-drop-title">
              {dragging ? "Déposez pour ajouter" : "Glissez vos documents ici"}
            </span>
            <span className="import-drop-hint">
              PNG · JPG · TIFF · WebP · BMP · PDF — jusqu'à 10&nbsp;000 fichiers par lot
            </span>
            <button
              type="button"
              className="btn"
              onClick={(event) => {
                event.stopPropagation();
                inputRef.current?.click();
              }}
            >
              Parcourir…
            </button>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp,.pdf"
              style={{ display: "none" }}
              onChange={(event) => {
                addFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </div>

          {files.length > 0 && (
            <div className="import-count">
              <span className="import-count-ok">
                {accepted.length} fichier{accepted.length > 1 ? "s" : ""} accepté
                {accepted.length > 1 ? "s" : ""}
              </span>
              {rejected > 0 && (
                <span className="import-count-ko">
                  {rejected} rejeté{rejected > 1 ? "s" : ""}
                </span>
              )}
            </div>
          )}

          <div className="dialog-field-row">
            <label className="dialog-field">
              <span className="dialog-label">Nom du lot</span>
              <input
                type="text"
                className="input"
                value={name}
                placeholder="Optionnel — ex. Session 2026, centre de Sfax…"
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <label className="dialog-field dialog-field--narrow">
              <span className="dialog-label">Langue</span>
              <select value={lang} onChange={(event) => onLangChange(event.target.value)}>
                {LANGS.map((code) => (
                  <option key={code} value={code}>
                    {code}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="switch">
            <input
              type="checkbox"
              checked={preprocess}
              onChange={(event) => onPreprocessChange(event.target.checked)}
            />
            <span className="switch-track" />
            <span className="switch-text">Prétraitement OpenCV (CLAHE · redressement)</span>
          </label>
        </div>

        <div className="dialog-foot">
          {uploading && (
            <div className="import-progress">
              <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)}>
                <div className="bar" style={{ width: `${progress * 100}%` }} />
              </div>
              <span className="import-progress-text">Envoi… {Math.round(progress * 100)}%</span>
            </div>
          )}
          <span className="spacer" />
          <button type="button" className="btn" onClick={onClose} disabled={uploading}>
            Annuler
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleCreate}
            disabled={accepted.length === 0 || uploading}
          >
            <UploadIcon size={14} />
            <span>Créer le lot ({accepted.length})</span>
          </button>
        </div>
      </div>
    </div>
  );
}