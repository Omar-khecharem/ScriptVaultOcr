import { CheckIcon, XIcon } from "./icons.jsx";

const STATUS_LABELS = {
  pending: "En attente",
  processing: "En traitement",
  done: "Terminé",
  error: "Erreur",
  cancelled: "Annulé",
};

const FILE_EXT_PATTERN = /\.([a-z0-9]+)$/i;

export default function FileList({ files, selectedId, onSelect }) {
  if (files.length === 0) return null;

  return (
    <ul className="file-list" aria-label="Fichiers du lot">
      {files.map((file) => (
        <li
          key={file.id}
          className={`file-item${file.id === selectedId ? " selected" : ""}`}
          onClick={() => onSelect(file.id)}
          title={file.error ? `${file.name} — ${file.error}` : file.name}
        >
          <span className="file-ext">{extOf(file.name)}</span>
          <div className="file-info">
            <span className="file-name">{file.name}</span>
            {file.status === "processing" && (
              <span className="file-progress">
                <span className="bar" />
              </span>
            )}
          </div>
          <span className="file-meta">{metaOf(file)}</span>
          <StatusChip file={file} />
        </li>
      ))}
    </ul>
  );
}

export function FileListPager({ page, pageSize, total, onChange }) {
  const totalPages = Math.max(1, Math.ceil(total / pageSize));
  const from = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const to = Math.min(total, page * pageSize);
  return (
    <div className="pager">
      <span className="pager-count">
        {from}–{to} / {total}
      </span>
      <button
        type="button"
        className="pager-btn"
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
        title="Page précédente"
      >
        ‹
      </button>
      <button
        type="button"
        className="pager-btn"
        disabled={page >= totalPages}
        onClick={() => onChange(page + 1)}
        title="Page suivante"
      >
        ›
      </button>
    </div>
  );
}

function extOf(name) {
  const match = FILE_EXT_PATTERN.exec(name || "");
  const ext = match ? match[1].toUpperCase() : "FILE";
  return ext.slice(0, 4);
}

function metaOf(file) {
  if (file.status === "done") {
    return `${file.pages ?? 0} page${file.pages > 1 ? "s" : ""}`;
  }
  if (file.status === "error") {
    return "en échec";
  }
  return "";
}

function StatusChip({ file }) {
  if (file.status === "done") {
    return (
      <span className="chip chip-ok">
        <CheckIcon size={12} />
        {((file.confidence || 0) * 100).toFixed(0)}%
      </span>
    );
  }
  if (file.status === "error") {
    return (
      <span className="chip chip-err">
        <XIcon size={12} />
        Échec
      </span>
    );
  }
  if (file.status === "processing") {
    return (
      <span className="chip chip-run">
        <span className="spinner" />
        OCR
      </span>
    );
  }
  if (file.status === "cancelled") {
    return <span className="chip chip-wait">Annulé</span>;
  }
  return <span className="chip chip-wait">{STATUS_LABELS[file.status] || file.status}</span>;
}