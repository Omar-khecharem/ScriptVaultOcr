import { CheckIcon, XIcon } from "./icons.jsx";

export default function FileList({ files, selectedId, onSelect }) {
  if (files.length === 0) return null;

  return (
    <ul className="file-list" aria-label="Fichiers">
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
                <span style={{ width: `${Math.round((file.progress || 0) * 100)}%` }} />
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

function extOf(name) {
  const dot = name.lastIndexOf(".");
  const ext = dot !== -1 ? name.slice(dot + 1).toUpperCase() : "FILE";
  return ext.slice(0, 4);
}

function metaOf(file) {
  if (file.status === "done") {
    const count = file.result.pages.length;
    return `${count} page${count > 1 ? "s" : ""}`;
  }
  if (file.status === "processing") {
    return `${Math.round((file.progress || 0) * 100)}%`;
  }
  return "";
}

function StatusChip({ file }) {
  if (file.status === "done") {
    return (
      <span className="chip chip-ok">
        <CheckIcon size={12} />
        {(file.result.confidence * 100).toFixed(0)}%
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
        {Math.round((file.progress || 0) * 100)}%
      </span>
    );
  }
  return <span className="chip chip-wait">En attente</span>;
}
