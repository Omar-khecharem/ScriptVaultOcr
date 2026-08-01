/**
 * Liste des fichiers ajoutés avec leur statut (attente / analyse / OK / erreur).
 */
export default function FileList({ files, selectedId, onSelect }) {
  if (files.length === 0) return null;

  return (
    <ul className="file-list" aria-label="Fichiers">
      {files.map((file) => (
        <li
          key={file.id}
          className={file.id === selectedId ? "selected" : ""}
          onClick={() => onSelect(file.id)}
          title={file.error ? file.error : file.name}
        >
          <StatusIcon status={file.status} />
          <span className="name">{file.name}</span>
          <span className="meta">
            {file.error
              ? "échec"
              : file.status === "done"
                ? `${file.result.pages.length} page${file.result.pages.length > 1 ? "s" : ""} · ${(file.result.confidence * 100).toFixed(0)}%`
                : file.status === "processing"
                  ? `${Math.round((file.progress || 0) * 100)}%`
                  : "en attente"}
          </span>
        </li>
      ))}
    </ul>
  );
}

function StatusIcon({ status }) {
  if (status === "done") return <span className="status-ok">✓</span>;
  if (status === "error") return <span className="status-error">✗</span>;
  if (status === "processing") return <span className="spinner" />;
  return <span className="status-wait">○</span>;
}
