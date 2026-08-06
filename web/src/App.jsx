import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  abortOcr,
  exportDocument,
  getHealth,
  isSupportedFile,
  LANGS,
  ocrSingle,
} from "./api/client.js";
import ConfidenceGauge from "./components/ConfidenceGauge.jsx";
import DropZone from "./components/DropZone.jsx";
import EditorPanel from "./components/EditorPanel.jsx";
import FileList from "./components/FileList.jsx";
import ImageCanvas from "./components/ImageCanvas.jsx";
import {
  ChevronDownIcon,
  DownloadIcon,
  FileTextIcon,
  MoonIcon,
  PlayIcon,
  SunIcon,
  UploadIcon,
  XIcon,
} from "./components/icons.jsx";

const CONCURRENCY = 2;

let nextId = 0;

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("sv-theme") || "dark");
  const [files, setFiles] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [pageIndex, setPageIndex] = useState(0);
  const [lang, setLang] = useState("en");
  const [preprocess, setPreprocess] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [statusText, setStatusText] = useState("En attente de fichiers…");
  const [server, setServer] = useState({ state: "checking", preloading: false, lang: "en" });

  const filesRef = useRef(files);
  useEffect(() => {
    filesRef.current = files;
  }, [files]);
  const cancelledRef = useRef(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("sv-theme", theme);
  }, [theme]);

  useEffect(() => {
    let stop = false;
    async function poll() {
      try {
        const health = await getHealth();
        if (stop) return;
        setServer({
          state: "ok",
          preloading: Boolean(health.preloading),
          lang: health.lang,
        });
        if (health.preloading) {
          setTimeout(poll, 4000);
        } else if (health.engines && Object.keys(health.engines).length === 0) {
          setTimeout(poll, 2000);
        }
      } catch {
        if (!stop) {
          setServer({ state: "offline", preloading: false, lang: "" });
          setTimeout(poll, 5000);
        }
      }
    }
    poll();
    return () => {
      stop = true;
    };
  }, []);

  const selected = useMemo(
    () => files.find((file) => file.id === selectedId) || null,
    [files, selectedId]
  );

  const stats = useMemo(() => {
    const done = files.filter((f) => f.status === "done");
    const errors = files.filter((f) => f.status === "error");
    const elapsed = done.reduce((sum, f) => sum + (f.result?.elapsed_ms || 0), 0);
    const confidence =
      done.length > 0
        ? done.reduce((sum, f) => sum + (f.result?.confidence || 0), 0) / done.length
        : null;
    return { done: done.length, errors: errors.length, elapsed, confidence };
  }, [files]);

const hasPending = files.some((f) => f.status === "pending");
  const editorText = selected?.editedText ?? selected?.result?.text ?? "";
  const pageCount = selected?.result?.pages?.length ?? 0;
  const safePageIndex = pageCount > 0 ? Math.min(pageIndex, pageCount - 1) : 0;
  const selectedPage = selected?.result?.pages?.[safePageIndex] ?? null;
  const firstError = files.find((f) => f.status === "error")?.error ?? null;

  const addFiles = useCallback((fileList) => {
    const accepted = fileList.filter((file) => isSupportedFile(file.name));
    const rejected = fileList.length - accepted.length;
    if (accepted.length === 0) {
      setStatusText("Aucun fichier accepté (PNG · JPG · TIFF · WebP · PDF).");
      return;
    }
    const records = accepted.map((file) => ({
      id: ++nextId,
      file,
      name: file.name,
      status: "pending",
      progress: 0,
      result: null,
      error: null,
      editedText: null,
    }));
    setFiles((prev) => [...prev, ...records]);
    setSelectedId(records[0].id);
    setPageIndex(0);
    setStatusText(
      `${records.length} fichier(s) ajouté(s)${rejected ? `, ${rejected} ignoré(s)` : ""} — cliquez sur Démarrer l'OCR`
    );
  }, []);

  const updateFile = useCallback((id, patch) => {
    setFiles((prev) => prev.map((file) => (file.id === id ? { ...file, ...patch } : file)));
  }, []);

  const startOcr = useCallback(async () => {
    if (processing) return;
    const queue = filesRef.current
      .filter((file) => file.status === "pending")
      .map((file) => file.id);
    if (queue.length === 0) {
      setStatusText("Ajoutez des fichiers avant de démarrer.");
      return;
    }
    cancelledRef.current = false;
    setProcessing(true);
    setStatusText("Analyse en cours…");

    let next = 0;
    const worker = async () => {
      while (next < queue.length) {
        if (cancelledRef.current) break;
        const id = queue[next++];
        const record = filesRef.current.find((file) => file.id === id);
        if (!record) continue;
        updateFile(id, { status: "processing", progress: 0, error: null });
        try {
          const result = await ocrSingle(
            record.file,
            { lang, preprocess },
            (ratio) => updateFile(id, { progress: ratio })
          );
          if (cancelledRef.current) {
            updateFile(id, { status: "pending" });
            break;
          }
          updateFile(id, { status: "done", result, progress: 1 });
        } catch (error) {
          if (cancelledRef.current) {
            updateFile(id, { status: "pending" });
            break;
          }
          updateFile(id, { status: "error", error: error.message });
        }
      }
    };

    try {
      await Promise.all([worker(), worker()]);
    } finally {
      setProcessing(false);
      if (cancelledRef.current) {
        setStatusText("Traitement annulé.");
      } else {
        const current = filesRef.current;
        const done = current.filter((f) => f.status === "done").length;
        const failed = current.filter((f) => f.status === "error").length;
        setStatusText(
          `Terminé — ${done} fichier(s) analysé(s)${failed ? `, ${failed} en erreur` : ""}.`
        );
      }
    }
  }, [processing, lang, preprocess, updateFile]);

  const cancelOcr = useCallback(() => {
    cancelledRef.current = true;
    abortOcr();
    setStatusText("Annulation demandée…");
  }, []);

  const handleExport = useCallback(
    async (format) => {
      const text = editorText.trim();
      if (!text) {
        setStatusText("L'éditeur est vide : rien à exporter.");
        return;
      }
      try {
        await exportDocument(format, text, selected?.name);
        setStatusText(`Export ${format.toUpperCase()} téléchargé.`);
      } catch (error) {
        setStatusText(`Export échoué : ${error.message}`);
      }
    },
    [editorText, selected]
  );

  const handleEditorChange = useCallback(
    (text) => {
      if (selectedId) updateFile(selectedId, { editedText: text });
    },
    [selectedId, updateFile]
  );

  const selectFile = useCallback((id) => {
    setSelectedId(id);
    setPageIndex(0);
  }, []);

  const canExport = Boolean(selected?.result?.text || selected?.editedText);
  const progressPct = files.length
    ? ((stats.done + stats.errors) / files.length) * 100
    : 0;
  const elapsedLabel = stats.elapsed
    ? stats.elapsed < 1000
      ? `${Math.round(stats.elapsed)} ms`
      : `${(stats.elapsed / 1000).toFixed(1)} s`
    : "—";
  const statusDot =
    processing
      ? "working"
      : server.state === "offline"
        ? "error"
        : server.state === "ok"
          ? "ok"
          : "wait";
  const serverNotice =
    server.state === "offline"
      ? "Serveur indisponible — lancez python main.py dans backend/"
      : server.preloading
        ? "Chargement des modèles OCR en cours…"
        : "";

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">SV</span>
          <span className="brand-name">ScriptVault</span>
          <span className="brand-tag">OCR</span>
        </div>
        <span className="tb-divider" />
        <div className="tb-group">
          <button
            type="button"
            className="btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={processing}
            title="Ajouter des images ou des PDF"
          >
            <UploadIcon />
            <span>Ajouter</span>
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={startOcr}
            disabled={processing || !hasPending}
            title="Lancer la reconnaissance du texte"
          >
            <PlayIcon size={14} />
            <span>Démarrer l'OCR</span>
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={cancelOcr}
            disabled={!processing}
            title="Annuler le traitement en cours"
          >
            <XIcon size={14} />
            <span>Annuler</span>
          </button>
        </div>
        <span className="spacer" />
        <div className="tb-group">
          <label className="field" title="Langue du texte à reconnaître">
            <span className="field-label">Langue</span>
            <select value={lang} onChange={(event) => setLang(event.target.value)}>
              {LANGS.map((code) => (
                <option key={code} value={code}>
                  {code}
                </option>
              ))}
            </select>
          </label>
          <label
            className="switch"
            title="CLAHE · redressement · binarisation"
          >
            <input
              type="checkbox"
              checked={preprocess}
              onChange={(event) => setPreprocess(event.target.checked)}
            />
            <span className="switch-track" />
            <span className="switch-text">Prétraitement</span>
          </label>
        </div>
        <span className="tb-divider" />
        <ExportMenu canExport={canExport} onExport={handleExport} />
        <button
          type="button"
          className="btn btn-icon"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Passer au thème clair" : "Passer au thème sombre"}
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".png,.jpg,.jpeg,.tif,.tiff,.webp,.bmp,.pdf"
          style={{ display: "none" }}
          onChange={(event) => {
            addFiles(Array.from(event.target.files || []));
            event.target.value = "";
          }}
        />
      </header>

      <main className="app-main">
        <div className="split">
          <section className="card card-left">
            <div className="card-head">
              <h2 className="card-title">Document</h2>
              {selected && <span className="card-count">{shortName(selected.name)}</span>}
              <span className="spacer" />
              {selected?.result?.pages?.length > 1 && (
                <div className="pages" role="tablist" aria-label="Sélection de page">
                  <span className="page-label">Pages</span>
                  {selected.result.pages.map((page) => (
                    <button
                      key={page.page}
                      type="button"
                      role="tab"
                      aria-selected={page.page - 1 === safePageIndex}
                      className={`page-btn${page.page - 1 === safePageIndex ? " active" : ""}`}
                      onClick={() => setPageIndex(page.page - 1)}
                    >
                      {page.page}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="card-body">
              {files.length === 0 ? (
                <DropZone onFiles={addFiles} />
              ) : (
                <ImageCanvas
                  previewUrl={selectedPage?.preview || null}
                  boxes={
                    selectedPage
                      ? selectedPage.items.map((item) => ({
                          box: item.box,
                          text: item.text,
                          confidence: item.confidence,
                        }))
                      : []
                  }
                  message="Analyse en cours ou fichier non traité…"
                />
              )}
            </div>
            <div className="card-foot">
              <div className="card-foot-head">
                <h2 className="card-title">Fichiers</h2>
                {files.length > 0 && <span className="card-count">{files.length}</span>}
              </div>
              <FileList files={files} selectedId={selectedId} onSelect={selectFile} />
            </div>
          </section>

          <EditorPanel
            fileKey={selectedId ?? "empty"}
            text={editorText}
            placeholder={
              selected?.result?.text == null && selected?.editedText == null
                ? "Ce fichier n'a pas encore été analysé."
                : undefined
            }
            onChange={handleEditorChange}
          />
        </div>
      </main>

      <footer className="statusbar">
        <span className="status-dot" data-state={statusDot} />
        <span className="status-text">
          {statusText}
          {serverNotice && (
            <span className="notice info">{serverNotice}</span>
          )}
          {firstError && !processing && (
            <span className="notice error" title={firstError}>
              {firstError}
            </span>
          )}
        </span>
        <span className="spacer" />
        <div className="stat-chip">
          Fichiers <span className="stat-value">{stats.done}/{files.length}</span>
        </div>
        <div className="stat-chip">
          Temps <span className="stat-value mono">{elapsedLabel}</span>
        </div>
        <div
          className="progress"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(progressPct)}
        >
          <div className="bar" style={{ width: `${progressPct}%` }} />
        </div>
        <ConfidenceGauge
          value={stats.confidence == null ? null : stats.confidence * 100}
        />
      </footer>
    </>
  );
}

function ExportMenu({ canExport, onExport }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    function handlePointer(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) setOpen(false);
    }
    function handleKey(event) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", handlePointer);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handlePointer);
      document.removeEventListener("keydown", handleKey);
    };
  }, [open]);

  const formats = [
    { id: "txt", label: "TXT — texte brut" },
    { id: "docx", label: "DOCX — Word" },
    { id: "pdf", label: "PDF — document" },
  ];

  return (
    <div className="export-menu" ref={menuRef}>
      <button
        type="button"
        className="btn"
        onClick={() => setOpen((value) => !value)}
        disabled={!canExport}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Exporter le texte corrigé"
      >
        <DownloadIcon />
        <span>Exporter</span>
        <ChevronDownIcon size={14} />
      </button>
      {open && (
        <div className="export-pop" role="menu">
          <div className="pop-title">Formats d'export</div>
          {formats.map((format) => (
            <button
              key={format.id}
              type="button"
              role="menuitem"
              className="export-item"
              onClick={() => {
                setOpen(false);
                onExport(format.id);
              }}
            >
              <FileTextIcon />
              {format.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function shortName(name, max = 30) {
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}
