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

const CONCURRENCY = 2;

let nextId = 0;

export default function App() {
  // --- État global ------------------------------------------------------
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

  // --- Thème ------------------------------------------------------------
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("sv-theme", theme);
  }, [theme]);

  // --- Santé du serveur (polling pendant le pré-chargement) -------------
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

  // --- Dérivés ----------------------------------------------------------
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
  const selectedPage = selected?.result?.pages?.[pageIndex] ?? null;

  // --- Actions ----------------------------------------------------------
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

  // --- Rendu ------------------------------------------------------------
  const canExport = Boolean(selected?.result?.text || selected?.editedText);
  const progressPct = files.length
    ? ((stats.done + stats.errors) / files.length) * 100
    : 0;
  const serverNotice =
    server.state === "offline"
      ? "Serveur indisponible — lancez python main.py dans backend/"
      : server.preloading
        ? "Chargement des modèles OCR en cours…"
        : "";

  return (
    <>
      <div className="topbar">
        <span className="brand">ScriptVault OCR</span>
        <select
          aria-label="Langue"
          value={lang}
          onChange={(event) => setLang(event.target.value)}
        >
          {LANGS.map((code) => (
            <option key={code} value={code}>
              {code}
            </option>
          ))}
        </select>
        <label className="muted" title="CLAHE · redressement · binarisation">
          <input
            type="checkbox"
            checked={preprocess}
            onChange={(event) => setPreprocess(event.target.checked)}
          />{" "}
          Prétraitement
        </label>
        <span className="spacer" />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={processing}
        >
          Ajouter des fichiers
        </button>
        <button type="button" className="primary" onClick={startOcr} disabled={processing || !hasPending}>
          Démarrer l'OCR
        </button>
        <button type="button" className="danger" onClick={cancelOcr} disabled={!processing}>
          Annuler
        </button>
        <span className="spacer" />
        <button type="button" onClick={() => handleExport("txt")} disabled={!canExport}>
          Exporter TXT
        </button>
        <button type="button" onClick={() => handleExport("docx")} disabled={!canExport}>
          Exporter DOCX
        </button>
        <button type="button" onClick={() => handleExport("pdf")} disabled={!canExport}>
          Exporter PDF
        </button>
        <button
          type="button"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {theme === "dark" ? "Thème clair" : "Thème sombre"}
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
      </div>

      <div className="split">
        <div className="card left">
          <h3>Document</h3>
          {files.length === 0 ? (
            <DropZone onFiles={addFiles} />
          ) : (
            <>
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
              {selected?.result?.pages?.length > 1 && (
                <div className="toolbar-row muted">
                  Pages :
                  {selected.result.pages.map((page) => (
                    <button
                      key={page.page}
                      type="button"
                      className={`toolbtn${page.page - 1 === pageIndex ? " active" : ""}`}
                      onClick={() => setPageIndex(page.page - 1)}
                    >
                      {page.page}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
          <h3>Fichiers ({files.length})</h3>
          <FileList files={files} selectedId={selectedId} onSelect={selectFile} />
        </div>

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

      <div className="statusbar">
        <span className="status-text">
          {statusText}
          {serverNotice && <span className="notice info" style={{ marginLeft: 8 }}>{serverNotice}</span>}
        </span>
        <span className="mono">Temps : {stats.elapsed ? `${stats.elapsed.toFixed(0)} ms` : "—"}</span>
        <span className="mono">
          {stats.done} / {files.length}
        </span>
        <div className="progress" role="progressbar" aria-valuenow={Math.round(progressPct)}>
          <div className="bar" style={{ width: `${progressPct}%` }} />
        </div>
        <ConfidenceGauge
          value={stats.confidence == null ? null : stats.confidence * 100}
        />
      </div>
    </>
  );
}
