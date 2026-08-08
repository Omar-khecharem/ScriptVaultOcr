import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cancelBatch,
  deleteBatch,
  exportBatchExcel,
  getBatchFile,
  getBatchJob,
  getBatchPreview,
  getHealth,
  isSupportedFile,
  LANGS,
  listBatchFiles,
  uploadBatch,
} from "./api/client.js";
import ConfidenceGauge from "./components/ConfidenceGauge.jsx";
import DropZone from "./components/DropZone.jsx";
import FileList, { FileListPager } from "./components/FileList.jsx";
import FormPanel from "./components/FormPanel.jsx";
import ImageCanvas from "./components/ImageCanvas.jsx";
import {
  DownloadIcon,
  MoonIcon,
  SunIcon,
  UploadIcon,
  XIcon,
} from "./components/icons.jsx";

const PAGE_SIZE = 50;
const TERMINAL = new Set(["done", "cancelled", "error"]);

function isTerminal(status) {
  return TERMINAL.has(status);
}

function shortName(name, max = 34) {
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}

function formatElapsed(ms) {
  if (!ms) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("sv-theme") || "dark");
  const [lang, setLang] = useState("en");
  const [preprocess, setPreprocess] = useState(true);

  // --- Lot en cours ------------------------------------------------------- #
  const [job, setJob] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [files, setFiles] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [q, setQ] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [preview, setPreview] = useState(null);
  const [pageIndex, setPageIndex] = useState(0);

  const [statusText, setStatusText] = useState("En attente d'un lot…");
  const [notice, setNotice] = useState(null);
  const [server, setServer] = useState({ state: "checking", preloading: false, lang: "en" });

  const fileInputRef = useRef(null);
  const jobId = job?.id ?? null;

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

  const refreshFiles = useCallback(async () => {
    if (!jobId) return;
    try {
      const data = await listBatchFiles(jobId, { page, pageSize: PAGE_SIZE, q });
      setFiles(data.items);
      setTotal(data.total);
      setJob((prev) => ({ ...(prev || {}), ...data.job }));
    } catch {
      // le prochain tick réessaiera
    }
  }, [jobId, page, q]);

  useEffect(() => {
    if (!job || isTerminal(job.status)) return undefined;
    let stop = false;
    const tick = async () => {
      try {
        const meta = await getBatchJob(job.id);
        if (stop) return;
        setJob(meta);
        await refreshFiles();
      } catch {
        // hors-ligne temporaire
      }
    };
    tick();
    const timer = setInterval(tick, 1500);
    return () => {
      stop = true;
      clearInterval(timer);
    };
  }, [job, refreshFiles]);

  // --- Détail du fichier sélectionné --------------------------------------- #
  useEffect(() => {
    if (!jobId || !selectedId) {
      setDetail(null);
      setPreview(null);
      return undefined;
    }
    let stop = false;
    setDetail(null);
    setPreview(null);
    setPageIndex(0);
    getBatchFile(jobId, selectedId)
      .then((payload) => {
        if (!stop) setDetail(payload);
      })
      .catch(() => {});
    return () => {
      stop = true;
    };
  }, [jobId, selectedId]);

  // --- Aperçu de la page courante ------------------------------------------- #
  useEffect(() => {
    if (!jobId || !selectedId || !detail || detail.status !== "done") {
      setPreview(null);
      return undefined;
    }
    const pageNumber = pageIndex + 1;
    let stop = false;
    getBatchPreview(jobId, selectedId, pageNumber)
      .then((payload) => {
        if (!stop) setPreview({ page: pageNumber, url: payload.preview });
      })
      .catch(() => {});
    return () => {
      stop = true;
    };
  }, [jobId, selectedId, detail, pageIndex]);

  // --- Sélection dérivée ----------------------------------------------------- #
  const selected = useMemo(
    () => files.find((file) => file.id === selectedId) || null,
    [files, selectedId]
  );
  const pageCount = detail?.pages?.length ?? 0;
  const safePageIndex = pageCount > 0 ? Math.min(pageIndex, pageCount - 1) : 0;
  const selectedPage = detail?.pages?.[safePageIndex] ?? null;
  const form = selectedPage?.form ?? null;

  const stats = useMemo(() => {
    if (!job) return null;
    return job.counts || { total: 0, done: 0, error: 0, processing: 0, pending: 0 };
  }, [job]);

  const progressPct = stats
    ? ((stats.done + stats.error) / Math.max(1, stats.total)) * 100
    : 0;

  // --- Actions ---------------------------------------------------------------- #
  const handleAddFiles = useCallback(
    async (fileList) => {
      const accepted = Array.from(fileList).filter((file) => isSupportedFile(file.name));
      if (accepted.length === 0) {
        setNotice({
          kind: "error",
          text: "Aucun fichier accepté (PNG · JPG · TIFF · WebP · PDF).",
        });
        return;
      }
      if (job && !isTerminal(job.status)) {
        setNotice({
          kind: "warn",
          text: "Un lot est en cours : attendez la fin ou annulez avant d'en créer un nouveau.",
        });
        return;
      }
      if (server.state === "offline") {
        setNotice({
          kind: "error",
          text: "Serveur indisponible — lancez python main.py dans backend/",
        });
        return;
      }
      setUploading(true);
      setUploadProgress(0);
      setNotice(null);
      try {
        const name = `Lot du ${new Date().toLocaleTimeString("fr-FR", {
          hour: "2-digit",
          minute: "2-digit",
        })} (${accepted.length})`;
        const payload = await uploadBatch(
          accepted,
          { name, lang, preprocess },
          setUploadProgress
        );
        const meta = payload.job;
        setJob(meta);
        setPage(1);
        setQ("");
        setFiles([]);
        setTotal(0);
        setSelectedId(null);
        setDetail(null);
        setPreview(null);
        setStatusText(
          `Lot créé — ${meta.counts.total} fichier(s) analysé(s) en arrière-plan.`
        );
        if (payload.rejected?.length) {
          setNotice({
            kind: "warn",
            text: `${payload.rejected.length} fichier(s) rejeté(s) (format ou taille).`,
          });
        }
      } catch (error) {
        setNotice({ kind: "error", text: `Envoi échoué : ${error.message}` });
      } finally {
        setUploading(false);
      }
    },
    [job, lang, preprocess, server.state]
  );

  const handleCancel = useCallback(async () => {
    if (!job) return;
    try {
      const meta = await cancelBatch(job.id);
      setJob(meta);
      setStatusText("Lot annulé.");
    } catch (error) {
      setNotice({ kind: "error", text: `Annulation échouée : ${error.message}` });
    }
  }, [job]);

  const handleDelete = useCallback(async () => {
    if (!job) return;
    if (!window.confirm("Supprimer ce lot et ses fichiers sur le serveur ?")) return;
    try {
      await deleteBatch(job.id);
      setJob(null);
      setFiles([]);
      setTotal(0);
      setSelectedId(null);
      setDetail(null);
      setPreview(null);
      setStatusText("Lot supprimé.");
    } catch (error) {
      setNotice({ kind: "error", text: `Suppression échouée : ${error.message}` });
    }
  }, [job]);

  const handleExportExcel = useCallback(async () => {
    if (!job) return;
    try {
      await exportBatchExcel(job.id);
      setStatusText("Export Excel téléchargé.");
    } catch (error) {
      setNotice({ kind: "error", text: `Export échoué : ${error.message}` });
    }
  }, [job]);

  const selectFile = useCallback((id) => {
    setSelectedId(id);
  }, []);

  const hasJob = Boolean(job);
  const jobTerminal = job ? isTerminal(job.status) : true;
  const canExportExcel = hasJob && (stats?.done ?? 0) > 0;

  const statusDot =
    uploading
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
            className="btn btn-primary"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            title="Importer des images ou des PDF (nouveau lot)"
          >
            <UploadIcon />
            <span>Importer</span>
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleExportExcel}
            disabled={!canExportExcel || uploading}
            title="Exporter les données extraites en Excel"
          >
            <DownloadIcon />
            <span>Export Excel</span>
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleCancel}
            disabled={!hasJob || jobTerminal}
            title="Annuler le traitement du lot"
          >
            <XIcon size={14} />
            <span>Annuler</span>
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={handleDelete}
            disabled={!hasJob || !jobTerminal || uploading}
            title="Supprimer le lot et ses fichiers"
          >
            <XIcon size={14} />
            <span>Supprimer</span>
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
          <label className="switch" title="CLAHE · redressement · binarisation">
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
            handleAddFiles(event.target.files);
            event.target.value = "";
          }}
        />
      </header>

      <main className="app-main">
        <div className="split split-3">
          {/* ----- Colonne gauche : documents ---- */}
          <section className="card card-left">
            <div className="card-head">
              <h2 className="card-title">Documents</h2>
              {job && <span className="card-count">{job.name}</span>}
              <span className="spacer" />
              {job && (
                <span
                  className={`chip chip-${
                    job.status === "done"
                      ? "ok"
                      : job.status === "error"
                        ? "err"
                        : "run"
                  }`}
                >
                  {job.status === "done"
                    ? "Terminé"
                    : job.status === "cancelled"
                      ? "Annulé"
                      : "En cours"}
                </span>
              )}
            </div>
            <div className="card-body">
              {!hasJob ? (
                <DropZone onFiles={handleAddFiles} />
              ) : (
                <>
                  <input
                    className="file-search"
                    type="search"
                    placeholder="Rechercher (nom, statut)…"
                    value={q}
                    onChange={(event) => {
                      setQ(event.target.value);
                      setPage(1);
                    }}
                  />
                  <div className="file-scroll">
                    <FileList files={files} selectedId={selectedId} onSelect={selectFile} />
                  </div>
                  {total > 0 && (
                    <FileListPager
                      page={page}
                      pageSize={PAGE_SIZE}
                      total={total}
                      onChange={setPage}
                    />
                  )}
                  <div className="batch-hint">
                    {stats
                      ? `${stats.done}/${stats.total} traités · ${stats.pending} en file · ${stats.processing} en cours · ${stats.error} en échec`
                      : ""}
                    {uploading && ` · envoi ${Math.round(uploadProgress * 100)}%`}
                  </div>
                </>
              )}
            </div>
          </section>

          {/* ----- Colonne centrale : image ---- */}
          <section className="card card-mid">
            <div className="card-head">
              <h2 className="card-title">Aperçu</h2>
              {selected && <span className="card-count">{shortName(selected.name)}</span>}
              {selected?.status === "done" && (
                <span className="chip chip-ok">
                  {((selected.confidence || 0) * 100).toFixed(0)}%
                </span>
              )}
              {detail?.pages?.length > 1 && (
                <>
                  <span className="spacer" />
                  <div className="pages" role="tablist" aria-label="Sélection de page">
                    <span className="page-label">Pages</span>
                    {detail.pages.map((pageItem) => (
                      <button
                        key={pageItem.page}
                        type="button"
                        role="tab"
                        aria-selected={pageItem.page - 1 === safePageIndex}
                        className={`page-btn${
                          pageItem.page - 1 === safePageIndex ? " active" : ""
                        }`}
                        onClick={() => setPageIndex(pageItem.page - 1)}
                      >
                        {pageItem.page}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
            <div className="card-body">
              <ImageCanvas
                previewUrl={preview?.url || null}
                boxes={
                  selectedPage
                    ? selectedPage.items.map((item) => ({
                        box: item.box,
                        text: item.text,
                        confidence: item.confidence,
                      }))
                    : []
                }
                message={
                  uploading
                    ? "Envoi du lot en cours…"
                    : detail && detail.status !== "done"
                      ? "Analyse en cours ou fichier non traité…"
                      : "Importer des documents pour démarrer"
                }
              />
              {uploading && (
                <div className="upload-banner">
                  <span className="upload-label">
                    Envoi du lot… {Math.round(uploadProgress * 100)}%
                  </span>
                  <div className="progress">
                    <div className="bar" style={{ width: `${uploadProgress * 100}%` }} />
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* ----- Colonne droite : formulaire (sans éditeur de texte) ---- */}
          <FormPanel form={form} />
        </div>
      </main>

      <footer className="statusbar">
        <span className="status-dot" data-state={statusDot} />
        <span className="status-text">
          {statusText}
          {serverNotice && <span className="notice info">{serverNotice}</span>}
          {notice && (
            <span className={`notice ${notice.kind === "error" ? "error" : "info"}`}>
              {notice.text}
            </span>
          )}
        </span>
        <span className="spacer" />
        <div className="stat-chip">
          Fichiers{" "}
          <span className="stat-value">
            {stats?.done ?? 0}/{stats?.total ?? 0}
          </span>
        </div>
        <div className="stat-chip">
          Temps <span className="stat-value mono">{formatElapsed(job?.elapsed_ms)}</span>
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
          value={job?.avg_confidence == null ? null : job.avg_confidence * 100}
        />
      </footer>
    </>
  );
}