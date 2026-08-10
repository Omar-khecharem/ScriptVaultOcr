// ============================================================================
// ScriptVault OCR — interface fixe style Windows 11 / Fluent Design
// Titlebar · CommandBar · 3 panneaux (Documents · Aperçu · Formulaire) ·
// ImportDialog · raccourcis clavier · StatusBar avec progression & confiance.
// ============================================================================

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
import ImportDialog from "./components/ImportDialog.jsx";
import {
  DownloadIcon,
  MinusIcon,
  MoonIcon,
  SquareIcon,
  SunIcon,
  UploadIcon,
  XIcon,
} from "./components/icons.jsx";

const PAGE_SIZES = [25, 50, 100, 200, 500];
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
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes} min ${seconds} s`;
}

export default function App() {
  // --- Préférences ---------------------------------------------------------- #
  const [theme, setTheme] = useState(() => localStorage.getItem("sv-theme") || "dark");
  const [lang, setLang] = useState("en");
  const [preprocess, setPreprocess] = useState(true);
  const [pageSize, setPageSize] = useState(50);

  // --- Lot actif ------------------------------------------------------------ #
  const [job, setJob] = useState(null);
  const [files, setFiles] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [q, setQ] = useState("");
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailTick, setDetailTick] = useState(0);
  const [preview, setPreview] = useState(null);
  const [pageIndex, setPageIndex] = useState(0);

  // --- Envoi / dialog --------------------------------------------------------- #
  const [importOpen, setImportOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // --- Système ---------------------------------------------------------------- #
  const [statusText, setStatusText] = useState("Prêt — créez un lot pour débuter l'OCR.");
  const [notice, setNotice] = useState(null);
  const [server, setServer] = useState({ state: "checking", preloading: false, lang: "en" });

  const searchRef = useRef(null);
  const jobId = job?.id ?? null;

  // --- Thème ------------------------------------------------------------------ #
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("sv-theme", theme);
  }, [theme]);

  // --- Santé du serveur ---------------------------------------------------------- #
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
        if (health.preloading) setTimeout(poll, 4000);
        else if (health.engines && Object.keys(health.engines).length === 0) setTimeout(poll, 2000);
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

  // --- Actualisation du lot actif ------------------------------------------------- #
  const refreshFiles = useCallback(async () => {
    if (!jobId) return;
    try {
      const data = await listBatchFiles(jobId, { page, pageSize, q });
      setFiles(data.items);
      setTotal(data.total);
      setJob((prev) => ({ ...(prev || {}), ...data.job }));
    } catch {
      // le prochain tick réessaiera
    }
  }, [jobId, page, pageSize, q]);

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

  // --- Détail du fichier sélectionné ------------------------------------------------ #
  const refreshDetail = useCallback(() => {
    setDetailTick((tick) => tick + 1);
  }, []);

  // Changement de fichier : efface l'état et reclasse à la page 1.
  useEffect(() => {
    if (!jobId || !selectedId) {
      setDetail(null);
      setPreview(null);
      return undefined;
    }
    setDetail(null);
    setPreview(null);
    setPageIndex(0);
  }, [jobId, selectedId]);

  // Chargement silencieux du détail : au changement de fichier ET à chaque
  // rafraîchissement demandé (fin de traitement, correction enregistrée).
  useEffect(() => {
    if (!jobId || !selectedId) return undefined;
    let stop = false;
    getBatchFile(jobId, selectedId)
      .then((payload) => {
        if (!stop) setDetail(payload);
      })
      .catch(() => {});
    return () => {
      stop = true;
    };
  }, [jobId, selectedId, detailTick]);

  // Fichier sélectionné pendant son traitement : dès qu'il est terminé, le
  // détail (pages + formulaire) est rechargé — cas image unique, où il n'y a
  // pas de bascule pour déclencher le rafraîchissement.
  useEffect(() => {
    if (!jobId || !selectedId) return;
    const fileStatus = files.find((file) => file.id === selectedId)?.status;
    const detailStatus = detail?.status;
    if (fileStatus && detailStatus && detailStatus !== fileStatus) {
      setDetailTick((tick) => tick + 1);
    }
  }, [jobId, selectedId, files, detail?.status]);

  // --- Aperçu de la page courante (image du fichier sélectionné) -------------------- #
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

  // --- Dérivés ------------------------------------------------------------------------- #
  const selected = useMemo(
    () => files.find((file) => file.id === selectedId) || null,
    [files, selectedId]
  );
  const pageCount = detail?.pages?.length ?? 0;
  const safePageIndex = pageCount > 0 ? Math.min(pageIndex, pageCount - 1) : 0;
  const selectedPage = detail?.pages?.[safePageIndex] ?? null;
  const form = selectedPage?.form ?? null;
  const overrides = detail?.overrides?.[String(safePageIndex + 1)] ?? null;

  const stats = useMemo(() => {
    if (!job) return null;
    return job.counts || { total: 0, done: 0, error: 0, processing: 0, pending: 0 };
  }, [job]);

  const progressPct = stats
    ? ((stats.done + stats.error) / Math.max(1, stats.total)) * 100
    : 0;

  const jobTerminal = job ? isTerminal(job.status) : true;
  const canExportExcel = (stats?.done ?? 0) > 0;

  // --- Actions -------------------------------------------------------------------------- #
  const resetWorkspace = useCallback(() => {
    setFiles([]);
    setTotal(0);
    setPage(1);
    setQ("");
    setSelectedId(null);
    setDetail(null);
    setPreview(null);
  }, []);

  const handleCreate = useCallback(
    async (acceptedFiles, name) => {
      if (acceptedFiles.length === 0) return;
      if (job && !jobTerminal) {
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
        const payload = await uploadBatch(
          acceptedFiles,
          { name, lang, preprocess },
          setUploadProgress
        );
        const meta = payload.job;
        resetWorkspace();
        setJob(meta);
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
        setImportOpen(false);
      }
    },
    [job, jobTerminal, lang, preprocess, server.state, resetWorkspace]
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
    if (!window.confirm(`Supprimer le lot « ${job.name || "sans nom"} » et ses fichiers sur le serveur ?`)) {
      return;
    }
    try {
      await deleteBatch(job.id);
      setJob(null);
      resetWorkspace();
      setStatusText("Lot supprimé.");
    } catch (error) {
      setNotice({ kind: "error", text: `Suppression échouée : ${error.message}` });
    }
  }, [job, resetWorkspace]);

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

  // --- Raccourcis clavier ---------------------------------------------------------------- #
  useEffect(() => {
    function onKeyDown(event) {
      const mod = event.ctrlKey || event.metaKey;
      if (mod && event.key.toLowerCase() === "o") {
        event.preventDefault();
        setImportOpen(true);
      } else if (mod && event.key.toLowerCase() === "e") {
        event.preventDefault();
        handleExportExcel();
      } else if (mod && event.key.toLowerCase() === "l") {
        event.preventDefault();
        if (jobId) searchRef.current?.focus();
      } else if (event.key === "Escape" && importOpen) {
        event.preventDefault();
        setImportOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [importOpen, jobId, handleExportExcel]);

  // --- Barre d'état ------------------------------------------------------------------------- #
  const statusDot = uploading
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
      {/* ------------------------------- Titlebar ------------------------------- */}
      <header className="titlebar">
        <div className="brand">
          <span className="brand-mark">SV</span>
          <span className="brand-name">ScriptVault</span>
          <span className="brand-tag">OCR</span>
        </div>
        {job && (
          <>
            <span className="tb-divider" />
            <span className={`chip chip-${
              job.status === "done"
                ? "ok"
                : job.status === "error"
                  ? "err"
                  : job.status === "cancelled"
                    ? "wait"
                    : "run"
            }`} title={job.name}>
              {shortName(job.name, 42)}
            </span>
          </>
        )}
        <span className="spacer" />
        <button
          type="button"
          className="btn btn-icon titlebar-action"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Passer au thème clair" : "Passer au thème sombre"}
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
        <span className="tb-divider" />
        <div className="window-controls" aria-hidden="true">
          <button type="button" className="wc-btn" tabIndex={-1} title="Réduire">
            <MinusIcon size={14} />
          </button>
          <button type="button" className="wc-btn" tabIndex={-1} title="Agrandir">
            <SquareIcon size={12} />
          </button>
          <button type="button" className="wc-btn wc-close" tabIndex={-1} title="Fermer">
            <XIcon size={14} />
          </button>
        </div>
      </header>

      {/* ------------------------------- CommandBar --------------------------------- */}
      <div className="commandbar">
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => setImportOpen(true)}
          disabled={uploading}
          title="Importer des images ou des PDF (Ctrl+O)"
        >
          <UploadIcon size={14} />
          <span>Importer</span>
        </button>
        <span className="tb-divider" />
        <button
          type="button"
          className="btn"
          onClick={handleExportExcel}
          disabled={!canExportExcel || uploading}
          title="Exporter les données extraites en Excel (Ctrl+E)"
        >
          <DownloadIcon size={14} />
          <span>Export Excel</span>
        </button>
        <button
          type="button"
          className="btn"
          onClick={handleCancel}
          disabled={!job || jobTerminal}
          title="Annuler le traitement du lot"
        >
          <XIcon size={14} />
          <span>Annuler</span>
        </button>
        <button
          type="button"
          className="btn btn-danger"
          onClick={handleDelete}
          disabled={!job || !jobTerminal || uploading}
          title="Supprimer le lot et ses fichiers"
        >
          <XIcon size={14} />
          <span>Supprimer</span>
        </button>
        <span className="spacer" />
        <label className="field field-inline" title="Langue du texte à reconnaître">
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

      {/* ------------------------ 3 panneaux : Documents · Aperçu · Formulaire --------- */}
      <main className="app-main">
        <div className="split split-3">
          {/* -------- Panneau gauche : documents -------- */}
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
                        : job.status === "cancelled"
                          ? "wait"
                          : "run"
                  }`}
                >
                  {job.status === "done"
                    ? "Terminé"
                    : job.status === "cancelled"
                      ? "Annulé"
                      : job.status === "error"
                        ? "Échec"
                        : "En cours"}
                </span>
              )}
            </div>
            <div className="card-body">
              {!job ? (
                <DropZone
                  onFiles={(list) => {
                    const accepted = Array.from(list).filter((file) =>
                      isSupportedFile(file.name)
                    );
                    if (accepted.length) handleCreate(accepted, null);
                  }}
                />
              ) : (
                <>
                  <div className="file-toolbar">
                    <span className="search-box">
                      <input
                        ref={searchRef}
                        className="file-search"
                        type="search"
                        placeholder="Rechercher (nom, statut)…"
                        value={q}
                        onChange={(event) => {
                          setQ(event.target.value);
                          setPage(1);
                        }}
                      />
                    </span>
                    <label className="field field-inline" title="Taille de page">
                      <select
                        value={pageSize}
                        onChange={(event) => {
                          setPageSize(Number(event.target.value));
                          setPage(1);
                        }}
                        aria-label="Fichiers par page"
                      >
                        {PAGE_SIZES.map((size) => (
                          <option key={size} value={size}>
                            {size}/page
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="file-scroll">
                    <FileList files={files} selectedId={selectedId} onSelect={selectFile} />
                  </div>
                  {total > 0 && (
                    <FileListPager
                      page={page}
                      pageSize={pageSize}
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

          {/* -------- Panneau central : aperçu (image du fichier sélectionné) -------- */}
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

          {/* -------- Panneau droit : formulaire éditable -------- */}
          <FormPanel
            form={form}
            jobId={jobId}
            fileId={selectedId}
            pageNumber={pageIndex + 1}
            overrides={overrides}
            onSaved={refreshDetail}
          />
        </div>
      </main>

      {/* ------------------------------ StatusBar ---------------------------------- */}
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
        {job && (
          <>
            <div className="stat-chip mono">
              {stats?.done ?? 0}/{stats?.total ?? 0} fichiers
            </div>
            <div className="stat-chip mono">Temps {formatElapsed(job.elapsed_ms)}</div>
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
          </>
        )}
      </footer>

      {/* ------------------------------ ImportDialog ---------------------------------- */}
      <ImportDialog
        open={importOpen}
        lang={lang}
        preprocess={preprocess}
        uploading={uploading}
        progress={uploadProgress}
        onLangChange={setLang}
        onPreprocessChange={setPreprocess}
        onCreate={handleCreate}
        onClose={() => {
          if (!uploading) setImportOpen(false);
        }}
      />
    </>
  );
}