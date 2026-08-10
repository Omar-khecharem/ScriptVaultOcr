// ============================================================================
// Client API — ScriptVault OCR backend (FastAPI)
// Workflow entreprise : upload par lots, traitement en arrière-plan,
// progression interrogée par polling, menus paginés côté client.
// ============================================================================

const API_BASE = "/api";

export const LANGS = [
  "en",
  "fr",
  "ch",
  "japan",
  "korean",
  "de",
  "es",
  "it",
  "pt",
];

const SUPPORTED_EXTENSIONS = new Set([
  "png",
  "jpg",
  "jpeg",
  "tif",
  "tiff",
  "webp",
  "bmp",
  "pdf",
]);

export function isSupportedFile(name) {
  const dot = name.lastIndexOf(".");
  return dot !== -1 && SUPPORTED_EXTENSIONS.has(name.slice(dot + 1).toLowerCase());
}

async function parseError(response) {
  try {
    const payload = await response.json();
    const detail = Array.isArray(payload.detail)
      ? payload.detail.map((entry) => entry.msg).join(" · ")
      : payload.detail;
    return detail || payload.error || `Erreur ${response.status}`;
  } catch {
    return `Erreur ${response.status}`;
  }
}

function downloadBlob(blob, fallbackName) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fallbackName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

/** Habitat du serveur et du pool d'engines. */
export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/**
 * Dépose un lot de fichiers (multipart, progression d'upload via XHR).
 *
 * @param {File[]} files
 * @param {{name: string, lang: string, preprocess: boolean}} options
 * @param {(ratio: number) => void} onProgress
 * @returns {Promise<{job: object, rejected?: Array}>}
 */
export function uploadBatch(files, { name, lang, preprocess }, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("name", name);
    form.append("lang", lang);
    form.append("preprocess", String(preprocess));
    for (const file of files) form.append("files", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/batches`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(event.loaded / event.total);
      }
    };
    xhr.onload = () => {
      let payload = null;
      try {
        payload = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error("Réponse du serveur invalide."));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
      } else {
        reject(new Error(payload.detail || `Erreur ${xhr.status}`));
      }
    };
    xhr.onerror = () => reject(new Error("Impossible de joindre le serveur."));
    xhr.send(form);
  });
}

/** Résumé d'un lot (progression, compteurs, confiance moyenne). */
export async function getBatchJob(jobId) {
  const response = await fetch(`${API_BASE}/batches/${jobId}`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/**
 * Liste des fichiers d'un lot — paginée et filtrable.
 *
 * @param {string} jobId
 * @param {{page: number, pageSize: number, q: string}} params
 */
export async function listBatchFiles(jobId, { page, pageSize, q } = {}) {
  const query = new URLSearchParams();
  query.set("page", String(page || 1));
  query.set("page_size", String(pageSize || 50));
  if (q) query.set("q", q);
  const response = await fetch(`${API_BASE}/batches/${jobId}/files?${query}`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/** Détail complet d'un fichier : pages OCR + formulaire structuré. */
export async function getBatchFile(jobId, fileId) {
  const response = await fetch(`${API_BASE}/batches/${jobId}/files/${fileId}`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/** Aperçu PNG (data URL) d'une page, telle qu'analysée. */
export async function getBatchPreview(jobId, fileId, page) {
  const response = await fetch(
    `${API_BASE}/batches/${jobId}/files/${fileId}/preview?page=${page}`
  );
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/** Annule le traitement d'un lot en cours. */
export async function cancelBatch(jobId) {
  const response = await fetch(`${API_BASE}/batches/${jobId}/cancel`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/**
 * Enregistre les valeurs du formulaire corrigées à la main (export Excel
 * compris). `values` = {clé de champ → valeur} ; une valeur vide efface
 * la correction.
 */
export async function saveFormOverrides(jobId, fileId, page, values) {
  const response = await fetch(
    `${API_BASE}/batches/${jobId}/files/${fileId}/form`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page, values }),
    }
  );
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/** Supprime un lot (mémoire + zone de travail serveur). */
export async function deleteBatch(jobId) {
  const response = await fetch(`${API_BASE}/batches/${jobId}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

/** Export Excel de toutes les données du lot (téléchargement direct). */
export async function exportBatchExcel(jobId) {
  const response = await fetch(`${API_BASE}/batches/${jobId}/export.xlsx`);
  if (!response.ok) throw new Error(await parseError(response));
  const blob = await response.blob();
  downloadBlob(blob, `scriptvault-lot-${jobId.slice(0, 8)}.xlsx`);
}

/**
 * Exporte un texte corrigé (TXT / DOCX / PDF) et déclenche le téléchargement.
 */
export async function exportDocument(format, text, filename) {
  const response = await fetch(`${API_BASE}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format, text, filename }),
  });
  if (!response.ok) throw new Error(await parseError(response));
  const blob = await response.blob();

  const disposition = response.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  downloadBlob(blob, match ? match[1] : `scriptvault-export.${format}`);
}