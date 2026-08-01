// ============================================================================
// Client API — ScriptVault OCR backend (FastAPI)
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
    return payload.detail || payload.error || `Erreur ${response.status}`;
  } catch {
    return `Erreur ${response.status}`;
  }
}

/** État du serveur et du pool de moteurs. */
export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error(await parseError(response));
  return response.json();
}

let currentRequestRef = null;

/**
 * OCR d'un fichier avec progression d'upload (XHR).
 *
 * @param {File} file
 * @param {{lang: string, preprocess: boolean}} options
 * @param {(ratio: number) => void} onProgress
 * @returns {Promise<object>} réponse OCRResponse du backend
 */
export function ocrSingle(file, { lang, preprocess }, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("file", file);
    form.append("lang", lang);
    form.append("preprocess", String(preprocess));
    form.append("preview", "true");

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/ocr/single`);
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
    currentRequestRef = xhr;
  });
}

/** Annule la requête OCR en cours (bouton Annuler). */
export function abortOcr() {
  if (currentRequestRef) {
    currentRequestRef.abort();
    currentRequestRef = null;
  }
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

  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = /filename="([^"]+)"/.exec(disposition);
  anchor.download = match ? match[1] : `scriptvault-export.${format}`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
