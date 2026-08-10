// ============================================================================
// FormPanel — gabarit FIXE de la feuille d'examen, éditable
//
// Le gabarit est constant : Concours, Épreuve de, Date, Durée, Nom, Prénom,
// Date & lieu de naissance (un seul champ), Établissement d'origine, CIN ou
// passeport, Série, Identifiant. L'OCR remplit chaque champ ; l'utilisateur
// peut corriger à la main (les valeurs corrigées sont enregistrées côté
// serveur et reprises dans l'export Excel). Les champs non lus portent le
// statut "empty" (pointillés gris, "—") mais restent saisissables.
//   - "error"   -> encadrement rouge vif + fond clair + message d'alerte
//   - "warning" -> encadrement orange
//   - "valid"   -> liseré vert
//   - "empty"   -> pointillés gris, champ non lu
//   - champ corrigé -> boussole "modifié" (badge + saisie)
// ============================================================================

import { useCallback, useEffect, useRef, useState } from "react";
import { saveFormOverrides } from "../api/client.js";

const SECTION_ORDER = ["concours", "candidat", "codification"];

const DEBOUNCE_MS = 700;

export default function FormPanel({ form, jobId, fileId, pageNumber, overrides, onSaved }) {
  const { fields } = form && Array.isArray(form.fields) ? form : { fields: [] };
  const [values, setValues] = useState({});
  const [dirty, setDirty] = useState(false);
  const [saved, setSaved] = useState(false);

  // Reflets synchrones de l'état : indispensables pour ne jamais perdre une
  // correction si l'utilisateur change de fichier/page pendant le débounce.
  const valuesRef = useRef(values);
  const dirtyRef = useRef(false);
  const timerRef = useRef(null);
  const snapshotRef = useRef("");
  const editedKeys = useRef(new Set());
  valuesRef.current = values;

  const persist = useCallback(
    async (vals) => {
      if (!jobId || !fileId) return;
      const payload = {};
      for (const [key, value] of Object.entries(vals)) {
        if (value !== "") payload[key] = value;
      }
      await saveFormOverrides(jobId, fileId, pageNumber, payload);
      setSaved(true);
      onSaved?.();
    },
    [jobId, fileId, pageNumber, onSaved]
  );

  // Sauvegarde immédiate des corrections en attente (fin de débounce ou
  // changement de fichier/page : le timeout ne doit jamais avaler la saisie).
  const flushPending = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (dirtyRef.current && jobId && fileId) {
      const pending = { ...valuesRef.current };
      dirtyRef.current = false;
      persist(pending).catch(() => setSaved(false));
    }
  }, [jobId, fileId, persist]);

  // Réinitialise la saisie quand on change de fichier ou de page.
  const snapshotKey = `${jobId}/${fileId}/${pageNumber}`;
  useEffect(() => {
    if (snapshotRef.current !== snapshotKey) {
      snapshotRef.current = snapshotKey;
      flushPending();
    }
    const next = {};
    for (const field of fields) {
      next[field.key] = field.value ?? "";
    }
    // Les corrections déjà enregistrées côté serveur réapparaissent.
    for (const [key, value] of Object.entries(overrides || {})) {
      if (value) next[key] = value;
    }
    setValues(next);
    setDirty(false);
    dirtyRef.current = false;
    setSaved(false);
    editedKeys.current = new Set(Object.keys(overrides || {}));
  }, [snapshotKey, fields.length]);

  // Sauvegarde différée des corrections (déclenchée sur chaque modification).
  useEffect(() => {
    if (!dirty || !jobId || !fileId) return undefined;
    setSaved(false);
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(() => {
      timerRef.current = null;
      persist(values).catch(() => setSaved(false));
    }, DEBOUNCE_MS);
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [values, dirty, jobId, fileId, pageNumber, persist]);

  const handleChange = useCallback((key, value) => {
    setValues((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
    dirtyRef.current = true;
  }, []);

  const markEdited = useCallback((key) => {
    editedKeys.current.add(key);
  }, []);

  if (!form || !Array.isArray(form.fields)) {
    return (
      <section className="card card-right">
        <div className="card-head">
          <h2 className="card-title">Formulaire d'examen</h2>
        </div>
        <div className="card-body form-empty">Aucune analyse disponible.</div>
      </section>
    );
  }

  const { global_confidence: confidence, processing_time_ms: elapsed } = form;
  const sections = groupBySection(fields);
  const errors = fields.filter((field) => field.status === "error").length;
  const warnings = fields.filter((field) => field.status === "warning").length;
  const detected = fields.filter((field) => field.status !== "empty").length;
  const allEmpty = detected === 0;

  return (
    <section className="card card-right">
      <div className="card-head">
        <h2 className="card-title">Formulaire d'examen</h2>
        <span className="card-count">
          {detected} champ{detected === 1 ? "" : "s"} lu{detected === 1 ? "" : "s"}
        </span>
        <span className="spacer" />
        {saved && dirty ? (
          <span className="form-save form-save--ok" role="status">
            Corrigé ✓
          </span>
        ) : dirty ? (
          <span className="form-save" role="status">
            Enregistrement…
          </span>
        ) : null}
      </div>
      <div className="card-body form-scroll">
        <div className="form-summary">
          {allEmpty ? (
            <span className="form-badge form-badge--warning">
              Gabarit — champs non lus par l'OCR
            </span>
          ) : (
            <>
              <span className="form-badge form-badge--error">
                {errors} alerte{errors === 1 ? "" : "s"} rouge
              </span>
              {warnings > 0 && (
                <span className="form-badge form-badge--warning">
                  {warnings} avertissement{warnings === 1 ? "" : "s"}
                </span>
              )}
            </>
          )}
          <span className="spacer" />
          <span className="form-meta">
            Confiance {Math.round((confidence ?? 0) * 100)}% · post-traitement {elapsed ?? 0} ms
          </span>
        </div>
        <p className="form-hint">
          Les valeurs lues par l'OCR sont modifiables : les corrections sont
          reprises dans l'export Excel.
        </p>
        {sections.map(({ name, label, fields: sectionFields }) => (
          <section key={name} className="form-section">
            <h3 className="form-section-title">{label}</h3>
            <div className="form-fields">
              {sectionFields.map((field) => (
                <FormField
                  key={field.key}
                  field={field}
                  value={values[field.key] ?? ""}
                  edited={editedKeys.current.has(field.key)}
                  onChange={(value) => handleChange(field.key, value)}
                  onFocus={() => markEdited(field.key)}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}

/** Regroupe les champs par section (ordre canonique du gabarit). */
function groupBySection(fields) {
  const grouped = new Map();
  for (const field of fields) {
    const name = field.section || "candidat";
    if (!grouped.has(name)) {
      grouped.set(name, {
        name,
        label: field.section_label || name,
        fields: [],
      });
    }
    grouped.get(name).fields.push(field);
  }
  const order = new Map(SECTION_ORDER.map((name, index) => [name, index]));
  return [...grouped.values()].sort(
    (a, b) => (order.get(a.name) ?? 99) - (order.get(b.name) ?? 99)
  );
}

function FormField({ field, value, edited, onChange, onFocus }) {
  const { label, confidence, status, error_message: errorMessage } = field;
  const isEmpty = status === "empty";
  const displayStatus = edited ? "edited" : status;
  const title = edited
    ? `${label} — corrigé manuellement`
    : isEmpty
      ? `${label} — non lu par l'OCR`
      : `${label} — confiance ${Math.round((confidence ?? 0) * 100)}%`;

  return (
    <label className={`form-field form-field--${displayStatus}`} title={title}>
      <span className="form-field-label">
        {label}
        <span
          className={`form-field-dot form-field-dot--${displayStatus}`}
        />
      </span>
      <input
        type="text"
        className={`form-field-input form-field-value--${displayStatus}`}
        value={value ?? ""}
        placeholder={isEmpty ? "—" : ""}
        onChange={(event) => onChange(event.target.value)}
        onFocus={onFocus}
      />
      <span className="form-field-note" data-status={displayStatus}>
        {edited ? (
          <span className="form-edited-badge">✏️ Corrigé à la main</span>
        ) : status === "error" && errorMessage ? (
          `⚠️ ${errorMessage}`
        ) : status === "warning" ? (
          `⚡ ${errorMessage ?? `Confiance modérée (${Math.round((confidence ?? 0) * 100)}%)`}`
        ) : isEmpty ? (
          "Champ non lu par l'OCR — saisissable"
        ) : (
          "✓ Lecture validée"
        )}
      </span>
    </label>
  );
}