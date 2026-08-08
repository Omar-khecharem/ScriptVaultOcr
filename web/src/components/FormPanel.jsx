// ============================================================================
// FormPanel — rendu du gabarit FIXE de la feuille d'examen
//
// Le gabarit est constant : Concours, Épreuve de, Date, Durée, Nom, Prénom,
// Date & lieu de naissance (un seul champ), Établissement d'origine, CIN ou
// passeport, Série, Identifiant. L'OCR ne fait que remplir le contenu de
// chaque champ ; les champs non lus portent le statut "empty" (pointillés
// gris, "—"). Aucune zone de saisie.
//   - "error"   -> encadrement rouge vif + fond clair + message d'alerte
//   - "warning" -> encadrement orange
//   - "valid"   -> liseré vert
//   - "empty"   -> pointillés gris, champ non lu
// ============================================================================

const SECTION_ORDER = ["concours", "candidat", "codification"];

export default function FormPanel({ form }) {
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

  const { fields, global_confidence: confidence, processing_time_ms: elapsed } = form;
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
        {sections.map(({ name, label, fields: sectionFields }) => (
          <section key={name} className="form-section">
            <h3 className="form-section-title">{label}</h3>
            <div className="form-fields">
              {sectionFields.map((field) => (
                <FormField key={field.key} field={field} />
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

function FormField({ field }) {
  const { label, value, confidence, status, error_message: errorMessage } = field;
  const isEmpty = status === "empty";
  const title = isEmpty
    ? `${label} — non lu par l'OCR`
    : `${label} — confiance ${Math.round((confidence ?? 0) * 100)}%`;

  return (
    <label className={`form-field form-field--${status}`} title={title}>
      <span className="form-field-label">
        {label}
        <span className={`form-field-dot form-field-dot--${status}`} />
      </span>
      <span className={`form-field-input form-field-value--${status}`}>
        {isEmpty ? "—" : value}
      </span>
      <span className="form-field-note" data-status={status}>
        {status === "error" && errorMessage
          ? `⚠️ ${errorMessage}`
          : status === "warning"
            ? `⚡ ${errorMessage ?? `Confiance modérée (${Math.round((confidence ?? 0) * 100)}%)`}`
            : isEmpty
              ? "Champ non lu par l'OCR"
              : "✓ Lecture validée"}
      </span>
    </label>
  );
}