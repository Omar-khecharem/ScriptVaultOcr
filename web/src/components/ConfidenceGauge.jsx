/**
 * Jauge de confiance circulaire (SVG) — % moyen des fichiers traités.
 */
export default function ConfidenceGauge({ value }) {
  const pct = value == null ? null : Math.max(0, Math.min(100, value));
  const circumference = 2 * Math.PI * 26;
  const color = pct == null ? "var(--muted)" : pct >= 70 ? "var(--success)" : pct >= 50 ? "var(--warning)" : "var(--danger)";

  return (
    <svg className="gauge" width="70" height="70" viewBox="0 0 70 70" role="img" aria-label="Confiance moyenne">
      <circle
        cx="35"
        cy="35"
        r="26"
        fill="none"
        stroke="var(--surface-alt)"
        strokeWidth="7"
        strokeLinecap="round"
        strokeDasharray={`${circumference * 0.75} ${circumference}`}
        transform="rotate(135 35 35)"
      />
      {pct != null && (
        <circle
          cx="35"
          cy="35"
          r="26"
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${circumference * 0.75 * (pct / 100)} ${circumference}`}
          transform="rotate(135 35 35)"
          style={{ transition: "stroke-dasharray 0.3s ease" }}
        />
      )}
      <text x="35" y="40">
        {pct == null ? "—" : `${Math.round(pct)}%`}
      </text>
    </svg>
  );
}
