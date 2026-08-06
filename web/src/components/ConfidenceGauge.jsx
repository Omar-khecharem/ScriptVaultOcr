export default function ConfidenceGauge({ value }) {
  const pct = value == null ? null : Math.max(0, Math.min(100, value));
  const R = 26;
  const C = 2 * Math.PI * R;
  const color =
    pct == null
      ? "var(--muted)"
      : pct >= 70
        ? "var(--success)"
        : pct >= 50
          ? "var(--warning)"
          : "var(--danger)";

  return (
    <svg
      className="gauge"
      width="58"
      height="58"
      viewBox="0 0 58 58"
      role="img"
      aria-label="Confiance moyenne"
    >
      <circle
        className="track"
        cx="29"
        cy="29"
        r={R}
        fill="none"
        strokeWidth="7"
        strokeLinecap="round"
        strokeDasharray={`${C * 0.75} ${C}`}
        transform="rotate(135 29 29)"
      />
      {pct != null && (
        <circle
          className="value"
          cx="29"
          cy="29"
          r={R}
          fill="none"
          stroke={color}
          strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${(C * 0.75 * pct) / 100} ${C}`}
          transform="rotate(135 29 29)"
        />
      )}
      <text x="29" y="33.5">
        {pct == null ? "—" : `${Math.round(pct)}%`}
      </text>
    </svg>
  );
}
