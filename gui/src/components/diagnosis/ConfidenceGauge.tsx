interface Props {
  value: number;
  threshold: number;
  size?: number;
}

export default function ConfidenceGauge({ value, threshold, size = 160 }: Props) {
  // value is already in percentage (0-100)
  const pct = Math.min(value, 100);
  const radius = size * 0.35;
  const strokeW = size * 0.08;
  const cx = size / 2;
  const cy = size * 0.55;

  const startAngle = -225;
  const endAngle = 45;
  const span = endAngle - startAngle;
  const filledAngle = startAngle + (span * pct) / 100;

  const toCartesian = (angleDeg: number, r: number) => {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
  };

  const bgStart = toCartesian(startAngle, radius);
  const bgEnd = toCartesian(endAngle, radius);
  const largeArc = pct > 50 ? 1 : 0;
  const fillEnd = toCartesian(filledAngle, radius);

  const bgPath =
    `M ${bgStart.x} ${bgStart.y} A ${radius} ${radius} 0 0 1 ${bgEnd.x} ${bgEnd.y}`;
  const fillPath =
    `M ${bgStart.x} ${bgStart.y} A ${radius} ${radius} 0 ${largeArc} 1 ${fillEnd.x} ${fillEnd.y}`;

  const color =
    pct >= 75 ? "#ef4444" : pct >= 50 ? "#eab308" : "#22c55e";

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.85} viewBox={`0 0 ${size} ${size}`}>
        <path
          d={bgPath}
          fill="none"
          stroke="#1f2937"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
        <path
          d={fillPath}
          fill="none"
          stroke={color}
          strokeWidth={strokeW}
          strokeLinecap="round"
          className="transition-all duration-700"
        />
        <text
          x={cx}
          y={cy + size * 0.02}
          textAnchor="middle"
          className="fill-white text-lg font-bold"
          fontSize={size * 0.13}
        >
          {pct.toFixed(1)}%
        </text>
        <text
          x={cx}
          y={cy + size * 0.14}
          textAnchor="middle"
          className="fill-gray-500"
          fontSize={size * 0.06}
        >
          Ngưỡng: {(threshold * 100).toFixed(0)}%
        </text>
      </svg>
    </div>
  );
}
