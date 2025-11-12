import React, { useMemo } from "react";

function toPoints(data, width, height, accessor) {
  if (!data.length) return "";
  const values = data.map(accessor);
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max === min) {
    return data
      .map((_, index) => {
        const x = (index / Math.max(1, data.length - 1)) * width;
        const y = height / 2;
        return `${x},${y}`;
      })
      .join(" ");
  }
  return data
    .map((point, index) => {
      const value = accessor(point);
      const x = (index / Math.max(1, data.length - 1)) * width;
      const y = height - ((value - min) / (max - min)) * height;
      return `${x},${y}`;
    })
    .join(" ");
}

export default function Sparkline({
  data,
  width = 320,
  height = 120,
  stroke = "var(--color-accent)",
  fill = "rgba(37, 99, 235, 0.12)",
  accessor = (point) => point.close ?? 0,
}) {
  const points = useMemo(() => toPoints(data, width, height, accessor), [data, width, height, accessor]);
  const gradientId = useMemo(() => `sparkline-${Math.random().toString(36).slice(2, 11)}`, []);

  if (!points) {
    return (
      <div className="flex h-24 w-full items-center justify-center text-sm text-[var(--color-text-muted)]">
        No chart data
      </div>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="h-24 w-full"
      role="img"
      aria-label="Sparkline chart"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={fill} stopOpacity="0.8" />
          <stop offset="100%" stopColor={fill} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth="2"
        points={points}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <polygon
        fill={`url(#${gradientId})`}
        points={`${points} ${width},${height} 0,${height}`}
      />
    </svg>
  );
}

