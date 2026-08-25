import React from 'react';

const W = 520, H = 200, PAD = { t: 25, r: 20, b: 35, l: 55 };

function LineChart({ data, series, title, yLabel, color1, color2 }) {
  if (!data.length) return <p className="no-data">No data available yet.</p>;

  const keys = Object.keys(series);
  const allVals = data.flatMap((d) => keys.map((k) => d[k] ?? 0));
  const yMin = Math.min(...allVals), yMax = Math.max(...allVals);
  const yPad = (yMax - yMin) * 0.1 || 0.5;
  const y0 = yMin - yPad, y1 = yMax + yPad;
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;
  const n = data.length;
  const xStep = plotW / Math.max(1, n - 1);

  const mapX = (i) => PAD.l + i * xStep;
  const mapY = (v) => PAD.t + plotH - ((v - y0) / (y1 - y0)) * plotH;

  const pathD = (key) =>
    data.map((d, i) => `${i === 0 ? 'M' : 'L'}${mapX(i).toFixed(1)},${mapY(d[key]).toFixed(1)}`).join(' ');

  const colors = [color1 || '#60a5fa', color2 || '#f472b6'];
  const yTicks = 5;
  const xTicks = Math.min(10, n);
  const xTickEvery = Math.max(1, Math.floor(n / xTicks));

  return (
    <div className="chart-wrap">
      <h3 className="chart-title">{title}</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg">
        {/* grid */}
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const v = y0 + (i / yTicks) * (y1 - y0);
          return (
            <g key={i}>
              <line x1={PAD.l} y1={mapY(v)} x2={W - PAD.r} y2={mapY(v)} stroke="#ffffff10" />
              <text x={PAD.l - 8} y={mapY(v) + 4} textAnchor="end" className="axis-text">
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}
        {data.map((d, i) =>
          i % xTickEvery === 0 ? (
            <text key={i} x={mapX(i)} y={H - 8} textAnchor="middle" className="axis-text">
              {d.epoch}
            </text>
          ) : null
        )}
        <text x={PAD.l - 8} y={10} textAnchor="end" className="axis-label">{yLabel}</text>
        {/* lines */}
        {keys.map((key, ki) => (
          <path key={key} d={pathD(key)} fill="none" stroke={colors[ki]}
            strokeWidth={2} strokeLinecap="round" />
        ))}
        {/* legend */}
        {keys.map((key, ki) => (
          <g key={key + 'leg'} transform={`translate(${PAD.l + 8 + ki * 110}, ${PAD.t})`}>
            <line x1={0} y1={0} x2={18} y2={0} stroke={colors[ki]} strokeWidth={2} />
            <text x={22} y={4} className="legend-text">{series[key]}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export default function ChartsCard({ history }) {
  const data = history?.history || [];
  const available = history?.available ?? data.length > 0;
  if (!available) {
    return (
      <div className="card charts-card">
        <h2 className="card-title">Training History</h2>
        <p className="no-data">No training history found. Train the model first.</p>
      </div>
    );
  }

  return (
    <div className="card charts-card">
      <h2 className="card-title">Training History</h2>
      <div className="charts-row">
        <LineChart
          data={data}
          series={{ train_loss: 'Train Loss', val_loss: 'Val Loss' }}
          title="Loss"
          yLabel="Loss"
          color1="#60a5fa"
          color2="#f472b6"
        />
        <LineChart
          data={data}
          series={{ train_em: 'Train EM', val_em: 'Val EM', train_f1: 'Train F1', val_f1: 'Val F1' }}
          title="Accuracy (EM / F1)"
          yLabel="Score"
          color1="#34d399"
          color2="#fb923c"
        />
      </div>
    </div>
  );
}
