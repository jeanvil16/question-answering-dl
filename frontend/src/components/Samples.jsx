import React from 'react';

export default function Samples({ samples, onSelect }) {
  return (
    <div className="card samples-card">
      <h2 className="card-title">Try an Example</h2>
      <div className="sample-chips">
        {samples.map((s, i) => (
          <button key={i} className="chip" onClick={() => onSelect(s)} title={s.question}>
            {s.title}
          </button>
        ))}
      </div>
    </div>
  );
}
