import React from 'react';

export default function Architecture() {
  const steps = [
    { label: 'Text Input', color: '#60a5fa' },
    { label: 'Tokenization', color: '#818cf8' },
    { label: 'Embeddings', color: '#a78bfa' },
    { label: 'CNN (n-grams)', color: '#c084fc' },
    { label: 'BiLSTM', color: '#e879f9' },
    { label: 'Attention Fusion', color: '#f472b6' },
    { label: 'Start / End Heads', color: '#fb923c' },
    { label: 'Answer Extraction', color: '#34d399' },
  ];

  return (
    <div className="card architecture-card">
      <h2 className="card-title">Model Architecture</h2>
      <div className="pipeline">
        {steps.map((s, i) => (
          <React.Fragment key={i}>
            <div className="pipe-node" style={{ borderColor: s.color }}>
              <span className="pipe-label">{s.label}</span>
            </div>
            {i < steps.length - 1 && <span className="pipe-arrow">&#8594;</span>}
          </React.Fragment>
        ))}
      </div>
      <p className="arch-note">
        The <strong>CNN</strong> captures local n-gram patterns (bigrams, trigrams)
        before the <strong>BiLSTM</strong> models sequential context.
        <strong>Attention</strong> fuses question and context representations
        so that answer span prediction is conditioned on the question content.
      </p>
    </div>
  );
}
