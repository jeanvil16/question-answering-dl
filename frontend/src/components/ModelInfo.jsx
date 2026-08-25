import React from 'react';

export default function ModelInfo({ info, health }) {
  if (!info && !health) return (
    <div className="card model-info-card"><h2 className="card-title">Model Information</h2><p className="no-data">Loading...</p></div>
  );

  const cfg = info?.config || {};
  const best = info?.best_metrics || {};
  const params = info?.trainable_parameters;
  const loaded = health?.model_loaded;

  return (
    <div className="card model-info-card">
      <h2 className="card-title">Model Information</h2>
      {!loaded ? (
        <p className="no-data">Model not loaded. Train with <code>python training/train.py</code></p>
      ) : (
        <>
          <div className="info-grid">
            <InfoItem label="Parameters" value={params ? params.toLocaleString() : '-'} />
            <InfoItem label="Embedding dim" value={cfg.embed_dim} />
            <InfoItem label="CNN kernels" value={cfg.cnn_kernel_sizes?.join(', ')} />
            <InfoItem label="CNN filters" value={cfg.cnn_num_filters} />
            <InfoItem label="CNN output dim" value={cfg.cnn_out_dim} />
            <InfoItem label="RNN hidden dim" value={cfg.rnn_hidden_dim} />
            <InfoItem label="Max context len" value={cfg.max_context_len} />
            <InfoItem label="Max answer len" value={cfg.max_answer_len} />
            <InfoItem label="Dropout" value={cfg.dropout} />
            <InfoItem label="Device" value={health?.device} />
          </div>
          {best.epoch && (
            <div className="best-metrics">
              <h3>Best Training Results</h3>
              <span>Epoch {best.epoch}</span>
              <span>EM: {(best.val_em * 100).toFixed(1)}%</span>
              <span>F1: {(best.val_f1 * 100).toFixed(1)}%</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function InfoItem({ label, value }) {
  return (
    <div className="info-item">
      <span className="info-label">{label}</span>
      <span className="info-value">{value ?? '-'}</span>
    </div>
  );
}
