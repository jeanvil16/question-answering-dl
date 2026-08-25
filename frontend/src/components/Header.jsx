import React from 'react';

export default function Header({ health }) {
  const loaded = health?.model_loaded;
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-title">
          <span className="logo">QA&middot;DL</span>
          <h1>Neural Question Answering System</h1>
          <p className="subtitle">
            Extractive QA &middot; CNN + BiLSTM + Attention &middot; Deep Learning Techniques
          </p>
        </div>
        <div className={`status-pill ${loaded ? 'online' : 'offline'}`}>
          <span className="dot" />
          {loaded ? 'Model Loaded' : 'Model Not Loaded'}
        </div>
      </div>
    </header>
  );
}
