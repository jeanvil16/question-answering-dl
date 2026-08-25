import React, { useState } from 'react';

export default function QAForm({ context, setContext, question, setQuestion, onSubmit, onClear, loading, error }) {
  const [contextFocused, setContextFocused] = useState(false);

  return (
    <div className="card qa-form-card">
      <h2 className="card-title">Ask a Question</h2>
      <p className="card-desc">Provide a passage and ask any factual question about it.</p>

      <label className="field-label" htmlFor="context">Context / Passage</label>
      <textarea
        id="context"
        className={`field textarea ${contextFocused && context.trim().length < 10 && context.length > 0 ? 'field-error' : ''}`}
        rows={8}
        placeholder="Paste or type a passage here (min 10 characters)..."
        value={context}
        onChange={(e) => setContext(e.target.value)}
        onFocus={() => setContextFocused(true)}
        onBlur={() => setContextFocused(false)}
      />
      <span className="char-count">{context.length} chars</span>

      <label className="field-label" htmlFor="question">Question</label>
      <input
        id="question"
        className="field input"
        type="text"
        placeholder="e.g. Where does photosynthesis take place?"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => { if (e.key === 'Enter') onSubmit(); }}
      />

      {error && <div className="error-msg">{error}</div>}

      <div className="btn-row">
        <button className="btn btn-primary" onClick={onSubmit} disabled={loading}>
          {loading ? <><span className="spinner" /> Running...</> : 'Get Answer'}
        </button>
        <button className="btn btn-ghost" onClick={onClear} disabled={loading}>Clear</button>
      </div>
    </div>
  );
}
