import React from 'react';

export default function AnswerPanel({ result, loading, context }) {
  if (loading) {
    return (
      <div className="card answer-card">
        <div className="skeleton-block" />
        <div className="skeleton-line short" />
        <div className="skeleton-line medium" />
        <p className="skeleton-label">Running CNN + BiLSTM inference...</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="card answer-card empty-state">
        <div className="empty-icon">?</div>
        <p>Ask a question to see the model's answer here.</p>
      </div>
    );
  }

  const { answer, confidence, start_char, end_char, start_token, end_token, latency_ms, context_tokens, question_tokens, truncated } = result;
  const confPct = Math.round(confidence * 100);
  const level = confPct >= 70 ? 'high' : confPct >= 40 ? 'medium' : 'low';

  const highlighted = context && start_char !== undefined
    ? (
      <>
        {context.slice(0, start_char)}
        <mark className={`highlight-${level}`}>{answer}</mark>
        {context.slice(end_char)}
      </>
    )
    : answer;

  return (
    <div className="card answer-card">
      <h2 className="card-title">Answer</h2>
      <div className="answer-text">{answer || <em>No answer found</em>}</div>

      <div className="confidence-block">
        <span className="conf-label">Confidence</span>
        <div className="conf-bar-bg">
          <div className={`conf-bar conf-${level}`} style={{ width: `${confPct}%` }} />
        </div>
        <span className={`conf-value conf-${level}`}>{confPct}%</span>
        <span className={`conf-tag conf-tag-${level}`}>{level}</span>
      </div>

      <div className="meta-row">
        <span>Latency: <strong>{latency_ms}ms</strong></span>
        <span>Context tokens: {context_tokens}</span>
        <span>Question tokens: {question_tokens}</span>
        <span>Span: [{start_token}, {end_token}]</span>
        {truncated && <span className="warn">Input was truncated</span>}
      </div>

      {context && (
        <div className="highlighted-context">
          <h3>Context with Highlighted Answer</h3>
          <p>{highlighted}</p>
        </div>
      )}
    </div>
  );
}
