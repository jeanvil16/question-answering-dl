import React, { useState, useEffect, useCallback } from 'react';
import { apiHealth, apiPredict, apiModelInfo, apiHistory, SAMPLES } from './api';
import Header from './components/Header';
import QAForm from './components/QAForm';
import AnswerPanel from './components/AnswerPanel';
import Samples from './components/Samples';
import Architecture from './components/Architecture';
import ChartsCard from './components/ChartsCard';
import ModelInfo from './components/ModelInfo';

export default function App() {
  const [context, setContext] = useState('');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [history, setHistory] = useState(null);

  useEffect(() => {
    (async () => {
      try { setHealth(await apiHealth()); } catch {}
      try { setModelInfo(await apiModelInfo()); } catch {}
      try { setHistory(await apiHistory()); } catch {}
    })();
    const id = setInterval(async () => {
      try { setHealth(await apiHealth()); } catch {}
    }, 20000);
    return () => clearInterval(id);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!context.trim() || !question.trim()) {
      setError('Please provide both context and a question.');
      return;
    }
    if (context.trim().length < 10) {
      setError('Context must be at least 10 characters long.');
      return;
    }
    if (question.trim().length < 3) {
      setError('Question must be at least 3 characters long.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await apiPredict(context.trim(), question.trim());
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [context, question]);

  const handleClear = useCallback(() => {
    setContext('');
    setQuestion('');
    setResult(null);
    setError(null);
  }, []);

  const handleSample = useCallback((s) => {
    setContext(s.context);
    setQuestion(s.question);
    setResult(null);
    setError(null);
  }, []);

  return (
    <div className="app">
      <Header health={health} />

      <main className="main">
        <section className="qa-section">
          <div className="form-col">
            <QAForm
              context={context}
              setContext={setContext}
              question={question}
              setQuestion={setQuestion}
              onSubmit={handleSubmit}
              onClear={handleClear}
              loading={loading}
              error={error}
            />
            <Samples samples={SAMPLES} onSelect={handleSample} />
          </div>
          <div className="result-col">
            <AnswerPanel result={result} loading={loading} context={context} />
          </div>
        </section>

        <section className="info-section">
          <Architecture />
          <ModelInfo info={modelInfo} health={health} />
          <ChartsCard history={history} />
        </section>
      </main>

      <footer className="footer">
        <p>Deep Learning Techniques Mini-Project &middot; PyTorch CNN + BiLSTM + Attention &middot; Flask API</p>
      </footer>
    </div>
  );
}
