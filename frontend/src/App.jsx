import { useEffect, useRef, useState } from "react";

const SEQ_LEN = 30;
const APPEND_THRESHOLD = 0.50;  // Aggressive: lower for faster recognition
const APPEND_FRAMES = 1;  // Minimal: single confident prediction adds word
const CAPTURE_INTERVAL_MS = 25;  // Very fast: 40 FPS for rapid frame capture
const IDLE_WORD = "";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

function randomClientId() {
  if (window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `client-${Math.random().toString(36).slice(2, 12)}`;
}

function speakText(text) {
  if (!text) return;
  const utterance = new SpeechSynthesisUtterance(text.replace(/_/g, " "));
  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.lang = "en-US";
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

function formatLabel(word) {
  if (!word) return "···";
  return word.replace(/_/g, " ").toUpperCase();
}

function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? item : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(key, storedValue);
    } catch {
      // ignore
    }
  }, [key, storedValue]);

  return [storedValue, setStoredValue];
}

export default function App() {
  const videoRef = useRef(null);
  const previewCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const [clientId] = useLocalStorage("voice-client-id", randomClientId());
  const [cameraReady, setCameraReady] = useState(false);
  const [mirrorPreview, setMirrorPreview] = useState(true);
  const [status, setStatus] = useState("Starting camera...");
  const [prediction, setPrediction] = useState({ word: "", confidence: 0, top3: [] });
  const [landmarks, setLandmarks] = useState([]);
  const [bufferProgress, setBufferProgress] = useState(0);
  const [sentence, setSentence] = useState([]);
  const sentenceRef = useRef([]);
  const [sentenceBuilderActive, setSentenceBuilderActive] = useState(false);
  const [assistedCommunicationMode, setAssistedCommunicationMode] = useState(false);
  const [generatedSentence, setGeneratedSentence] = useState("");
  const prevWordsRef = useRef(null);
  const [sentenceStatusBase, setSentenceStatusBase] = useState("");
  const [sentenceStatusDotCount, setSentenceStatusDotCount] = useState(0);
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");
  const busyRef = useRef(false);
  const sentenceStatusTimerRef = useRef(null);
  const sentenceStatusTickerRef = useRef(null);
  const consecWordRef = useRef(null);
  const consecCountRef = useRef(0);
  const lastSpokenRef = useRef("");
  const builderCooldownRef = useRef({});
  const exitButtonRef = useRef(null);

  useEffect(() => {
    sentenceRef.current = sentence;
  }, [sentence]);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
          setCameraReady(true);
          setStatus("Camera ready");
        }
      } catch (err) {
        setError("Unable to access camera. Check permissions.");
        setStatus("Camera unavailable");
      }
    };
    startCamera();
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!cameraReady || busyRef.current || analyzing) return;
      captureFrame();
    }, CAPTURE_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [cameraReady, sentenceBuilderActive, autoSpeak, analyzing]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks.length) return;

    const handGroups = landmarks.reduce((acc, lm) => {
      const hand = (lm.hand || "left").toLowerCase();
      acc[hand] = acc[hand] || [];
      acc[hand].push(lm);
      return acc;
    }, {});

    ctx.strokeStyle = "rgba(0, 212, 255, 0.8)";
    ctx.fillStyle = "#7CFF7C";
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";
    ctx.textAlign = "center";

    const allPoints = [];

    Object.values(handGroups).forEach((handLandmarks) => {
      const sorted = [...handLandmarks].sort((a, b) => a.index - b.index);
      const points = sorted.map((lm) => ({
        x: mirrorPreview ? lm.x * canvas.width : (1 - lm.x) * canvas.width,
        y: lm.y * canvas.height,
      }));

      points.forEach(({ x, y }) => {
        allPoints.push({ x, y });
      });

      HAND_CONNECTIONS.forEach(([start, end]) => {
        const a = points[start];
        const b = points[end];
        if (!a || !b) return;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });

      points.forEach(({ x, y }) => {
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    });

    if (allPoints.length) {
      const xs = allPoints.map((p) => p.x);
      const ys = allPoints.map((p) => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const maxY = Math.max(...ys);
      const text = prediction.word ? `${prediction.word.replace(/_/g, " ")} ${Math.round(prediction.confidence * 100)}%` : "";
      if (text) {
        const textX = (minX + maxX) / 2;
        const textY = Math.min(maxY + 28, canvas.height - 10);
        ctx.fillText(text, textX, textY);
      }
    }
  }, [landmarks, mirrorPreview, prediction]);

  const captureFrame = async () => {
    const video = videoRef.current;
    const previewCanvas = previewCanvasRef.current;
    const captureCanvas = captureCanvasRef.current;
    if (!video || !previewCanvas || !captureCanvas) return;

    const previewCtx = previewCanvas.getContext("2d");
    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    if (mirrorPreview) {
      previewCtx.save();
      previewCtx.translate(previewCanvas.width, 0);
      previewCtx.scale(-1, 1);
      previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
      previewCtx.restore();
    } else {
      previewCtx.drawImage(video, 0, 0, previewCanvas.width, previewCanvas.height);
    }

    const captureCtx = captureCanvas.getContext("2d");
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const imageData = captureCanvas.toDataURL("image/jpeg", 0.30);
    busyRef.current = true;
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clientId, image: imageData }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || "Backend error");
      }
      const payload = await response.json();
      const word = payload.word || "";
      const confidence = payload.confidence || 0;
      const top3 = payload.top3 || [];
      setPrediction({ word, confidence, top3 });
      setLandmarks(payload.landmarks || []);
      setBufferProgress(payload.bufferProgress || 0);
      setError("");
      if (payload.bufferProgress < 1) {
        setStatus(`Warming buffer — ${Math.round(payload.bufferProgress * 100)}%`);
      } else {
        setStatus("Live recognition");
      }

      if (autoSpeak && word && word !== lastSpokenRef.current && confidence >= APPEND_THRESHOLD) {
        speakText(word);
        lastSpokenRef.current = word;
      }
      if (!word) {
        lastSpokenRef.current = "";
      }

      if (!analyzing) {
        maybeAppendWord(word, confidence);
      }
    } catch (err) {
      setError(err.message);
      setStatus("Connection error");
    } finally {
      busyRef.current = false;
    }
  };

  const maybeAppendWord = (word, confidence) => {
    if (!word || confidence < APPEND_THRESHOLD) {
      consecWordRef.current = null;
      consecCountRef.current = 0;
      return false;
    }
    if (word === consecWordRef.current) {
      consecCountRef.current += 1;
    } else {
      consecWordRef.current = word;
      consecCountRef.current = 1;
    }
    if (consecCountRef.current >= APPEND_FRAMES) {
      const lastWord = sentenceRef.current[sentenceRef.current.length - 1] || "";
      if (word === lastWord) {
        return false;
      }
      const now = Date.now();
      const lastAdded = builderCooldownRef.current[word] || 0;
      if (now - lastAdded < 600) {  // Reduced from 800ms to 600ms for faster back-to-back words
        return false;
      }
      builderCooldownRef.current[word] = now;
      const nextSentence = [...sentenceRef.current, word];
      sentenceRef.current = nextSentence;
      setSentence(nextSentence);
      consecCountRef.current = 0;
      return true;
    }
    return false;
  };

  const handleSpeakWord = () => {
    if (sentenceBuilderActive || analyzing) return;
    const lastWord = sentence[sentence.length - 1] || prediction.word;
    if (lastWord) speakText(lastWord);
  };

  const handleSpeakSentence = () => {
    if (analyzing) return;
    // Prefer the AI-generated sentence when available (assisted mode)
    if (generatedSentence) {
      speakText(generatedSentence);
      return;
    }
    if (sentenceBuilderActive) {
      // No generated sentence yet
      return;
    }
    if (!sentence.length) return;
    speakText(sentence.map((w) => w.replace(/_/g, " ")).join(" "));
  };

  const clearSentenceStatusTimer = () => {
    if (sentenceStatusTimerRef.current) {
      window.clearTimeout(sentenceStatusTimerRef.current);
      sentenceStatusTimerRef.current = null;
    }
    if (sentenceStatusTickerRef.current) {
      window.clearInterval(sentenceStatusTickerRef.current);
      sentenceStatusTickerRef.current = null;
    }
  };

  const handleRemove = () => {
    if (sentenceBuilderActive || analyzing) return;
    const nextSentence = sentence.slice(0, -1);
    sentenceRef.current = nextSentence;
    setSentence(nextSentence);
  };

  const handleClear = () => {
    if (analyzing) return;
    clearSentenceStatusTimer();
    sentenceRef.current = [];
    setSentence([]);
    setGeneratedSentence("");
    setSentenceStatusBase("");
    setSentenceStatusDotCount(0);
    consecWordRef.current = null;
    consecCountRef.current = 0;
    builderCooldownRef.current = {};
  };

  const handleStartSentenceBuilder = () => {
    clearSentenceStatusTimer();
    // Save current words history so we can restore on exit
    if (!assistedCommunicationMode) {
      prevWordsRef.current = sentenceRef.current ? [...sentenceRef.current] : [];
    }
    sentenceRef.current = [];
    setSentence([]);
    setGeneratedSentence("");
    setSentenceStatusBase("");
    setSentenceStatusDotCount(0);
    setSentenceBuilderActive(true);
    setAssistedCommunicationMode(true);
    setError("");
    consecWordRef.current = null;
    consecCountRef.current = 0;
    builderCooldownRef.current = {};
    setStatus("Assisted communication started");
  };

  const handleStopSentenceBuilder = async () => {
    if (!sentence.length) {
      setError("No collected words to generate a sentence.");
      return;
    }
    setSentenceBuilderActive(false);
    setAnalyzing(true);
    setGeneratedSentence("");
    setSentenceStatusBase("Analysing");
    setSentenceStatusDotCount(0);
    clearSentenceStatusTimer();
    sentenceStatusTickerRef.current = window.setInterval(() => {
      setSentenceStatusDotCount((count) => (count + 1) % 4);
    }, 500);
    setStatus("Analysing");
    setError("");

    try {
      const response = await fetch("/api/generate-sentence", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words: sentence }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Sentence generation failed");
      }
      const finalSentence = payload.sentence || "";
      clearSentenceStatusTimer();
      setGeneratedSentence(finalSentence);
      setSentenceStatusBase("");
      setSentenceStatusDotCount(0);
        setStatus("Sentence generated");
      setError("");
      setAnalyzing(false);
      if (autoSpeak && finalSentence) {
        speakText(finalSentence);
      }
    } catch (err) {
      clearSentenceStatusTimer();
      setSentenceStatusBase("");
      setSentenceStatusDotCount(0);
      setError(err.message);
      setStatus("Sentence generation failed");
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    if (generatedSentence && !sentenceBuilderActive && !analyzing) {
      exitButtonRef.current?.focus();
    }
  }, [generatedSentence, sentenceBuilderActive, analyzing]);

  const handleExitCommunicationMode = () => {
    if (analyzing) return;
    clearSentenceStatusTimer();
    setSentenceBuilderActive(false);
    setAssistedCommunicationMode(false);
    // Restore previous words history if available
    setGeneratedSentence("");
    setSentenceStatusBase("");
    setSentenceStatusDotCount(0);
    if (prevWordsRef.current) {
      sentenceRef.current = [...prevWordsRef.current];
      setSentence(sentenceRef.current);
      prevWordsRef.current = null;
    } else {
      sentenceRef.current = [];
      setSentence([]);
    }
    consecWordRef.current = null;
    consecCountRef.current = 0;
    builderCooldownRef.current = {};
    setError("");
    setStatus("Assisted communication exited");
  };

  return (
    <div className="page-shell">
      <div className="top-bar">
        <div>
          <div className="hero-title">VOICE</div>
          <div className="hero-subtitle">Real-Time Sign Language Communication</div>
        </div>
        <div className="status-pill">{status}</div>
      </div>

      <div className="grid-layout">
        <section className="panel panel-glow">
          <div className="panel-header">
            <span>Live camera</span>
            <span className="micro-status">{cameraReady ? "Ready" : "Waiting…"}</span>
          </div>
          <div className="camera-frame">
            <canvas ref={previewCanvasRef} className="camera-video-canvas" width={640} height={480} />
            <canvas
              ref={overlayCanvasRef}
              className="camera-overlay-canvas"
              width={640}
              height={480}
            />
            <canvas ref={captureCanvasRef} width={320} height={240} hidden />
            <video ref={videoRef} muted playsInline hidden />
          </div>
          <div className="button-row camera-button-row">
            <button
              onClick={sentenceBuilderActive ? handleStopSentenceBuilder : handleStartSentenceBuilder}
              className={sentenceBuilderActive ? "btn-active-comm" : "camera-start-btn"}
              disabled={analyzing}
              style={{
                position: "relative",
                paddingLeft: assistedCommunicationMode ? "32px" : "inherit"
              }}
            >
              {assistedCommunicationMode && !sentenceBuilderActive && (
                <span
                  style={{
                    position: "absolute",
                    left: "6px",
                    top: "50%",
                    transform: "translateY(-50%)",
                    width: "14px",
                    height: "14px",
                    backgroundColor: "#ff0000",
                    borderRadius: "50%",
                    display: "inline-block",
                    animation: "signal-blink 0.8s ease-in-out infinite",
                    boxShadow: "0 0 12px rgba(255, 0, 0, 1), 0 0 24px rgba(255, 0, 0, 0.7)"
                  }}
                />
              )}
              {sentenceBuilderActive ? "Stop detecting" : assistedCommunicationMode ? "Continue signing" : "Start assisted communication mode"}
            </button>
            {assistedCommunicationMode && (
              <button
                ref={exitButtonRef}
                onClick={handleExitCommunicationMode}
                disabled={analyzing}
              >
                Exit assisted communication mode
              </button>
            )}
          </div>
          {error ? <div className="error-banner">{error}</div> : null}
        </section>

        <section className="panel panel-glow panel-right">
          <div className="panel-header">
            <span>Prediction</span>
            <span className="micro-status">{prediction.word ? "Detected" : "Idle"}</span>
          </div>
          <div className="big-word">{formatLabel(prediction.word)}</div>
          <div className="confidence-card">
            <div>Confidence</div>
            <div className="confidence-score">
              {(!prediction.word || analyzing || !cameraReady) ? 0 : Math.round(prediction.confidence * 100)}%
            </div>
          </div>
          <div className="top3-list">
            {prediction.top3.map((item, index) => (
              <div key={index} className="top3-row">
                <span>{item.label}</span>
                <span>{Math.round(item.confidence * 100)}%</span>
              </div>
            ))}
          </div>

          <div className="toggle-row">
            <label className="toggle-switch">
              <input type="checkbox" checked={autoSpeak} onChange={() => setAutoSpeak(!autoSpeak)} />
              <span className="slider" />
            </label>
            <span>Auto speak</span>
          </div>
          <div className="button-row">
            <button onClick={handleSpeakWord} disabled={sentenceBuilderActive || analyzing || assistedCommunicationMode}>Speak word</button>
            <button onClick={handleSpeakSentence} disabled={analyzing || (sentenceBuilderActive && !generatedSentence)}>
              Speak sentence
            </button>
            <button onClick={handleRemove} disabled={sentenceBuilderActive || analyzing || assistedCommunicationMode}>Remove last</button>
            <button onClick={handleClear} disabled={analyzing || assistedCommunicationMode}>Clear</button>
          </div>
        </section>

        <section className="panel panel-glow panel-bottom">
          <div className="panel-header">
            <span>Words detected</span>
            <span className="micro-status">{sentenceBuilderActive ? "Recording" : analyzing ? "Analysing" : "Live"}</span>
          </div>
          <div className="sentence-box">
            {sentence.length > 0 ? sentence.map((word, index) => (
              <span key={`${word}-${index}`} className="sentence-token">{word.replace(/_/g, " ")}</span>
            )) : <span className="sentence-empty">Words history will appear here as you sign.</span>}
          </div>
        </section>
      </div>
      {generatedSentence ? (
        <div className="sentence-banner">{generatedSentence}</div>
      ) : sentenceStatusBase ? (
        <div className="sentence-status-banner">
          {sentenceStatusBase}{".".repeat(sentenceStatusDotCount)}
        </div>
      ) : null}
    </div>
  );
}
