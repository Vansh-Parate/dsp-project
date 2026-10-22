const micButton = document.querySelector('#micButton');
const micButtonText = document.querySelector('#micButtonText');
const inputTitle = document.querySelector('#inputTitle');
const inputPill = document.querySelector('#inputPill');
const connectionLabel = document.querySelector('#connectionLabel');
const streamLabel = document.querySelector('#streamLabel');
const emptyVisualizer = document.querySelector('#emptyVisualizer');
const waveform = document.querySelector('#waveform');
const canvasContext = waveform.getContext('2d');
const timer = document.querySelector('#timer');
const levelValue = document.querySelector('#levelValue');
const snrValue = document.querySelector('#snrValue');
const enhancementValue = document.querySelector('#enhancementValue');
const enhancementCaption = document.querySelector('#enhancementCaption');
const levelMeter = document.querySelector('#levelMeter');
const gainBar = document.querySelector('#gainBar');
const updatedLabel = document.querySelector('#updatedLabel');
const insightTitle = document.querySelector('#insightTitle');
const insightBody = document.querySelector('#insightBody');

let audioContext;
let analyser;
let source;
let stream;
let animationFrame;
let startedAt;
let noiseFloor = 0.012;
let frames = 0;

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  waveform.width = waveform.clientWidth * ratio;
  waveform.height = waveform.clientHeight * ratio;
  canvasContext.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawIdle() {
  resizeCanvas();
  const width = waveform.clientWidth;
  const height = waveform.clientHeight;
  canvasContext.clearRect(0, 0, width, height);
  canvasContext.beginPath();
  canvasContext.strokeStyle = 'rgba(101,121,94,.32)';
  canvasContext.lineWidth = 1.5;
  canvasContext.moveTo(0, height / 2);
  canvasContext.lineTo(width, height / 2);
  canvasContext.stroke();
}

function formatTime(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

function drawFrame() {
  if (!analyser) return;
  const width = waveform.clientWidth;
  const height = waveform.clientHeight;
  const samples = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(samples);
  canvasContext.clearRect(0, 0, width, height);
  canvasContext.beginPath();
  canvasContext.strokeStyle = '#65795e';
  canvasContext.lineWidth = 1.5;
  let sum = 0;
  samples.forEach((sample, index) => {
    const value = (sample - 128) / 128;
    sum += value * value;
    const x = index / (samples.length - 1) * width;
    const y = height / 2 + value * height * 1.2;
    index === 0 ? canvasContext.moveTo(x, y) : canvasContext.lineTo(x, y);
  });
  canvasContext.stroke();

  const rms = Math.sqrt(sum / samples.length);
  const db = Math.max(-60, 20 * Math.log10(Math.max(rms, 0.0001)));
  if (frames < 100) noiseFloor = noiseFloor * .985 + rms * .015;
  const snr = Math.max(0, Math.min(30, 20 * Math.log10(Math.max(rms / Math.max(noiseFloor, .001), 1))));
  const gain = Math.max(0, Math.min(99, 48 + snr * 1.25));
  levelValue.textContent = db.toFixed(1);
  snrValue.textContent = snr.toFixed(1);
  enhancementValue.textContent = gain.toFixed(0);
  levelMeter.style.width = `${Math.max(2, Math.min(100, (db + 60) / 60 * 100))}%`;
  gainBar.style.width = `${gain}%`;
  enhancementCaption.textContent = snr > 12 ? 'Speech signal is coming through clearly' : 'Adaptive gain is opening up the signal';
  updatedLabel.textContent = 'updated just now';
  timer.textContent = formatTime((performance.now() - startedAt) / 1000);
  frames += 1;
  animationFrame = requestAnimationFrame(drawFrame);
}

async function startListening() {
  try {
    micButton.disabled = true;
    micButtonText.textContent = 'Requesting permission…';
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    analyser.smoothingTimeConstant = .72;
    source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    startedAt = performance.now();
    inputTitle.textContent = 'Listening to your room';
    inputPill.textContent = 'LIVE';
    inputPill.style.color = '#65795e';
    connectionLabel.textContent = 'live session';
    streamLabel.textContent = 'MIC STREAM ACTIVE';
    emptyVisualizer.style.display = 'none';
    micButtonText.textContent = 'Stop listening';
    micButton.querySelector('.button-icon').textContent = '■';
    micButton.disabled = false;
    insightTitle.textContent = 'A live signal, made legible.';
    insightBody.textContent = 'The waveform shows your raw input. The readout below estimates clarity and adaptive enhancement as you speak.';
    drawFrame();
  } catch (error) {
    micButton.disabled = false;
    micButtonText.textContent = 'Allow microphone access';
    insightTitle.textContent = 'Microphone access is needed.';
    insightBody.textContent = 'Your browser blocked the request. Check the site permissions in the address bar, then try again.';
    connectionLabel.textContent = 'permission needed';
  }
}

function stopListening() {
  cancelAnimationFrame(animationFrame);
  stream?.getTracks().forEach(track => track.stop());
  source?.disconnect();
  audioContext?.close();
  analyser = null;
  inputTitle.textContent = 'Microphone is resting';
  inputPill.textContent = 'IDLE';
  inputPill.style.color = '';
  connectionLabel.textContent = 'ready to listen';
  streamLabel.textContent = 'NO STREAM';
  emptyVisualizer.style.display = 'flex';
  micButtonText.textContent = 'Allow microphone access';
  micButton.querySelector('.button-icon').textContent = '◉';
  drawIdle();
}

micButton.addEventListener('click', () => analyser ? stopListening() : startListening());
window.addEventListener('resize', () => { if (analyser) resizeCanvas(); else drawIdle(); });
drawIdle();
