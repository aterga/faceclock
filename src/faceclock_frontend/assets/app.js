import { HttpAgent, Actor } from 'https://esm.run/@dfinity/agent';

/**
 * FaceClock – Main Application Logic
 *
 * Flow:
 * 1. Load face-api.js models (TinyFaceDetector + AgeGenderNet)
 * 2. Start webcam
 * 3. User clicks "Start Scanning"
 * 4. Detect face continuously for ~3 seconds
 * 5. Capture keyframes with age estimates
 * 6. Send to backend canister for aggregated prediction
 * 7. Display result
 */

// ============================================================
// Configuration
// ============================================================

const BACKEND_CANISTER_ID = 'uxrrr-q7777-77774-qaaaq-cai';

const CONFIG = {
    SCAN_DURATION_MS: 2000,      // How long to scan
    MIN_FACE_CONFIDENCE: 0.5,    // Minimum face detection confidence
    DETECTION_INTERVAL_MS: 300,  // How often to run detection (less frequent now since we just need one good crop)
    MODEL_PATH: './models',      // Path to face-api.js model weights
};

// ============================================================
// State
// ============================================================

let state = {
    modelsLoaded: false,
    cameraReady: false,
    scanning: false,
    detectionLoop: null,
    scanStartTime: null,
    frameData: [],
};

// ============================================================
// DOM References
// ============================================================

const $ = (sel) => document.querySelector(sel);
const webcam = $('#webcam');
const overlay = $('#overlay');
const scanFrame = $('#scanFrame');
const statusBadge = $('#statusBadge');
const statusText = $('#statusText');
const startBtn = $('#startBtn');
const resetBtn = $('#resetBtn');
const progressRing = $('#progressRing');
const progressCircle = $('#progressCircle');
const progressText = $('#progressText');
const resultSection = $('#resultSection');
const resultAge = $('#resultAge');
const resultConfidence = $('#resultConfidence');
const resultFrames = $('#resultFrames');

// ============================================================
// Status Helpers
// ============================================================

function setStatus(text, type = '') {
    statusText.textContent = text;
    statusBadge.className = 'status-badge ' + type;
}

function setProgress(pct) {
    const circumference = 2 * Math.PI * 54; // r=54
    const offset = circumference - (pct / 100) * circumference;
    progressCircle.style.strokeDashoffset = offset;
    progressText.textContent = Math.round(pct) + '%';
}

// ============================================================
// Initialize
// ============================================================

async function init() {
    setStatus('Loading AI models...', '');

    try {
        // Load face-api.js models (only detection now)
        await faceapi.nets.tinyFaceDetector.loadFromUri(CONFIG.MODEL_PATH);
        state.modelsLoaded = true;
        setStatus('Starting camera...', '');
    } catch (err) {
        console.error('Failed to load models:', err);
        setStatus('Failed to load AI models', 'error');
        return;
    }

    try {
        // Request webcam access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user',
            },
            audio: false,
        });
        webcam.srcObject = stream;

        // Wait for video to be ready
        await new Promise((resolve) => {
            webcam.onloadedmetadata = () => {
                webcam.play();
                resolve();
            };
        });

        // Set canvas size to match video
        overlay.width = webcam.videoWidth;
        overlay.height = webcam.videoHeight;

        state.cameraReady = true;
        startBtn.disabled = false;
        setStatus('Ready — click Start Scanning', '');
    } catch (err) {
        console.error('Camera access denied:', err);
        setStatus('Camera access denied', 'error');
        return;
    }
}

// ============================================================
// Face Detection Loop
// ============================================================

async function detectFace() {
    if (!state.scanning) return;

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    const detection = await faceapi
        .detectSingleFace(webcam, new faceapi.TinyFaceDetectorOptions({
            inputSize: 320,
            scoreThreshold: CONFIG.MIN_FACE_CONFIDENCE,
        }));

    if (detection) {
        // Draw face bounding box
        const box = detection.box;
        ctx.strokeStyle = '#a29bfe';
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // Record frame data (crop the face)
        if (state.scanStartTime) {

            // Extract face image crop onto a temporary hidden canvas
            const cropCanvas = document.createElement('canvas');
            // Expand the bounding box slightly to capture full head context
            const padX = box.width * 0.2;
            const padY = box.height * 0.2;
            const sx = Math.max(0, box.x - padX);
            const sy = Math.max(0, box.y - padY);
            const sw = Math.min(webcam.videoWidth - sx, box.width + padX * 2);
            const sh = Math.min(webcam.videoHeight - sy, box.height + padY * 2);

            cropCanvas.width = sw;
            cropCanvas.height = sh;
            const cropCtx = cropCanvas.getContext('2d');
            cropCtx.drawImage(webcam, sx, sy, sw, sh, 0, 0, sw, sh);

            // Get JPEG blob
            const dataUrl = cropCanvas.toDataURL('image/jpeg', 0.9);

            state.frameData.push({
                image_data: dataUrl,
                confidence: detection.score,
                size: sw * sh
            });

            // Update progress
            const elapsed = Date.now() - state.scanStartTime;
            const progress = Math.min(100, (elapsed / CONFIG.SCAN_DURATION_MS) * 100);
            setProgress(progress);

            // Check if scan is complete
            if (elapsed >= CONFIG.SCAN_DURATION_MS && state.frameData.length > 0) {
                finishScanning();
                return;
            }
        }

        // First face detected — start the scan timer
        if (!state.scanStartTime) {
            state.scanStartTime = Date.now();
            scanFrame.classList.add('active');
            progressRing.style.display = '';
            setStatus('Face detected — scanning...', 'scanning');
        }
    } else {
        // No face — pause the scan timer
        if (state.scanStartTime) {
            setStatus('Face lost — hold still', 'detecting');
        }
    }

    // Continue loop
    if (state.scanning) {
        state.detectionLoop = setTimeout(detectFace, CONFIG.DETECTION_INTERVAL_MS);
    }
}

// ============================================================
// Scanning Flow
// ============================================================

function startScanning() {
    state.scanning = true;
    state.scanStartTime = null;
    state.frameData = [];
    resultSection.style.display = 'none';
    startBtn.style.display = 'none';
    resetBtn.style.display = 'none';
    setProgress(0);
    setStatus('Looking for face...', 'detecting');

    detectFace();
}

async function finishScanning() {
    state.scanning = false;
    if (state.detectionLoop) {
        clearTimeout(state.detectionLoop);
        state.detectionLoop = null;
    }

    scanFrame.classList.remove('active');
    progressRing.style.display = 'none';
    setStatus('Sending to canister...', 'processing');

    // Clear overlay
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    try {
        if (state.frameData.length === 0) throw new Error("No face captured");

        // Select the "best" frame (highest score/size combination)
        state.frameData.sort((a, b) => (b.confidence * b.size) - (a.confidence * a.size));
        const bestImageObj = state.frameData[0];

        // Convert to UInt8Array for backend
        const res = await fetch(bestImageObj.image_data);
        const blob = await res.blob();
        const arrayBuffer = await blob.arrayBuffer();
        const imageBytes = new Uint8Array(arrayBuffer);

        // Call backend canister
        const resultAge = await callBackend(imageBytes);

        // Display result (Mocking confidence/frames used for the UI structure)
        showResult({
            predicted_age: resultAge,
            confidence: bestImageObj.confidence,
            frames_used: 1 // We sent 1 image to the backend
        });
        setStatus('Prediction complete!', 'done');
    } catch (err) {
        console.error('Prediction failed:', err);
        setStatus('Backend Error: ' + (err.message || String(err)), 'error');
        resetBtn.style.display = '';
    }
}

// ============================================================
// Utility / Helpers
// ============================================================
// (Removed selectKeyframes since we only send 1 best crop now)

function reset() {
    resultSection.style.display = 'none';
    resetBtn.style.display = 'none';
    startBtn.style.display = '';
    scanFrame.classList.remove('active');
    progressRing.style.display = 'none';
    setProgress(0);
    setStatus('Ready — click Start Scanning', '');

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

// ============================================================
// Backend Communication (ICP)
// ============================================================

/**
 * Candid Interface Definition
 */
const idlFactory = ({ IDL }) => {
    return IDL.Service({
        'predict_age_from_image': IDL.Func([IDL.Vec(IDL.Nat8)], [IDL.Float64], []),
    });
};

async function callBackend(imageBytes) {
    setStatus('Predicting on-chain...', 'processing');

    try {
        const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.hostname.endsWith('.localhost');
        const agent = new HttpAgent({
            host: isLocal ? `http://${window.location.host}` : 'https://icp-api.io',
        });

        if (isLocal) {
            await agent.fetchRootKey();
        }

        const actor = Actor.createActor(idlFactory, {
            agent,
            canisterId: BACKEND_CANISTER_ID,
        });

        // Call the backend service
        const ageResult = await actor.predict_age_from_image(imageBytes);

        return ageResult;
    } catch (err) {
        console.error('Failed to call backend canister:', err);
        throw new Error('Could not reach ICP canister: ' + (err.message || 'Unknown network error.'));
    }
}


// ============================================================
// Result Display
// ============================================================

function showResult(result) {
    resultSection.style.display = '';
    resetBtn.style.display = '';

    // Animate the age number counting up
    const targetAge = Math.round(result.predicted_age);
    animateNumber(resultAge, 0, targetAge, 1000);

    resultConfidence.textContent = Math.round(result.confidence * 100) + '%';
    resultFrames.textContent = result.frames_used;
}

function animateNumber(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(start + (end - start) * eased);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ============================================================
// Event Listeners
// ============================================================

startBtn.addEventListener('click', startScanning);
resetBtn.addEventListener('click', reset);

// ============================================================
// Boot
// ============================================================

document.addEventListener('DOMContentLoaded', init);
