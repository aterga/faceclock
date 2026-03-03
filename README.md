# FaceClock ⏱

AI Age Prediction · Powered by the Internet Computer

FaceClock is a full-stack decentralized application built on the Internet Computer (ICP). It uses `face-api.js` in the browser to detect faces via webcam, captures a sequence of keyframes, and sends them to a Rust-based backend canister for statistical age prediction.

## Architecture

- **Frontend**: Vanilla HTML/CSS/JS served from an ICP asset canister. Uses `face-api.js` (TinyFaceDetector + AgeGenderNet) for real-time client-side face detection and bounding box rendering.
- **Backend**: Rust canister that receives multiple face crops/estimates and performs statistical aggregation (IQR outlier rejection and confidence-weighted mean) to produce a final, robust age prediction.
- **Communication**: The frontend communicates with the backend via the `@dfinity/agent` library.

## Getting Started

### Prerequisites

- [DFINITY Canister SDK (`dfx`)](https://internetcomputer.org/docs/current/developer-docs/getting-started/install)
- [Rust Toolchain](https://www.rust-lang.org/tools/install) (with `wasm32-unknown-unknown` target)

### Installation

1. Clone the repository
2. Start the local Internet Computer replica in the background:
   ```bash
   dfx start --background
   ```
3. Deploy the canisters to your local replica:
   ```bash
   dfx deploy
   ```
4. Open the frontend URL provided in the `dfx deploy` output in your browser (usually `http://<canister-id>.localhost:4943/`).

## Project Structure

- `src/faceclock_backend/`: Contains the Rust source code for the backend canister.
  - `src/lib.rs`: The main logic for age prediction aggregation.
  - `faceclock_backend.did`: The Candid interface definition.
- `src/faceclock_frontend/`: Contains the frontend assets.
  - `assets/`: HTML, CSS, JavaScript, and the `face-api.js` models.
- `dfx.json`: The dfx project configuration file.

## Privacy

Face detection runs entirely locally in your browser. No images or video feeds are stored or transmitted. Only the structured data (the estimated age and confidence per frame) is sent to the backend canister for final statistical processing.

## License

This project is open-source and available under the MIT License.
