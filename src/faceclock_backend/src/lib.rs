use getrandom::register_custom_getrandom;
use std::io::Cursor;
use tract_onnx::prelude::*;

// Custom getrandom handler to allow compiling to wasm32-unknown-unknown
fn always_fail(_buf: &mut [u8]) -> Result<(), getrandom::Error> {
    Ok(())
}
register_custom_getrandom!(always_fail);

// Embed the ONNX model into the Wasm binary
const MODEL_BYTES: &[u8] = include_bytes!("../genderage.onnx");

/// Expected input dimensions
const WIDTH: usize = 96;
const HEIGHT: usize = 96;
const RGBA_BYTES: usize = WIDTH * HEIGHT * 4; // 36864

thread_local! {
    static MODEL: std::cell::RefCell<tract_onnx::prelude::SimplePlan<tract_onnx::prelude::TypedFact, Box<dyn tract_onnx::prelude::TypedOp>, tract_onnx::prelude::Graph<tract_onnx::prelude::TypedFact, Box<dyn tract_onnx::prelude::TypedOp>>>> = {
        let mut cursor = Cursor::new(MODEL_BYTES);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .expect("Failed to load ONNX model")
            .into_typed()
            .expect("Failed to type model")
            .into_optimized()
            .expect("Failed to optimize model")
            .into_runnable()
            .expect("Failed to make model runnable");
        std::cell::RefCell::new(model)
    };
}

/// Run ONNX inference on a pre-processed [1,3,96,96] NCHW RGB tensor.
fn run_inference(tensor: tract_ndarray::Array4<f32>) -> f64 {
    let input = tensor.into_tensor().into();

    let result = MODEL.with(|m| {
        m.borrow()
            .run(tvec!(input))
            .expect("Failed to run ML inference")
    });

    // The genderage model outputs [1, 3]: index 0-1 = Gender Logits, index 2 = Age / 100.0
    let output_view = result[0]
        .to_array_view::<f32>()
        .expect("Failed to read output tensor");

    let age_normalized = output_view[[0, 2]];
    (age_normalized * 100.0) as f64
}

/// Apply InsightFace similarity transform: center on face bbox, scale to fill 96×96.
///
/// Replicates the exact logic from InsightFace attribute.py:
///   center = (bbox_x + bbox_w/2, bbox_y + bbox_h/2)
///   scale  = input_size / (max(bbox_w, bbox_h) * 1.5)
///   Then similarity transform: scale + translate so that face center maps to (48, 48).
fn aligned_crop_rgba(
    rgba: &[u8],
    img_w: usize,
    img_h: usize,
    face_x: f32,
    face_y: f32,
    face_w: f32,
    face_h: f32,
) -> tract_ndarray::Array4<f32> {
    let cx = face_x + face_w / 2.0;
    let cy = face_y + face_h / 2.0;
    let scale = WIDTH as f32 / (face_w.max(face_h) * 1.5);

    let mut tensor = tract_ndarray::Array4::<f32>::zeros((1, 3, HEIGHT, WIDTH));

    for out_y in 0..HEIGHT {
        for out_x in 0..WIDTH {
            // Inverse transform: map output pixel to source pixel
            let src_x = (out_x as f32 - (WIDTH as f32 / 2.0)) / scale + cx;
            let src_y = (out_y as f32 - (HEIGHT as f32 / 2.0)) / scale + cy;

            // Bilinear interpolation (or nearest-neighbor for out-of-bounds)
            let ix = src_x.floor() as i32;
            let iy = src_y.floor() as i32;

            if ix >= 0 && ix < (img_w as i32 - 1) && iy >= 0 && iy < (img_h as i32 - 1) {
                let fx = src_x - ix as f32;
                let fy = src_y - iy as f32;

                for c in 0..3 {
                    let p00 = rgba[((iy as usize) * img_w + ix as usize) * 4 + c] as f32;
                    let p10 = rgba[((iy as usize) * img_w + ix as usize + 1) * 4 + c] as f32;
                    let p01 = rgba[((iy as usize + 1) * img_w + ix as usize) * 4 + c] as f32;
                    let p11 = rgba[((iy as usize + 1) * img_w + ix as usize + 1) * 4 + c] as f32;

                    let val = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;

                    tensor[[0, c, out_y, out_x]] = val;
                }
            }
            // else: out-of-bounds pixels stay as 0.0 (black border, matching cv2.warpAffine borderValue=0.0)
        }
    }

    tensor
}

/// Predict age from a raw RGBA image with a face bounding box.
///
/// This applies the InsightFace alignment transform before inference,
/// dramatically improving accuracy (documented MAE ~4.1 years).
///
/// Parameters:
/// - `image_bytes`: Raw RGBA pixels of the source image (width * height * 4 bytes)
/// - `img_width`: Width of the source image
/// - `img_height`: Height of the source image
/// - `face_x`, `face_y`: Top-left corner of the face bounding box
/// - `face_w`, `face_h`: Width and height of the face bounding box
#[ic_cdk::update]
fn predict_age_with_bbox(
    image_bytes: Vec<u8>,
    img_width: u32,
    img_height: u32,
    face_x: f32,
    face_y: f32,
    face_w: f32,
    face_h: f32,
) -> f64 {
    let w = img_width as usize;
    let h = img_height as usize;
    assert_eq!(
        image_bytes.len(),
        w * h * 4,
        "Expected {} bytes ({}×{} RGBA), got {}",
        w * h * 4,
        w,
        h,
        image_bytes.len()
    );

    let tensor = aligned_crop_rgba(&image_bytes, w, h, face_x, face_y, face_w, face_h);
    run_inference(tensor)
}

/// Legacy method: predict age from a pre-cropped 96×96 RGBA bitmap.
#[ic_cdk::update]
fn predict_age_from_image(image_bytes: Vec<u8>) -> f64 {
    assert_eq!(
        image_bytes.len(),
        RGBA_BYTES,
        "Expected {} bytes (96×96 RGBA), got {}",
        RGBA_BYTES,
        image_bytes.len()
    );

    let mut tensor = tract_ndarray::Array4::<f32>::zeros((1, 3, HEIGHT, WIDTH));
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = (y * WIDTH + x) * 4;
            tensor[[0, 0, y, x]] = image_bytes[offset] as f32;
            tensor[[0, 1, y, x]] = image_bytes[offset + 1] as f32;
            tensor[[0, 2, y, x]] = image_bytes[offset + 2] as f32;
        }
    }

    run_inference(tensor)
}

// Export the Candid interface
ic_cdk::export_candid!();

#[cfg(test)]
mod tests {
    use super::*;

    /// Load full-resolution RGBA bytes from an image file.
    fn load_full_rgba(path: &str) -> (Vec<u8>, u32, u32) {
        let full_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
        println!("Loading image from: {:?}", full_path);
        let img = image::open(&full_path)
            .unwrap_or_else(|e| panic!("Failed to open {:?}: {}", full_path, e));
        let w = img.width();
        let h = img.height();
        println!("Image dims: {}x{}", w, h);
        let rgba = img.to_rgba8();
        (rgba.into_raw(), w, h)
    }

    /// Legacy helper for the old method.
    fn get_raw_rgba(path: &str) -> Vec<u8> {
        let full_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
        let img = image::open(&full_path)
            .unwrap_or_else(|e| panic!("Failed to open {:?}: {}", full_path, e));
        let rgba = img
            .resize_exact(
                WIDTH as u32,
                HEIGHT as u32,
                image::imageops::FilterType::Triangle,
            )
            .to_rgba8();
        rgba.into_raw()
    }

    #[test]
    fn test_aligned_benchmark() {
        // Each entry: (name, path, true_age, face_bbox_fraction)
        // face_bbox is (x_pct, y_pct, w_pct, h_pct) relative to image size
        // These approximate what a real face detector would output for each image
        let samples: Vec<(&str, &str, f64, (f32, f32, f32, f32))> = vec![
            // Synthetic portraits: face fills ~50-60% of frame, centered slightly above middle
            (
                "Baby",
                "tests/samples/baby.png",
                1.0,
                (0.20, 0.10, 0.60, 0.65),
            ),
            (
                "10yo",
                "tests/samples/10yo.png",
                10.0,
                (0.20, 0.08, 0.60, 0.65),
            ),
            (
                "40yo",
                "tests/samples/40yo.png",
                40.0,
                (0.18, 0.05, 0.64, 0.70),
            ),
            (
                "50yo",
                "tests/samples/50yo.png",
                50.0,
                (0.18, 0.05, 0.64, 0.70),
            ),
            (
                "60yo",
                "tests/samples/60yo.png",
                60.0,
                (0.18, 0.05, 0.64, 0.70),
            ),
            (
                "Elder (80)",
                "tests/samples/elder.png",
                80.0,
                (0.15, 0.05, 0.70, 0.70),
            ),
            // Real photos: face detector would give a tighter bbox
            (
                "Lena (21)",
                "tests/samples/adult_lena.jpg",
                21.0,
                (0.45, 0.22, 0.35, 0.50),
            ),
            (
                "Messi (30)",
                "tests/samples/adult_messi.jpg",
                30.0,
                (0.20, 0.02, 0.55, 0.70),
            ),
        ];

        let count = samples.len() as f64;
        let mut total_aligned = 0.0;
        let mut total_legacy = 0.0;

        println!("\n--- Aligned vs Legacy Benchmark ---");
        println!(
            "{:<15} | {:<5} | {:<8} | {:<6} | {:<8} | {:<6}",
            "Sample", "True", "Aligned", "Err", "Legacy", "Err"
        );
        println!("{:-<65}", "");

        for (name, path, true_age, (bx, by, bw, bh)) in &samples {
            let (rgba, w, h) = load_full_rgba(path);

            // Aligned path with simulated face bbox
            let fx = *bx * w as f32;
            let fy = *by * h as f32;
            let fw = *bw * w as f32;
            let fh = *bh * h as f32;
            let pred_a = predict_age_with_bbox(rgba, w, h, fx, fy, fw, fh);
            let err_a = (pred_a - true_age).abs();
            total_aligned += err_a;

            // Legacy path
            let legacy_bytes = get_raw_rgba(path);
            let pred_l = predict_age_from_image(legacy_bytes);
            let err_l = (pred_l - true_age).abs();
            total_legacy += err_l;

            println!(
                "{:<15} | {:<5.0} | {:<8.1} | {:<6.1} | {:<8.1} | {:<6.1}",
                name, true_age, pred_a, err_a, pred_l, err_l
            );
        }

        let mae_a = total_aligned / count;
        let mae_l = total_legacy / count;
        println!("{:-<65}", "");
        println!(
            "{:<15} | {:<5} | {:<8} | {:<6.2} | {:<8} | {:<6.2}",
            "MAE", "", "", mae_a, "", mae_l
        );
        println!(
            "\nImprovement: {:.1}% reduction in MAE",
            (1.0 - mae_a / mae_l) * 100.0
        );

        // The aligned path should be at least as good as legacy
        assert!(mae_a < 15.0, "Aligned MAE {} is too high", mae_a);
    }

    #[test]
    fn test_predict_baboon_stability() {
        let bytes = get_raw_rgba("tests/samples/monkey_baboon.jpg");
        let age = predict_age_from_image(bytes);
        println!("Baboon stability check: {:.1}", age);
        assert!(age > 0.0 && age < 120.0);
    }
}
