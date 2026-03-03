use getrandom::register_custom_getrandom;
use image::GenericImageView;
use std::io::Cursor;
use tract_onnx::prelude::*;

// Custom getrandom handler to allow compiling to wasm32-unknown-unknown
fn always_fail(_buf: &mut [u8]) -> Result<(), getrandom::Error> {
    Ok(())
}
register_custom_getrandom!(always_fail);

// Embed the ONNX model into the Wasm binary
const MODEL_BYTES: &[u8] = include_bytes!("../genderage.onnx");

/// Predict the user's age directly from a JPEG image drop sent from the frontend.
#[ic_cdk::update]
fn predict_age_from_image(image_bytes: Vec<u8>) -> f64 {
    // 1. Load the embedded model
    let mut cursor = Cursor::new(MODEL_BYTES);
    let model = tract_onnx::onnx()
        .model_for_read(&mut cursor)
        .expect("Failed to load ONNX model")
        .into_optimized()
        .expect("Failed to optimize model")
        .into_runnable()
        .expect("Failed to make model runnable");

    // 2. Decode the incoming image (JPEG)
    let img = image::load_from_memory(&image_bytes).expect("Failed to decode image");
    let img = img.resize_exact(96, 96, image::imageops::FilterType::Triangle);

    // 3. Convert the image to a Tract tensor
    // The InsightFace genderage model expects [1, 3, 96, 96] BGR or RGB float32
    // with normalization (pixel - 127.5) / 128.0
    let mut tensor = tract_ndarray::Array4::zeros((1, 3, 96, 96));
    for (x, y, pixel) in img.pixels() {
        // InsightFace generally expects BGR, but we can try RGB first. Let's map to BGR to be safe.
        tensor[[0, 0, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 128.0; // B
        tensor[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 128.0; // G
        tensor[[0, 2, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 128.0;
        // R
    }

    let input = tensor.into_tensor().into();

    // 4. Run inference
    let result = model.run(tvec!(input)).expect("Failed to run ML inference");

    // 5. Process backend result
    // The genderage model outputs [1, 3], where index 0 is Female, index 1 is Male, and index 2 is Age / 100.0
    let output_view = result[0]
        .to_array_view::<f32>()
        .expect("Failed to read output tensor");

    // The shape is [1, 3]. We access the 2nd index for age.
    let age_normalized = output_view[[0, 2]];

    // Convert back to actual age
    let predicted_age = age_normalized * 100.0;

    predicted_age as f64
}

// Export the Candid interface
ic_cdk::export_candid!();
