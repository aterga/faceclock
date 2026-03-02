use candid::CandidType;
use serde::Deserialize;

/// Data from a single captured frame
#[derive(CandidType, Deserialize, Clone, Debug)]
struct FrameData {
    age_estimate: f64,
    confidence: f64,
}

/// Result of the age prediction
#[derive(CandidType, Clone, Debug)]
struct AgeResult {
    predicted_age: f64,
    confidence: f64,
    frames_used: u32,
}

/// Predict the user's age from multiple frame observations.
///
/// Uses robust statistical aggregation:
/// 1. Filter out low-confidence frames (< 0.3)
/// 2. Reject outliers using IQR method
/// 3. Compute confidence-weighted mean of remaining estimates
/// 4. Report overall confidence based on agreement between frames
#[ic_cdk::update]
fn predict_age(frames: Vec<FrameData>) -> AgeResult {
    if frames.is_empty() {
        return AgeResult {
            predicted_age: 0.0,
            confidence: 0.0,
            frames_used: 0,
        };
    }

    // Step 1: Filter low-confidence frames
    let mut valid_frames: Vec<&FrameData> = frames
        .iter()
        .filter(|f| f.confidence >= 0.3 && f.age_estimate > 0.0 && f.age_estimate < 120.0)
        .collect();

    if valid_frames.is_empty() {
        return AgeResult {
            predicted_age: 0.0,
            confidence: 0.0,
            frames_used: 0,
        };
    }

    // Sort by age estimate for IQR computation
    valid_frames.sort_by(|a, b| a.age_estimate.partial_cmp(&b.age_estimate).unwrap());

    // Step 2: Outlier rejection using IQR if we have enough frames
    let filtered: Vec<&FrameData> = if valid_frames.len() >= 4 {
        let q1_idx = valid_frames.len() / 4;
        let q3_idx = (3 * valid_frames.len()) / 4;
        let q1 = valid_frames[q1_idx].age_estimate;
        let q3 = valid_frames[q3_idx].age_estimate;
        let iqr = q3 - q1;
        let lower = q1 - 1.5 * iqr;
        let upper = q3 + 1.5 * iqr;

        valid_frames
            .into_iter()
            .filter(|f| f.age_estimate >= lower && f.age_estimate <= upper)
            .collect()
    } else {
        valid_frames
    };

    if filtered.is_empty() {
        return AgeResult {
            predicted_age: 0.0,
            confidence: 0.0,
            frames_used: 0,
        };
    }

    // Step 3: Confidence-weighted mean
    let total_weight: f64 = filtered.iter().map(|f| f.confidence).sum();
    let weighted_age: f64 = filtered
        .iter()
        .map(|f| f.age_estimate * f.confidence)
        .sum::<f64>()
        / total_weight;

    // Step 4: Compute overall confidence based on frame agreement
    let mean_age = weighted_age;
    let variance: f64 = filtered
        .iter()
        .map(|f| {
            let diff = f.age_estimate - mean_age;
            diff * diff * f.confidence
        })
        .sum::<f64>()
        / total_weight;
    let std_dev = variance.sqrt();

    // Confidence: high when frames agree (low std_dev) and individual confidences are high
    // A std_dev of 0 gives confidence 1.0, std_dev of 10+ gives ~0.5
    let agreement_factor = 1.0 / (1.0 + std_dev / 5.0);
    let mean_confidence: f64 =
        filtered.iter().map(|f| f.confidence).sum::<f64>() / filtered.len() as f64;
    let overall_confidence = agreement_factor * mean_confidence;

    AgeResult {
        predicted_age: (weighted_age * 10.0).round() / 10.0, // Round to 1 decimal
        confidence: (overall_confidence * 100.0).round() / 100.0,
        frames_used: filtered.len() as u32,
    }
}

// Export the Candid interface
ic_cdk::export_candid!();
