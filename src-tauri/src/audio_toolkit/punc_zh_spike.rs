/// spike: verify sherpa-onnx 1.13 OfflinePunctuation API
///
/// Result: CONFIRMED AVAILABLE in sherpa-onnx 1.13.0
/// - OfflinePunctuation::create(&config) -> Option<OfflinePunctuation>
/// - add_punctuation(&str) -> Option<String>
/// - OfflinePunctuationConfig.model.ct_transformer: Option<String>
///
/// Model path: <models_dir>/sherpa-onnx-punct-ct-transformer-zh-cn-2024-04-12/model.onnx
/// Model URL: https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-cn-2024-04-12.tar.bz2
/// Approx size: ~50 MB
///
/// This file is kept as reference documentation but is NOT wired into mod.rs.
/// See punc_zh.rs for the production implementation.
#[allow(dead_code)]
fn _spike_main() {
    use sherpa_onnx::{OfflinePunctuation, OfflinePunctuationConfig};

    let model_dir = std::path::Path::new("./sherpa-onnx-punct-ct-transformer-zh-cn-2024-04-12");
    let onnx_path = model_dir.join("model.onnx");

    let mut config = OfflinePunctuationConfig::default();
    config.model.ct_transformer = Some(onnx_path.to_string_lossy().to_string());

    let punct = OfflinePunctuation::create(&config).expect("create punctuator");

    let text = "我今天去了超市买了很多东西包括苹果香蕉和葡萄还有一些蔬菜";
    let result = punct.add_punctuation(text).expect("punctuate");
    println!("Input:  {}", text);
    println!("Output: {}", result);
    // Expected: "我今天去了超市，买了很多东西，包括苹果、香蕉和葡萄，还有一些蔬菜。"
}
