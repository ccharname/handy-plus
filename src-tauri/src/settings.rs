use log::{debug, warn};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use specta::Type;
use std::collections::HashMap;
use std::fmt;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

pub const APPLE_INTELLIGENCE_PROVIDER_ID: &str = "apple_intelligence";
pub const APPLE_INTELLIGENCE_DEFAULT_MODEL_ID: &str = "Apple Intelligence";

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

// Custom deserializer to handle both old numeric format (1-5) and new string format ("trace", "debug", etc.)
impl<'de> Deserialize<'de> for LogLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LogLevelVisitor;

        impl<'de> Visitor<'de> for LogLevelVisitor {
            type Value = LogLevel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string or integer representing log level")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<LogLevel, E> {
                match value.to_lowercase().as_str() {
                    "trace" => Ok(LogLevel::Trace),
                    "debug" => Ok(LogLevel::Debug),
                    "info" => Ok(LogLevel::Info),
                    "warn" => Ok(LogLevel::Warn),
                    "error" => Ok(LogLevel::Error),
                    _ => Err(E::unknown_variant(
                        value,
                        &["trace", "debug", "info", "warn", "error"],
                    )),
                }
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<LogLevel, E> {
                match value {
                    1 => Ok(LogLevel::Trace),
                    2 => Ok(LogLevel::Debug),
                    3 => Ok(LogLevel::Info),
                    4 => Ok(LogLevel::Warn),
                    5 => Ok(LogLevel::Error),
                    _ => Err(E::invalid_value(de::Unexpected::Unsigned(value), &"1-5")),
                }
            }
        }

        deserializer.deserialize_any(LogLevelVisitor)
    }
}

impl From<LogLevel> for tauri_plugin_log::LogLevel {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tauri_plugin_log::LogLevel::Trace,
            LogLevel::Debug => tauri_plugin_log::LogLevel::Debug,
            LogLevel::Info => tauri_plugin_log::LogLevel::Info,
            LogLevel::Warn => tauri_plugin_log::LogLevel::Warn,
            LogLevel::Error => tauri_plugin_log::LogLevel::Error,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct ShortcutBinding {
    pub id: String,
    pub name: String,
    pub description: String,
    pub default_binding: String,
    pub current_binding: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct LLMPrompt {
    pub id: String,
    pub name: String,
    pub prompt: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct PostProcessProvider {
    pub id: String,
    pub label: String,
    pub base_url: String,
    #[serde(default)]
    pub allow_base_url_edit: bool,
    #[serde(default)]
    pub models_endpoint: Option<String>,
    #[serde(default)]
    pub supports_structured_output: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum OverlayPosition {
    None,
    Top,
    Bottom,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum ModelUnloadTimeout {
    Never,
    Immediately,
    Min2,
    #[default]
    Min5,
    Min10,
    Min15,
    Hour1,
    Sec15, // Debug mode only
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum PasteMethod {
    CtrlV,
    Direct,
    None,
    ShiftInsert,
    CtrlShiftV,
    ExternalScript,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum ClipboardHandling {
    #[default]
    DontModify,
    CopyToClipboard,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum AutoSubmitKey {
    #[default]
    Enter,
    CtrlEnter,
    CmdEnter,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum RecordingRetentionPeriod {
    Never,
    PreserveLimit,
    Days3,
    Weeks2,
    Months3,
}

impl Default for PasteMethod {
    fn default() -> Self {
        // Default to CtrlV for macOS and Windows, Direct for Linux
        #[cfg(target_os = "linux")]
        return PasteMethod::Direct;
        #[cfg(not(target_os = "linux"))]
        return PasteMethod::CtrlV;
    }
}

impl ModelUnloadTimeout {
    pub fn to_minutes(self) -> Option<u64> {
        match self {
            ModelUnloadTimeout::Never => None,
            ModelUnloadTimeout::Immediately => Some(0), // Special case for immediate unloading
            ModelUnloadTimeout::Min2 => Some(2),
            ModelUnloadTimeout::Min5 => Some(5),
            ModelUnloadTimeout::Min10 => Some(10),
            ModelUnloadTimeout::Min15 => Some(15),
            ModelUnloadTimeout::Hour1 => Some(60),
            ModelUnloadTimeout::Sec15 => Some(0), // Special case for debug - handled separately
        }
    }

    pub fn to_seconds(self) -> Option<u64> {
        match self {
            ModelUnloadTimeout::Never => None,
            ModelUnloadTimeout::Immediately => Some(0), // Special case for immediate unloading
            ModelUnloadTimeout::Sec15 => Some(15),
            _ => self.to_minutes().map(|m| m * 60),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum SoundTheme {
    Marimba,
    Pop,
    Custom,
}

impl SoundTheme {
    fn as_str(&self) -> &'static str {
        match self {
            SoundTheme::Marimba => "marimba",
            SoundTheme::Pop => "pop",
            SoundTheme::Custom => "custom",
        }
    }

    pub fn to_start_path(self) -> String {
        format!("resources/{}_start.wav", self.as_str())
    }

    pub fn to_stop_path(self) -> String {
        format!("resources/{}_stop.wav", self.as_str())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum TypingTool {
    #[default]
    Auto,
    Wtype,
    Kwtype,
    Dotool,
    Ydotool,
    Xdotool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Default, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum OrtAcceleratorSetting {
    #[default]
    Auto,
    Cpu,
    Cuda,
    #[serde(rename = "directml")]
    DirectMl,
    Rocm,
}

#[derive(Clone, Serialize, Deserialize, Type)]
#[serde(transparent)]
pub(crate) struct SecretMap(HashMap<String, String>);

impl fmt::Debug for SecretMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let redacted: HashMap<&String, &str> = self
            .0
            .iter()
            .map(|(k, v)| (k, if v.is_empty() { "" } else { "[REDACTED]" }))
            .collect();
        redacted.fmt(f)
    }
}

impl std::ops::Deref for SecretMap {
    type Target = HashMap<String, String>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SecretMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Power Mode: App-aware profiles
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, Type, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProfileMatcher {
    /// macOS bundle id (exact or wildcard suffix '*'); on Windows/Linux falls back to process name match
    BundleId { value: String },
    /// case-insensitive process executable name (without extension)
    ProcessName { value: String },
    /// case-insensitive substring match against window title
    WindowTitleSubstring { value: String },
    #[default]
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct AppProfile {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub matchers: Vec<ProfileMatcher>,
    pub selected_language: Option<String>,
    #[serde(default)]
    pub custom_words_extra: Vec<String>,
    pub post_process_provider_id: Option<String>,
    pub post_process_selected_prompt_id: Option<String>,
    pub paste_method: Option<PasteMethod>,
    pub append_trailing_space: Option<bool>,
    pub auto_submit: Option<bool>,
    /// When set, transcription for this profile uses the given model id
    /// instead of the global `selected_model`. Honoured only when
    /// `profile_hot_swap_engine` is enabled in AppSettings.
    #[serde(default)]
    pub selected_model: Option<String>,
    /// Override active_preset_id when profile matches.  None = inherit global.
    /// This is UI-only — the pipeline uses the resolved model/language/punc
    /// from other fields; this just reflects which preset label to show.
    #[serde(default)]
    pub active_preset_id: Option<String>,
    /// Override post_process_chain.  None = inherit global; Some(empty vec) = disable chain.
    #[serde(default)]
    pub post_process_chain: Option<Vec<String>>,
}

fn default_power_mode_enabled() -> bool {
    // Handy+ default: opt users into Power Mode — bundled profiles ship disabled
    // individually, so this only flips the master switch on; no behavior change
    // until the user enables a specific profile.
    true
}

pub fn default_app_profiles() -> Vec<AppProfile> {
    vec![
        AppProfile {
            id: "builtin_code".to_string(),
            name: "Code".to_string(),
            enabled: true,
            matchers: vec![
                ProfileMatcher::BundleId {
                    value: "com.microsoft.VSCode".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.todesktop.230313mzl4w4u92".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.googlecode.iterm2".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.apple.Terminal".to_string(),
                },
                ProfileMatcher::ProcessName {
                    value: "Code".to_string(),
                },
                ProfileMatcher::ProcessName {
                    value: "Cursor".to_string(),
                },
            ],
            selected_language: None,
            custom_words_extra: vec![],
            post_process_provider_id: None,
            post_process_selected_prompt_id: None,
            paste_method: Some(PasteMethod::Direct),
            append_trailing_space: Some(false),
            auto_submit: Some(false),
            selected_model: None,
            active_preset_id: None,
            post_process_chain: None,
        },
        AppProfile {
            id: "builtin_chat".to_string(),
            name: "Chat".to_string(),
            enabled: true,
            matchers: vec![
                ProfileMatcher::BundleId {
                    value: "com.tencent.xinWeChat".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.apple.MobileSMS".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.tinyspeck.slackmacgap".to_string(),
                },
            ],
            selected_language: None,
            custom_words_extra: vec![],
            post_process_provider_id: None,
            post_process_selected_prompt_id: None,
            paste_method: None,
            append_trailing_space: Some(true),
            auto_submit: Some(false),
            selected_model: None,
            active_preset_id: None,
            post_process_chain: None,
        },
        AppProfile {
            id: "builtin_writing".to_string(),
            name: "Writing".to_string(),
            enabled: true,
            matchers: vec![
                ProfileMatcher::BundleId {
                    value: "md.obsidian".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.apple.mail".to_string(),
                },
                ProfileMatcher::BundleId {
                    value: "com.microsoft.Word".to_string(),
                },
            ],
            selected_language: None,
            custom_words_extra: vec![],
            post_process_provider_id: None,
            post_process_selected_prompt_id: None,
            paste_method: None,
            append_trailing_space: Some(true),
            auto_submit: Some(false),
            selected_model: None,
            active_preset_id: None,
            post_process_chain: None,
        },
        AppProfile {
            id: "builtin_claude_code".to_string(),
            name: "Claude Code (terminal AI)".to_string(),
            enabled: true,
            matchers: vec![ProfileMatcher::WindowTitleSubstring {
                value: "claude".to_string(),
            }],
            selected_language: None,
            custom_words_extra: vec![],
            post_process_provider_id: None,
            post_process_selected_prompt_id: None,
            paste_method: Some(PasteMethod::Direct),
            append_trailing_space: Some(false),
            auto_submit: Some(true),
            selected_model: None,
            active_preset_id: None,
            post_process_chain: None,
        },
        AppProfile {
            id: "builtin_default_fallback".to_string(),
            name: "Default fallback (sample)".to_string(),
            enabled: false,
            matchers: vec![],
            selected_language: None,
            custom_words_extra: vec![],
            post_process_provider_id: None,
            post_process_selected_prompt_id: None,
            paste_method: None,
            append_trailing_space: None,
            auto_submit: None,
            selected_model: None,
            active_preset_id: None,
            post_process_chain: None,
        },
    ]
}

/* still handy for composing the initial JSON in the store ------------- */
#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct AppSettings {
    pub bindings: HashMap<String, ShortcutBinding>,
    pub push_to_talk: bool,
    pub audio_feedback: bool,
    #[serde(default = "default_audio_feedback_volume")]
    pub audio_feedback_volume: f32,
    #[serde(default = "default_sound_theme")]
    pub sound_theme: SoundTheme,
    #[serde(default = "default_start_hidden")]
    pub start_hidden: bool,
    #[serde(default = "default_autostart_enabled")]
    pub autostart_enabled: bool,
    #[serde(default = "default_update_checks_enabled")]
    pub update_checks_enabled: bool,
    #[serde(default = "default_model")]
    pub selected_model: String,
    #[serde(default = "default_always_on_microphone")]
    pub always_on_microphone: bool,
    #[serde(default)]
    pub selected_microphone: Option<String>,
    #[serde(default)]
    pub clamshell_microphone: Option<String>,
    #[serde(default)]
    pub selected_output_device: Option<String>,
    #[serde(default = "default_translate_to_english")]
    pub translate_to_english: bool,
    #[serde(default = "default_selected_language")]
    pub selected_language: String,
    #[serde(default = "default_overlay_position")]
    pub overlay_position: OverlayPosition,
    #[serde(default = "default_debug_mode")]
    pub debug_mode: bool,
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,
    #[serde(default)]
    pub custom_words: Vec<String>,
    #[serde(default)]
    pub model_unload_timeout: ModelUnloadTimeout,
    #[serde(default = "default_word_correction_threshold")]
    pub word_correction_threshold: f64,
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
    #[serde(default = "default_recording_retention_period")]
    pub recording_retention_period: RecordingRetentionPeriod,
    #[serde(default)]
    pub paste_method: PasteMethod,
    #[serde(default)]
    pub clipboard_handling: ClipboardHandling,
    #[serde(default = "default_auto_submit")]
    pub auto_submit: bool,
    #[serde(default)]
    pub auto_submit_key: AutoSubmitKey,
    #[serde(default = "default_post_process_enabled")]
    pub post_process_enabled: bool,
    #[serde(default = "default_post_process_provider_id")]
    pub post_process_provider_id: String,
    #[serde(default = "default_post_process_providers")]
    pub post_process_providers: Vec<PostProcessProvider>,
    #[serde(default = "default_post_process_api_keys")]
    pub post_process_api_keys: SecretMap,
    #[serde(default = "default_post_process_models")]
    pub post_process_models: HashMap<String, String>,
    #[serde(default = "default_post_process_prompts")]
    pub post_process_prompts: Vec<LLMPrompt>,
    #[serde(default)]
    pub post_process_selected_prompt_id: Option<String>,
    #[serde(default)]
    pub mute_while_recording: bool,
    #[serde(default)]
    pub append_trailing_space: bool,
    #[serde(default = "default_app_language")]
    pub app_language: String,
    #[serde(default = "default_show_tray_icon")]
    pub show_tray_icon: bool,
    #[serde(default = "default_paste_delay_ms")]
    pub paste_delay_ms: u64,
    #[serde(default = "default_typing_tool")]
    pub typing_tool: TypingTool,
    pub external_script_path: Option<String>,
    #[serde(default)]
    pub custom_filler_words: Option<Vec<String>>,
    #[serde(default)]
    pub ort_accelerator: OrtAcceleratorSetting,
    #[serde(default)]
    pub extra_recording_buffer_ms: u64,
    #[serde(default = "default_power_mode_enabled")]
    pub power_mode_enabled: bool,
    #[serde(default = "default_app_profiles")]
    pub app_profiles: Vec<AppProfile>,
    /// When true, the transcription pipeline honours each profile's
    /// `selected_model` override and may hot-swap the loaded engine on
    /// stop. Off by default — model swap costs 1-3s of latency, so users
    /// must opt in.
    #[serde(default)]
    pub profile_hot_swap_engine: bool,
    /// When true (default), Apple Speech first attempts on-device recognition.
    /// If on-device recognition is unavailable (e.g. the language's dictation model
    /// hasn't been downloaded in System Settings), it automatically retries with
    /// network-based recognition. Set to false to skip the on-device attempt entirely.
    #[serde(default = "default_apple_speech_require_on_device")]
    pub apple_speech_require_on_device: bool,
    /// Directory path where diary entries are written (e.g. ~/obsidian/diary).
    /// None means the diary archival feature is disabled.
    #[serde(default)]
    pub diary_dir: Option<String>,
    /// Keywords that trigger diary archival when the transcription starts with one.
    /// Matching is case-insensitive and handles common CJK/Latin punctuation after the keyword.
    #[serde(default = "default_diary_keywords")]
    pub diary_keywords: Vec<String>,
    /// Ordered list of prompt IDs to run sequentially after transcription.
    /// When `Some(vec)` and non-empty, each step's output feeds the next.
    /// When `None` or empty, falls back to `post_process_selected_prompt_id` (backward-compat).
    #[serde(default)]
    pub post_process_chain: Option<Vec<String>>,
    /// Hotwords bias score used when loading the sherpa-onnx SenseVoice path
    /// (sense-voice-small-sherpa model). Maps to OfflineRecognizerConfig.hotwords_score.
    /// Range: 0.5 – 5.0. Default: 2.0. Has no effect on the transcribe-rs SenseVoice path.
    #[serde(default = "default_hotwords_boost")]
    pub hotwords_boost: f32,
    /// ID of the currently active ASR preset, or None when the user has detached
    /// (i.e. manually changed one or more settings after applying a preset).
    #[serde(default)]
    pub active_preset_id: Option<String>,
    /// Tracks which one-time migrations have been applied. Keys are migration
    /// identifiers (e.g. "v_0_8_10_enable_builtins"); value=true means the
    /// migration ran. Missing key = not applied yet.
    #[serde(default)]
    pub migration_applied: HashMap<String, bool>,
    /// Phonetic transliteration aliases — exact substring substitution to recover
    /// proper nouns the ASR mangles into Chinese phonetic approximations.
    /// Format: { "Anthropic": ["aobic", "an thro pic"], "Obsidian": ["op店"] }
    /// Applied AFTER apply_custom_words (fuzzy) and BEFORE filter/punc,
    /// so corrections feed the punctuation model.
    #[serde(default)]
    pub custom_word_aliases: HashMap<String, Vec<String>>,
    /// When true (default), Apple Speech partials are progressively pasted into
    /// the active application as the user speaks — matching the behaviour of
    /// Wispr Flow / macOS native dictation.  Set to false to revert to the
    /// original batch-paste-after-completion behaviour.
    #[serde(default = "default_apple_speech_incremental_paste")]
    pub apple_speech_incremental_paste: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// ASR Presets
// ────────────────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct AsrPreset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub model_id: String,
    pub language: String,
    pub require_post_process_chain: Option<Vec<String>>,
    #[serde(default)]
    pub require_apple_speech_on_device: Option<bool>,
    #[serde(default)]
    pub builtin: bool,
}

pub fn default_asr_presets() -> Vec<AsrPreset> {
    vec![
        AsrPreset {
            id: "chinese_balanced".to_string(),
            name: "Chinese Balanced".to_string(),
            description: "SenseVoice — Chinese-balanced offline".to_string(),
            icon: "🇨🇳".to_string(),
            // NOTE: kept on transcribe-rs sense-voice-int8 path. The sherpa
            // sense-voice-small variant would honour hotwords_file but
            // OfflineRecognizer::create() silently returns None when our
            // 60+ entry hotwords file is loaded with modeling_unit=cjkchar+bpe
            // (root cause TBD — sherpa-onnx C layer gives no error message).
            // The L1 alias layer (custom_word_aliases) covers proper-noun
            // recovery in the meantime.
            model_id: "sense-voice-int8".to_string(),
            language: "zh-Hans".to_string(),
            require_post_process_chain: None,
            require_apple_speech_on_device: None,
            builtin: true,
        },
        // Qwen3-ASR-0.6B via mlx-audio-swift (Apple Silicon macOS only).
        // Promoted to builtin 2026-05-04 (M1): p50 1.2s, 粤语/沪语/闽南方言全覆盖。
        // Gate: only shown on macOS aarch64 via visible_preset_ids_from_settings().
        AsrPreset {
            id: "qwen3_mlx".to_string(),
            name: "Qwen3-ASR Realtime".to_string(),
            description:
                "~600 MB MLX 量化 / p50 1.2s / 粤语+沪语+闽南方言识别 / 仅 Apple Silicon。"
                    .to_string(),
            icon: "🪶".to_string(),
            model_id: "qwen3-asr-mlx-8bit".to_string(),
            language: "zh-Hans".to_string(),
            require_post_process_chain: None,
            require_apple_speech_on_device: None,
            builtin: true,
        },
    ]
}

fn default_model() -> String {
    "".to_string()
}

fn default_apple_speech_require_on_device() -> bool {
    // Default: prefer on-device for privacy. The transcription path will automatically
    // retry with network recognition if the on-device model is unavailable.
    true
}

fn default_always_on_microphone() -> bool {
    false
}

fn default_translate_to_english() -> bool {
    false
}

fn default_start_hidden() -> bool {
    false
}

fn default_autostart_enabled() -> bool {
    false
}

fn default_update_checks_enabled() -> bool {
    true
}

fn default_selected_language() -> String {
    "auto".to_string()
}

fn default_overlay_position() -> OverlayPosition {
    #[cfg(target_os = "linux")]
    return OverlayPosition::None;
    #[cfg(not(target_os = "linux"))]
    return OverlayPosition::Bottom;
}

fn default_debug_mode() -> bool {
    false
}

fn default_log_level() -> LogLevel {
    LogLevel::Debug
}

fn default_word_correction_threshold() -> f64 {
    0.18
}

fn default_paste_delay_ms() -> u64 {
    60
}

fn default_auto_submit() -> bool {
    false
}

fn default_history_limit() -> usize {
    5
}

fn default_recording_retention_period() -> RecordingRetentionPeriod {
    RecordingRetentionPeriod::PreserveLimit
}

fn default_audio_feedback_volume() -> f32 {
    1.0
}

fn default_sound_theme() -> SoundTheme {
    SoundTheme::Marimba
}

fn default_post_process_enabled() -> bool {
    // Handy+ default: on macOS, Apple Intelligence post-processing is on-device,
    // free, and ships with the OS — enable by default. Other platforms keep it off
    // until the user adds an API key.
    cfg!(target_os = "macos")
}

fn default_app_language() -> String {
    tauri_plugin_os::locale()
        .map(|l| l.replace('_', "-"))
        .unwrap_or_else(|| "en".to_string())
}

fn default_show_tray_icon() -> bool {
    true
}

fn default_post_process_provider_id() -> String {
    // Handy+ default: prefer on-device Apple Intelligence on macOS so users get a
    // working post-process pipeline without entering an API key. Falls back to
    // OpenAI elsewhere (still requires user key, matching upstream behavior).
    #[cfg(target_os = "macos")]
    {
        APPLE_INTELLIGENCE_PROVIDER_ID.to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "openai".to_string()
    }
}

fn default_post_process_providers() -> Vec<PostProcessProvider> {
    let mut providers = vec![
        PostProcessProvider {
            id: "openai".to_string(),
            label: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "zai".to_string(),
            label: "Z.AI".to_string(),
            base_url: "https://api.z.ai/api/paas/v4".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "openrouter".to_string(),
            label: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "anthropic".to_string(),
            label: "Anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        },
        PostProcessProvider {
            id: "groq".to_string(),
            label: "Groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        },
        PostProcessProvider {
            id: "cerebras".to_string(),
            label: "Cerebras".to_string(),
            base_url: "https://api.cerebras.ai/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
    ];

    // Note: We always include Apple Intelligence on macOS ARM64 without checking availability
    // at startup. The availability check is deferred to when the user actually tries to use it
    // (in actions.rs). This prevents crashes on macOS 26.x beta where accessing
    // SystemLanguageModel.default during early app initialization causes SIGABRT.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        providers.push(PostProcessProvider {
            id: APPLE_INTELLIGENCE_PROVIDER_ID.to_string(),
            label: "Apple Intelligence".to_string(),
            base_url: "apple-intelligence://local".to_string(),
            allow_base_url_edit: false,
            models_endpoint: None,
            supports_structured_output: true,
        });
    }

    // AWS Bedrock via Mantle (OpenAI-compatible endpoint)
    providers.push(PostProcessProvider {
        id: "bedrock_mantle".to_string(),
        label: "AWS Bedrock (Mantle)".to_string(),
        base_url: "https://bedrock-mantle.us-east-1.api.aws/v1".to_string(),
        allow_base_url_edit: false,
        models_endpoint: Some("/models".to_string()),
        supports_structured_output: true,
    });

    // Custom provider always comes last
    providers.push(PostProcessProvider {
        id: "custom".to_string(),
        label: "Custom".to_string(),
        base_url: "http://localhost:11434/v1".to_string(),
        allow_base_url_edit: true,
        models_endpoint: Some("/models".to_string()),
        supports_structured_output: false,
    });

    providers
}

fn default_post_process_api_keys() -> SecretMap {
    let mut map = HashMap::new();
    for provider in default_post_process_providers() {
        map.insert(provider.id, String::new());
    }
    SecretMap(map)
}

fn default_model_for_provider(provider_id: &str) -> String {
    if provider_id == APPLE_INTELLIGENCE_PROVIDER_ID {
        return APPLE_INTELLIGENCE_DEFAULT_MODEL_ID.to_string();
    }
    String::new()
}

fn default_post_process_models() -> HashMap<String, String> {
    let mut map = HashMap::new();
    for provider in default_post_process_providers() {
        map.insert(
            provider.id.clone(),
            default_model_for_provider(&provider.id),
        );
    }
    map
}

fn default_post_process_prompts() -> Vec<LLMPrompt> {
    vec![LLMPrompt {
        id: "default_improve_transcriptions".to_string(),
        name: "Improve Transcriptions".to_string(),
        prompt: "Clean this transcript:\n1. Fix spelling, capitalization, and punctuation errors\n2. Convert number words to digits (twenty-five → 25, ten percent → 10%, five dollars → $5)\n3. Replace spoken punctuation with symbols (period → ., comma → ,, question mark → ?)\n4. Remove filler words (um, uh, like as filler)\n5. Keep the language in the original version (if it was french, keep it in french for example)\n\nPreserve exact meaning and word order. Do not paraphrase or reorder content.\n\nReturn only the cleaned transcript.\n\nTranscript:\n${output}".to_string(),
    }]
}

fn default_diary_keywords() -> Vec<String> {
    vec![
        "日记".to_string(),
        "备忘".to_string(),
        "diary".to_string(),
        "memo".to_string(),
    ]
}

fn default_typing_tool() -> TypingTool {
    TypingTool::Auto
}

fn default_hotwords_boost() -> f32 {
    2.0
}

fn default_apple_speech_incremental_paste() -> bool {
    true
}

fn ensure_post_process_defaults(settings: &mut AppSettings) -> bool {
    let mut changed = false;
    for provider in default_post_process_providers() {
        // Use match to do a single lookup - either sync existing or add new
        match settings
            .post_process_providers
            .iter_mut()
            .find(|p| p.id == provider.id)
        {
            Some(existing) => {
                // Sync supports_structured_output field for existing providers (migration)
                if existing.supports_structured_output != provider.supports_structured_output {
                    debug!(
                        "Updating supports_structured_output for provider '{}' from {} to {}",
                        provider.id,
                        existing.supports_structured_output,
                        provider.supports_structured_output
                    );
                    existing.supports_structured_output = provider.supports_structured_output;
                    changed = true;
                }
            }
            None => {
                // Provider doesn't exist, add it
                settings.post_process_providers.push(provider.clone());
                changed = true;
            }
        }

        if !settings.post_process_api_keys.contains_key(&provider.id) {
            settings
                .post_process_api_keys
                .insert(provider.id.clone(), String::new());
            changed = true;
        }

        let default_model = default_model_for_provider(&provider.id);
        match settings.post_process_models.get_mut(&provider.id) {
            Some(existing) => {
                if existing.is_empty() && !default_model.is_empty() {
                    *existing = default_model.clone();
                    changed = true;
                }
            }
            None => {
                settings
                    .post_process_models
                    .insert(provider.id.clone(), default_model);
                changed = true;
            }
        }
    }

    changed
}

/// One-time migration: v0.8.10 made builtin_chat/writing/code/claude_code
/// enabled-by-default, but existing users have these stuck at false from
/// before. Force-enable them once if the user has never explicitly toggled.
const MIGRATION_V_0_8_10_ENABLE_BUILTINS: &str = "v_0_8_10_enable_builtins";

pub fn ensure_v_0_8_10_enable_builtins_migration(settings: &mut AppSettings) -> bool {
    if settings
        .migration_applied
        .get(MIGRATION_V_0_8_10_ENABLE_BUILTINS)
        .copied()
        .unwrap_or(false)
    {
        return false; // already applied
    }

    const TARGETS: &[&str] = &[
        "builtin_code",
        "builtin_chat",
        "builtin_writing",
        "builtin_claude_code",
    ];

    let mut changed = false;
    for profile in settings.app_profiles.iter_mut() {
        if TARGETS.contains(&profile.id.as_str()) && !profile.enabled {
            profile.enabled = true;
            changed = true;
            log::info!(
                "[migration v0.8.10] force-enabled builtin profile '{}'",
                profile.id
            );
        }
    }

    settings
        .migration_applied
        .insert(MIGRATION_V_0_8_10_ENABLE_BUILTINS.to_string(), true);
    // Always return true so migration_applied flag itself gets persisted
    let _ = changed;
    true
}

/// Ensure app_profiles is populated for users upgrading from a version before Power Mode.
pub fn ensure_app_profiles_defaults(settings: &mut AppSettings) -> bool {
    if settings.app_profiles.is_empty() {
        debug!("app_profiles is empty; populating with built-in templates");
        settings.app_profiles = default_app_profiles();
        true
    } else {
        false
    }
}

/// Returns the set of preset IDs that are visible to the user given the current
/// settings, mirroring the logic in `commands::asr_presets::filtered_asr_presets`
/// but operating on an already-loaded `AppSettings` (no `AppHandle` needed, so
/// it is safe to call before or inside `get_settings`).
pub fn visible_preset_ids_from_settings(_settings: &AppSettings) -> Vec<String> {
    let mut presets = default_asr_presets();
    // qwen3_mlx requires macOS aarch64 (mlx-audio-swift). Hide on Intel/Linux/Win
    // so the v_0_8_17 migration treats stored qwen3_mlx as hidden → falls back to
    // chinese_balanced on non-Apple-Silicon platforms.
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        presets.retain(|p| p.id != "qwen3_mlx");
    }
    presets.into_iter().map(|p| p.id).collect()
}

/// One-time migration: if the stored `active_preset_id` is no longer present in
/// the visible preset list (e.g. `apple_native` on macOS 26+ where
/// SFSpeechRecognizer internally routes through SpeechAnalyzer and crashes),
/// reset the preset fields to the `chinese_balanced` default so the user does
/// not end up with a broken state on first transcription attempt.
///
/// `visible_preset_ids` is injected so the core logic stays testable without an
/// `AppHandle` (callers supply `visible_preset_ids_from_settings(&settings)` in
/// production and a mock slice in tests).
const MIGRATION_V_0_8_15_DROP_HIDDEN_PRESET: &str = "v_0_8_15_drop_hidden_active_preset";

pub fn ensure_v_0_8_15_drop_hidden_preset_migration(
    settings: &mut AppSettings,
    visible_preset_ids: &[&str],
) -> bool {
    if settings
        .migration_applied
        .get(MIGRATION_V_0_8_15_DROP_HIDDEN_PRESET)
        .copied()
        .unwrap_or(false)
    {
        return false; // already applied
    }

    let needs_reset = settings
        .active_preset_id
        .as_deref()
        .map(|id| !visible_preset_ids.contains(&id))
        .unwrap_or(false);

    if needs_reset {
        let old_id = settings
            .active_preset_id
            .as_deref()
            .unwrap_or("<none>")
            .to_string();

        // Find the chinese_balanced preset definition and mirror apply_asr_preset
        // (without calling load_model — TranscriptionManager is not yet started).
        if let Some(preset) = default_asr_presets()
            .into_iter()
            .find(|p| p.id == "chinese_balanced")
        {
            settings.selected_language = preset.language.clone();
            if let Some(chain) = preset.require_post_process_chain.clone() {
                settings.post_process_chain = Some(chain);
            }
            if let Some(on_device) = preset.require_apple_speech_on_device {
                settings.apple_speech_require_on_device = on_device;
            }
            settings.active_preset_id = Some(preset.id.clone());
            settings.selected_model = preset.model_id.clone();

            log::info!(
                "Migration {}: stored preset '{}' is now hidden — reset to chinese_balanced",
                MIGRATION_V_0_8_15_DROP_HIDDEN_PRESET,
                old_id,
            );
        }
    }

    settings
        .migration_applied
        .insert(MIGRATION_V_0_8_15_DROP_HIDDEN_PRESET.to_string(), true);
    true
}

/// v_0_8_16 — re-runs the same hidden-preset reset logic for users who already
/// applied v_0_8_15 (so the flag is set) but ended up on a preset that becomes
/// hidden in a later release. Necessary because the v_0_8_15 flag is checked
/// before the visible-set diff, so once-set never re-triggers.
///
/// Concrete trigger: 2026-05-04 hid `experimental_qwen3` (sherpa-onnx Qwen3
/// deprecated by the MLX preset on every axis). Users with stored
/// `active_preset_id = "experimental_qwen3"` and `migration_applied[v_0_8_15] = true`
/// would otherwise stay on the now-invisible preset forever.
const MIGRATION_V_0_8_16_DROP_HIDDEN_PRESET: &str = "v_0_8_16_drop_hidden_active_preset";

pub fn ensure_v_0_8_16_drop_hidden_preset_migration(
    settings: &mut AppSettings,
    visible_preset_ids: &[&str],
) -> bool {
    if settings
        .migration_applied
        .get(MIGRATION_V_0_8_16_DROP_HIDDEN_PRESET)
        .copied()
        .unwrap_or(false)
    {
        return false;
    }

    let needs_reset = settings
        .active_preset_id
        .as_deref()
        .map(|id| !visible_preset_ids.contains(&id))
        .unwrap_or(false);

    if needs_reset {
        let old_id = settings
            .active_preset_id
            .as_deref()
            .unwrap_or("<none>")
            .to_string();

        if let Some(preset) = default_asr_presets()
            .into_iter()
            .find(|p| p.id == "chinese_balanced")
        {
            settings.selected_language = preset.language.clone();
            if let Some(chain) = preset.require_post_process_chain.clone() {
                settings.post_process_chain = Some(chain);
            }
            if let Some(on_device) = preset.require_apple_speech_on_device {
                settings.apple_speech_require_on_device = on_device;
            }
            settings.active_preset_id = Some(preset.id.clone());
            settings.selected_model = preset.model_id.clone();

            log::info!(
                "Migration {}: stored preset '{}' is now hidden — reset to chinese_balanced",
                MIGRATION_V_0_8_16_DROP_HIDDEN_PRESET,
                old_id,
            );
        }
    }

    settings
        .migration_applied
        .insert(MIGRATION_V_0_8_16_DROP_HIDDEN_PRESET.to_string(), true);
    true
}

/// v_0_8_17 — M1 preset registry collapse. Physical removal of 4 legacy presets
/// (apple_native / experimental_voxtral / experimental_qwen3 / multilingual_offline)
/// and rename of experimental_qwen3_mlx → qwen3_mlx (promoted to builtin). // migration:
///
/// Migration rules:
///   - stored id == "experimental_qwen3_mlx" → rewrite to "qwen3_mlx" (rename) // migration:
///   - stored id ∈ {apple_native, experimental_voxtral, experimental_qwen3,
///     multilingual_offline, qwen3_mlx (non-aarch64)} → reset to chinese_balanced
///   - stored id == "chinese_balanced" → no-op
///   - stored id == None → no-op (user never applied a preset)
///
/// Unlike v_0_8_15/16 this migration is NOT gated on visible_preset_ids — the
/// source-of-truth is now the registry itself, not runtime visibility.
const MIGRATION_V_0_8_17_COLLAPSE: &str = "v_0_8_17_collapse_to_two_presets";

pub fn ensure_v_0_8_17_collapse_migration(settings: &mut AppSettings) -> bool {
    if settings
        .migration_applied
        .get(MIGRATION_V_0_8_17_COLLAPSE)
        .copied()
        .unwrap_or(false)
    {
        return false; // already applied
    }

    let valid_ids = ["chinese_balanced", "qwen3_mlx"];
    // On non-aarch64 platforms qwen3_mlx is not runnable; treat as removed.
    let aarch64_macos = cfg!(all(target_os = "macos", target_arch = "aarch64"));

    let changed = match settings.active_preset_id.as_deref() {
        None => false,
        // migration: Rename: experimental_qwen3_mlx → qwen3_mlx
        Some("experimental_qwen3_mlx") => {
            // migration:
            if aarch64_macos {
                log::info!(
                    "Migration {}: renamed experimental_qwen3_mlx → qwen3_mlx", // migration:
                    MIGRATION_V_0_8_17_COLLAPSE
                );
                settings.active_preset_id = Some("qwen3_mlx".to_string());
                // model_id is already "qwen3-asr-mlx-8bit" — no change needed
                true
            } else {
                // Non-Apple-Silicon: qwen3_mlx not runnable, fall through to reset
                log::info!(
                    "Migration {}: experimental_qwen3_mlx on non-aarch64 → reset to chinese_balanced", // migration:
                    MIGRATION_V_0_8_17_COLLAPSE
                );
                reset_to_chinese_balanced(settings);
                true
            }
        }
        Some(id) if valid_ids.contains(&id) && (id != "qwen3_mlx" || aarch64_macos) => {
            // Already on a valid preset for this platform
            false
        }
        Some(old_id) => {
            log::info!(
                "Migration {}: stored preset '{}' removed — reset to chinese_balanced",
                MIGRATION_V_0_8_17_COLLAPSE,
                old_id
            );
            reset_to_chinese_balanced(settings);
            true
        }
    };

    settings
        .migration_applied
        .insert(MIGRATION_V_0_8_17_COLLAPSE.to_string(), true);
    // Always return true so the flag itself is persisted even when no reset was needed.
    let _ = changed;
    true
}

/// Helper: apply chinese_balanced preset fields to settings without calling
/// load_model (TranscriptionManager is not yet started at migration time).
fn reset_to_chinese_balanced(settings: &mut AppSettings) {
    if let Some(preset) = default_asr_presets()
        .into_iter()
        .find(|p| p.id == "chinese_balanced")
    {
        settings.selected_language = preset.language.clone();
        if let Some(chain) = preset.require_post_process_chain.clone() {
            settings.post_process_chain = Some(chain);
        }
        if let Some(on_device) = preset.require_apple_speech_on_device {
            settings.apple_speech_require_on_device = on_device;
        }
        settings.active_preset_id = Some(preset.id.clone());
        settings.selected_model = preset.model_id.clone();
    }
}

/// v_0_8_18 — CT-Transformer / punc_zh subsystem removal.
///
/// No data mutation required: serde already ignores the removed `punc_zh_enabled`
/// field via `#[serde(default)]` on both `AppSettings` and `AppProfile`.  This
/// migration exists solely as a version marker so the removal is visible in the
/// `migration_applied` audit log.
const MIGRATION_V_0_8_18_DROP_PUNC_ZH: &str = "v_0_8_18_drop_punc_zh_field";

pub fn ensure_v_0_8_18_drop_punc_zh_migration(settings: &mut AppSettings) -> bool {
    if settings
        .migration_applied
        .get(MIGRATION_V_0_8_18_DROP_PUNC_ZH)
        .copied()
        .unwrap_or(false)
    {
        return false; // already applied
    }

    log::info!(
        "Migration {}: CT-Transformer punc_zh subsystem removed; \
         punc_zh_enabled field (if present in stored JSON) silently ignored by serde",
        MIGRATION_V_0_8_18_DROP_PUNC_ZH
    );

    settings
        .migration_applied
        .insert(MIGRATION_V_0_8_18_DROP_PUNC_ZH.to_string(), true);
    true
}

/// v_0_8_19 — Experimental settings group removal.
///
/// Removes `experimental_enabled`, `keyboard_implementation`,
/// `whisper_accelerator`, and `whisper_gpu_device` fields.  No data mutation
/// required: serde's `#[serde(default)]` on `AppSettings` ignores these
/// removed fields if present in stored JSON.  This migration exists solely as
/// a version marker so the removal is visible in the `migration_applied` audit
/// log.
const MIGRATION_V_0_8_19_DROP_EXPERIMENTAL: &str = "v_0_8_19_drop_experimental_fields";

pub fn ensure_v_0_8_19_drop_experimental_migration(settings: &mut AppSettings) -> bool {
    if settings
        .migration_applied
        .get(MIGRATION_V_0_8_19_DROP_EXPERIMENTAL)
        .copied()
        .unwrap_or(false)
    {
        return false; // already applied
    }

    log::info!(
        "Migration {}: experimental settings group removed; \
         experimental_enabled, keyboard_implementation, whisper_accelerator, \
         whisper_gpu_device fields (if present in stored JSON) silently ignored by serde",
        MIGRATION_V_0_8_19_DROP_EXPERIMENTAL
    );

    settings
        .migration_applied
        .insert(MIGRATION_V_0_8_19_DROP_EXPERIMENTAL.to_string(), true);
    true
}

pub const SETTINGS_STORE_PATH: &str = "settings_store.json";

pub fn get_default_settings() -> AppSettings {
    #[cfg(target_os = "windows")]
    let default_shortcut = "ctrl+space";
    #[cfg(target_os = "macos")]
    let default_shortcut = "option+space";
    #[cfg(target_os = "linux")]
    let default_shortcut = "ctrl+space";
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let default_shortcut = "alt+space";

    let mut bindings = HashMap::new();
    bindings.insert(
        "transcribe".to_string(),
        ShortcutBinding {
            id: "transcribe".to_string(),
            name: "Transcribe".to_string(),
            description: "Converts your speech into text.".to_string(),
            default_binding: default_shortcut.to_string(),
            current_binding: default_shortcut.to_string(),
        },
    );
    #[cfg(target_os = "windows")]
    let default_post_process_shortcut = "ctrl+shift+space";
    #[cfg(target_os = "macos")]
    let default_post_process_shortcut = "option+shift+space";
    #[cfg(target_os = "linux")]
    let default_post_process_shortcut = "ctrl+shift+space";
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let default_post_process_shortcut = "alt+shift+space";

    bindings.insert(
        "transcribe_with_post_process".to_string(),
        ShortcutBinding {
            id: "transcribe_with_post_process".to_string(),
            name: "Transcribe with Post-Processing".to_string(),
            description: "Converts your speech into text and applies AI post-processing."
                .to_string(),
            default_binding: default_post_process_shortcut.to_string(),
            current_binding: default_post_process_shortcut.to_string(),
        },
    );
    bindings.insert(
        "cancel".to_string(),
        ShortcutBinding {
            id: "cancel".to_string(),
            name: "Cancel".to_string(),
            description: "Cancels the current recording.".to_string(),
            default_binding: "escape".to_string(),
            current_binding: "escape".to_string(),
        },
    );

    AppSettings {
        bindings,
        push_to_talk: true,
        audio_feedback: false,
        audio_feedback_volume: default_audio_feedback_volume(),
        sound_theme: default_sound_theme(),
        start_hidden: default_start_hidden(),
        autostart_enabled: default_autostart_enabled(),
        update_checks_enabled: default_update_checks_enabled(),
        // Handy+ default: SenseVoice (transcribe-rs path, ~152 MB).  Real-world
        // testing showed it beats both Apple Speech (which on the current
        // architecture only emits partials *after* stop, with no Chinese
        // punctuation) and Fun-ASR-Nano (1 GB, 1-2.5 s per utterance) for an
        // input-method workflow: ~70 ms inference per 10 s of audio, built-in
        // Chinese punctuation/ITN, auto-detects zh/en/ja/ko/yue, no system
        // permission flow, and CER trails Fun-ASR-Nano by under 1 point.
        #[cfg(target_os = "macos")]
        selected_model: "sense-voice-int8".to_string(),
        #[cfg(not(target_os = "macos"))]
        selected_model: "".to_string(),
        always_on_microphone: false,
        selected_microphone: None,
        clamshell_microphone: None,
        selected_output_device: None,
        translate_to_english: false,
        selected_language: "auto".to_string(),
        overlay_position: default_overlay_position(),
        debug_mode: false,
        log_level: default_log_level(),
        custom_words: Vec::new(),
        model_unload_timeout: ModelUnloadTimeout::default(),
        word_correction_threshold: default_word_correction_threshold(),
        history_limit: default_history_limit(),
        recording_retention_period: default_recording_retention_period(),
        paste_method: PasteMethod::default(),
        clipboard_handling: ClipboardHandling::default(),
        auto_submit: default_auto_submit(),
        auto_submit_key: AutoSubmitKey::default(),
        post_process_enabled: default_post_process_enabled(),
        post_process_provider_id: default_post_process_provider_id(),
        post_process_providers: default_post_process_providers(),
        post_process_api_keys: default_post_process_api_keys(),
        post_process_models: default_post_process_models(),
        post_process_prompts: default_post_process_prompts(),
        // Handy+ default: pre-select the bundled "Improve Transcriptions" prompt
        // so post-processing works out of the box (paired with Apple Intelligence
        // provider on macOS). Users can swap or write their own from settings.
        post_process_selected_prompt_id: Some("default_improve_transcriptions".to_string()),
        mute_while_recording: false,
        append_trailing_space: false,
        app_language: default_app_language(),
        show_tray_icon: default_show_tray_icon(),
        paste_delay_ms: default_paste_delay_ms(),
        typing_tool: default_typing_tool(),
        external_script_path: None,
        custom_filler_words: None,
        ort_accelerator: OrtAcceleratorSetting::default(),
        extra_recording_buffer_ms: 0,
        power_mode_enabled: default_power_mode_enabled(),
        app_profiles: default_app_profiles(),
        profile_hot_swap_engine: false,
        apple_speech_require_on_device: default_apple_speech_require_on_device(),
        diary_dir: None,
        diary_keywords: default_diary_keywords(),
        post_process_chain: None,
        hotwords_boost: default_hotwords_boost(),
        active_preset_id: None,
        migration_applied: HashMap::new(),
        custom_word_aliases: HashMap::new(),
        apple_speech_incremental_paste: default_apple_speech_incremental_paste(),
    }
}

impl AppSettings {
    pub fn active_post_process_provider(&self) -> Option<&PostProcessProvider> {
        self.post_process_providers
            .iter()
            .find(|provider| provider.id == self.post_process_provider_id)
    }

    pub fn post_process_provider(&self, provider_id: &str) -> Option<&PostProcessProvider> {
        self.post_process_providers
            .iter()
            .find(|provider| provider.id == provider_id)
    }

    pub fn post_process_provider_mut(
        &mut self,
        provider_id: &str,
    ) -> Option<&mut PostProcessProvider> {
        self.post_process_providers
            .iter_mut()
            .find(|provider| provider.id == provider_id)
    }
}

pub fn load_or_create_app_settings(app: &AppHandle) -> AppSettings {
    // Initialize store
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    let mut settings = if let Some(settings_value) = store.get("settings") {
        // Parse the entire settings object
        match serde_json::from_value::<AppSettings>(settings_value) {
            Ok(mut settings) => {
                debug!("Found existing settings: {:?}", settings);
                let default_settings = get_default_settings();
                let mut updated = false;

                // Merge default bindings into existing settings
                for (key, value) in default_settings.bindings {
                    use std::collections::hash_map::Entry;
                    if let Entry::Vacant(entry) = settings.bindings.entry(key) {
                        debug!("Adding missing binding: {}", entry.key());
                        entry.insert(value);
                        updated = true;
                    }
                }

                if updated {
                    debug!("Settings updated with new bindings");
                    store.set("settings", serde_json::to_value(&settings).unwrap());
                }

                settings
            }
            Err(e) => {
                warn!("Failed to parse settings: {}", e);
                // Fall back to default settings if parsing fails
                let default_settings = get_default_settings();
                store.set("settings", serde_json::to_value(&default_settings).unwrap());
                default_settings
            }
        }
    } else {
        let default_settings = get_default_settings();
        store.set("settings", serde_json::to_value(&default_settings).unwrap());
        default_settings
    };

    let mut changed = ensure_post_process_defaults(&mut settings);
    changed |= ensure_app_profiles_defaults(&mut settings);
    changed |= ensure_v_0_8_10_enable_builtins_migration(&mut settings);
    {
        let visible = visible_preset_ids_from_settings(&settings);
        let visible_refs: Vec<&str> = visible.iter().map(|s| s.as_str()).collect();
        changed |= ensure_v_0_8_15_drop_hidden_preset_migration(&mut settings, &visible_refs);
        changed |= ensure_v_0_8_16_drop_hidden_preset_migration(&mut settings, &visible_refs);
    }
    changed |= ensure_v_0_8_17_collapse_migration(&mut settings);
    changed |= ensure_v_0_8_18_drop_punc_zh_migration(&mut settings);
    changed |= ensure_v_0_8_19_drop_experimental_migration(&mut settings);
    if changed {
        store.set("settings", serde_json::to_value(&settings).unwrap());
    }

    settings
}

pub fn get_settings(app: &AppHandle) -> AppSettings {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    let mut settings = if let Some(settings_value) = store.get("settings") {
        serde_json::from_value::<AppSettings>(settings_value).unwrap_or_else(|_| {
            let default_settings = get_default_settings();
            store.set("settings", serde_json::to_value(&default_settings).unwrap());
            default_settings
        })
    } else {
        let default_settings = get_default_settings();
        store.set("settings", serde_json::to_value(&default_settings).unwrap());
        default_settings
    };

    let mut changed = ensure_post_process_defaults(&mut settings);
    changed |= ensure_app_profiles_defaults(&mut settings);
    changed |= ensure_v_0_8_10_enable_builtins_migration(&mut settings);
    {
        let visible = visible_preset_ids_from_settings(&settings);
        let visible_refs: Vec<&str> = visible.iter().map(|s| s.as_str()).collect();
        changed |= ensure_v_0_8_15_drop_hidden_preset_migration(&mut settings, &visible_refs);
        changed |= ensure_v_0_8_16_drop_hidden_preset_migration(&mut settings, &visible_refs);
    }
    changed |= ensure_v_0_8_17_collapse_migration(&mut settings);
    changed |= ensure_v_0_8_18_drop_punc_zh_migration(&mut settings);
    changed |= ensure_v_0_8_19_drop_experimental_migration(&mut settings);
    if changed {
        store.set("settings", serde_json::to_value(&settings).unwrap());
    }

    settings
}

pub fn write_settings(app: &AppHandle, settings: AppSettings) {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    store.set("settings", serde_json::to_value(&settings).unwrap());
}

pub fn get_bindings(app: &AppHandle) -> HashMap<String, ShortcutBinding> {
    let settings = get_settings(app);

    settings.bindings
}

pub fn get_stored_binding(app: &AppHandle, id: &str) -> ShortcutBinding {
    let bindings = get_bindings(app);

    let binding = bindings.get(id).unwrap().clone();

    binding
}

pub fn get_history_limit(app: &AppHandle) -> usize {
    let settings = get_settings(app);
    settings.history_limit
}

pub fn get_recording_retention_period(app: &AppHandle) -> RecordingRetentionPeriod {
    let settings = get_settings(app);
    settings.recording_retention_period
}

#[cfg(test)]
mod migration_tests {
    use super::*;

    fn build_settings_with_disabled_builtins() -> AppSettings {
        let mut s = get_default_settings();
        for p in s.app_profiles.iter_mut() {
            p.enabled = false;
        }
        s.migration_applied = HashMap::new();
        s
    }

    #[test]
    fn migration_enables_target_builtins() {
        let mut s = build_settings_with_disabled_builtins();
        ensure_v_0_8_10_enable_builtins_migration(&mut s);

        let enabled_ids: Vec<&str> = s
            .app_profiles
            .iter()
            .filter(|p| p.enabled)
            .map(|p| p.id.as_str())
            .collect();

        assert!(enabled_ids.contains(&"builtin_code"));
        assert!(enabled_ids.contains(&"builtin_chat"));
        assert!(enabled_ids.contains(&"builtin_writing"));
        assert!(enabled_ids.contains(&"builtin_claude_code"));
        // builtin_default_fallback should remain disabled (no matchers, won't fire)
        assert!(
            !s.app_profiles
                .iter()
                .find(|p| p.id == "builtin_default_fallback")
                .unwrap()
                .enabled
        );

        // migration_applied should be recorded
        assert_eq!(
            s.migration_applied.get("v_0_8_10_enable_builtins"),
            Some(&true)
        );
    }

    #[test]
    fn migration_idempotent() {
        let mut s = build_settings_with_disabled_builtins();
        ensure_v_0_8_10_enable_builtins_migration(&mut s);

        // user explicitly disables builtin_code after migration
        s.app_profiles
            .iter_mut()
            .find(|p| p.id == "builtin_code")
            .unwrap()
            .enabled = false;

        // run migration again — should NOT re-enable (already marked applied)
        ensure_v_0_8_10_enable_builtins_migration(&mut s);

        let code_profile = s
            .app_profiles
            .iter()
            .find(|p| p.id == "builtin_code")
            .unwrap();
        assert!(!code_profile.enabled, "migration should be idempotent");
    }

    // ── v0.8.15 drop-hidden-preset migration tests ────────────────────────

    /// Helper: build settings that look like an old user who had apple_native
    /// selected and then upgraded to macOS 26 (where apple_native is hidden).
    fn settings_with_active_preset(preset_id: &str) -> AppSettings {
        let mut s = get_default_settings();
        s.active_preset_id = Some(preset_id.to_string());
        s.selected_model = "apple-speech".to_string();
        s.migration_applied = HashMap::new();
        s
    }

    #[test]
    fn migration_resets_hidden_preset_to_chinese_balanced() {
        let mut s = settings_with_active_preset("apple_native");
        // Simulate macOS 26: apple_native is not visible.
        let visible: &[&str] = &["chinese_balanced", "multilingual_offline"];
        let changed = ensure_v_0_8_15_drop_hidden_preset_migration(&mut s, visible);

        assert!(changed, "migration should report a change");
        assert_eq!(
            s.active_preset_id.as_deref(),
            Some("chinese_balanced"),
            "active_preset_id must be reset"
        );
        assert_eq!(
            s.selected_model, "sense-voice-int8",
            "selected_model must match chinese_balanced"
        );
        assert_eq!(
            s.migration_applied
                .get("v_0_8_15_drop_hidden_active_preset"),
            Some(&true)
        );
    }

    #[test]
    fn migration_leaves_visible_preset_untouched() {
        let mut s = settings_with_active_preset("chinese_balanced");
        s.selected_model = "sense-voice-int8".to_string();
        let visible: &[&str] = &["chinese_balanced", "multilingual_offline"];
        let changed = ensure_v_0_8_15_drop_hidden_preset_migration(&mut s, visible);

        assert!(changed, "migration flag itself causes a change");
        assert_eq!(
            s.active_preset_id.as_deref(),
            Some("chinese_balanced"),
            "preset should be unchanged"
        );
        assert_eq!(s.selected_model, "sense-voice-int8");
    }

    #[test]
    fn migration_is_idempotent_when_already_applied() {
        let mut s = settings_with_active_preset("apple_native");
        s.selected_model = "apple-speech".to_string();
        // Pre-mark as applied (simulates a user who already migrated).
        s.migration_applied
            .insert("v_0_8_15_drop_hidden_active_preset".to_string(), true);
        let visible: &[&str] = &["chinese_balanced", "multilingual_offline"];
        let changed = ensure_v_0_8_15_drop_hidden_preset_migration(&mut s, visible);

        assert!(!changed, "already-applied migration must return false");
        // Data should be untouched (apple_native still stored — migration skipped).
        assert_eq!(s.active_preset_id.as_deref(), Some("apple_native"));
        assert_eq!(s.selected_model, "apple-speech");
    }

    // ── v0.8.17 collapse-to-two-presets migration tests ──────────────────────

    fn v17_settings(preset_id: &str) -> AppSettings {
        let mut s = get_default_settings();
        s.active_preset_id = Some(preset_id.to_string());
        s.migration_applied = HashMap::new();
        s
    }

    #[test]
    fn v17_apple_native_resets_to_chinese_balanced() {
        let mut s = v17_settings("apple_native");
        ensure_v_0_8_17_collapse_migration(&mut s);
        assert_eq!(s.active_preset_id.as_deref(), Some("chinese_balanced"));
        assert_eq!(s.selected_model, "sense-voice-int8");
        assert_eq!(
            s.migration_applied.get(MIGRATION_V_0_8_17_COLLAPSE),
            Some(&true)
        );
    }

    #[test]
    fn v17_experimental_voxtral_resets_to_chinese_balanced() {
        let mut s = v17_settings("experimental_voxtral");
        ensure_v_0_8_17_collapse_migration(&mut s);
        assert_eq!(s.active_preset_id.as_deref(), Some("chinese_balanced"));
        assert_eq!(s.selected_model, "sense-voice-int8");
    }

    #[test]
    fn v17_experimental_qwen3_resets_to_chinese_balanced() {
        let mut s = v17_settings("experimental_qwen3");
        ensure_v_0_8_17_collapse_migration(&mut s);
        assert_eq!(s.active_preset_id.as_deref(), Some("chinese_balanced"));
        assert_eq!(s.selected_model, "sense-voice-int8");
    }

    #[test]
    fn v17_multilingual_offline_resets_to_chinese_balanced() {
        let mut s = v17_settings("multilingual_offline");
        ensure_v_0_8_17_collapse_migration(&mut s);
        assert_eq!(s.active_preset_id.as_deref(), Some("chinese_balanced"));
        assert_eq!(s.selected_model, "sense-voice-int8");
    }

    #[test]
    fn v17_experimental_qwen3_mlx_renamed_to_qwen3_mlx_on_aarch64() {
        // migration:
        // This test exercises the rename path. On non-aarch64 CI it will still
        // compile and run; the cfg! check inside the migration handles the
        // platform difference — on non-aarch64 it resets to chinese_balanced
        // instead (both are valid outcomes, tested per-platform).
        let mut s = v17_settings("experimental_qwen3_mlx"); // migration:
        ensure_v_0_8_17_collapse_migration(&mut s);
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            assert_eq!(s.active_preset_id.as_deref(), Some("qwen3_mlx"));
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            assert_eq!(s.active_preset_id.as_deref(), Some("chinese_balanced"));
        }
        assert_eq!(
            s.migration_applied.get(MIGRATION_V_0_8_17_COLLAPSE),
            Some(&true)
        );
    }

    #[test]
    fn v17_chinese_balanced_untouched() {
        let mut s = v17_settings("chinese_balanced");
        s.selected_model = "sense-voice-int8".to_string();
        ensure_v_0_8_17_collapse_migration(&mut s);
        assert_eq!(s.active_preset_id.as_deref(), Some("chinese_balanced"));
        assert_eq!(s.selected_model, "sense-voice-int8");
    }

    #[test]
    fn v17_none_preset_id_untouched() {
        let mut s = get_default_settings();
        s.active_preset_id = None;
        s.migration_applied = HashMap::new();
        ensure_v_0_8_17_collapse_migration(&mut s);
        assert_eq!(s.active_preset_id, None);
    }

    #[test]
    fn v17_idempotent_when_already_applied() {
        let mut s = v17_settings("apple_native");
        s.selected_model = "apple-speech".to_string();
        s.migration_applied
            .insert(MIGRATION_V_0_8_17_COLLAPSE.to_string(), true);
        let changed = ensure_v_0_8_17_collapse_migration(&mut s);
        // Should return false (already applied) and leave data untouched.
        assert!(!changed);
        assert_eq!(s.active_preset_id.as_deref(), Some("apple_native"));
        assert_eq!(s.selected_model, "apple-speech");
    }

    // ── v0.8.18 drop-punc-zh migration tests ─────────────────────────────────

    #[test]
    fn v18_marks_migration_applied_and_returns_true() {
        let mut s = get_default_settings();
        s.migration_applied = HashMap::new();
        let changed = ensure_v_0_8_18_drop_punc_zh_migration(&mut s);
        assert!(changed, "first run should return true");
        assert_eq!(
            s.migration_applied.get(MIGRATION_V_0_8_18_DROP_PUNC_ZH),
            Some(&true),
            "migration flag must be set"
        );
    }

    #[test]
    fn v18_idempotent_when_already_applied() {
        let mut s = get_default_settings();
        s.migration_applied
            .insert(MIGRATION_V_0_8_18_DROP_PUNC_ZH.to_string(), true);
        let changed = ensure_v_0_8_18_drop_punc_zh_migration(&mut s);
        assert!(!changed, "second run must be a no-op");
    }

    /// Old settings JSON with punc_zh_enabled present should deserialise cleanly
    /// (serde ignores unknown/removed fields by default).
    #[test]
    fn v18_old_json_with_punc_zh_field_deserialises_ok() {
        // Serialise a default AppSettings, then inject a punc_zh_enabled field
        // into the JSON to simulate a stored settings from before v_0_8_18.
        let defaults = get_default_settings();
        let mut as_value = serde_json::to_value(&defaults).expect("serialise defaults");
        as_value
            .as_object_mut()
            .unwrap()
            .insert("punc_zh_enabled".to_string(), serde_json::Value::Bool(true));
        let old_json = serde_json::to_string(&as_value).expect("re-serialise");

        // serde must ignore the unknown field and produce a valid AppSettings.
        let result: Result<AppSettings, _> = serde_json::from_str(&old_json);
        assert!(
            result.is_ok(),
            "old JSON with punc_zh_enabled must deserialise without error: {:?}",
            result.err()
        );
    }

    // ── v0.8.19 drop-experimental-fields migration tests ─────────────────────

    #[test]
    fn v19_marks_migration_applied_and_returns_true() {
        let mut s = get_default_settings();
        s.migration_applied = HashMap::new();
        let changed = ensure_v_0_8_19_drop_experimental_migration(&mut s);
        assert!(changed, "first run must return true");
        assert_eq!(
            s.migration_applied
                .get(MIGRATION_V_0_8_19_DROP_EXPERIMENTAL),
            Some(&true),
            "migration flag must be set"
        );
    }

    #[test]
    fn v19_idempotent_when_already_applied() {
        let mut s = get_default_settings();
        s.migration_applied
            .insert(MIGRATION_V_0_8_19_DROP_EXPERIMENTAL.to_string(), true);
        let changed = ensure_v_0_8_19_drop_experimental_migration(&mut s);
        assert!(!changed, "second run must return false");
    }

    #[test]
    fn v19_old_json_with_experimental_fields_deserialises_ok() {
        // Simulate a stored settings JSON that still has the removed fields.
        let defaults = get_default_settings();
        let mut as_value = serde_json::to_value(&defaults).expect("serialise defaults");
        let obj = as_value.as_object_mut().unwrap();
        obj.insert(
            "experimental_enabled".to_string(),
            serde_json::Value::Bool(true),
        );
        obj.insert(
            "keyboard_implementation".to_string(),
            serde_json::Value::String("handy_keys".to_string()),
        );
        obj.insert(
            "whisper_accelerator".to_string(),
            serde_json::Value::String("auto".to_string()),
        );
        obj.insert(
            "whisper_gpu_device".to_string(),
            serde_json::Value::Number((-1i64).into()),
        );
        let old_json = serde_json::to_string(&as_value).expect("re-serialise");

        // serde must ignore the unknown fields and produce a valid AppSettings.
        let result: Result<AppSettings, _> = serde_json::from_str(&old_json);
        assert!(
            result.is_ok(),
            "old JSON with experimental fields must deserialise without error: {:?}",
            result.err()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_disable_auto_submit() {
        let settings = get_default_settings();
        assert!(!settings.auto_submit);
        assert_eq!(settings.auto_submit_key, AutoSubmitKey::Enter);
    }

    #[test]
    fn debug_output_redacts_api_keys() {
        let mut settings = get_default_settings();
        settings
            .post_process_api_keys
            .insert("openai".to_string(), "sk-proj-secret-key-12345".to_string());
        settings.post_process_api_keys.insert(
            "anthropic".to_string(),
            "sk-ant-secret-key-67890".to_string(),
        );
        settings
            .post_process_api_keys
            .insert("empty_provider".to_string(), "".to_string());

        let debug_output = format!("{:?}", settings);

        assert!(!debug_output.contains("sk-proj-secret-key-12345"));
        assert!(!debug_output.contains("sk-ant-secret-key-67890"));
        assert!(debug_output.contains("[REDACTED]"));
    }

    #[test]
    fn secret_map_debug_redacts_values() {
        let map = SecretMap(HashMap::from([("key".into(), "secret".into())]));
        let out = format!("{:?}", map);
        assert!(!out.contains("secret"));
        assert!(out.contains("[REDACTED]"));
    }
}
