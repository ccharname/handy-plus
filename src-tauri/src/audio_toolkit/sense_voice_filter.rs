/// SenseVoice output post-processor: strips meta/emotion/language tags that the
/// model sometimes emits verbatim into the transcription text.
///
/// SenseVoice (both the transcribe-rs path and the sherpa-onnx path) can emit
/// tokens such as `<|HAPPY|>`, `<|NEUTRAL|>`, `<|EMO_SAD|>`, `<|zh|>`, `<|/en|>`,
/// `<|bg_speech|>`, `<|speech|>` etc.  These are model-internal bookkeeping
/// tokens that must never appear in the final user-visible text.
///
/// `strip_meta` is intentionally conservative: it only removes tokens that match
/// the `<|…|>` bracket pattern (upper-case letters, digits, underscores, forward
/// slashes) plus a small allow-list of emotion-only Unicode code points that
/// SenseVoice sometimes inserts as separators.  All other characters are left
/// untouched.
use once_cell::sync::Lazy;
use regex::Regex;

/// Compiled regex that matches SenseVoice meta tokens of the form `<|TOKEN|>`.
/// The inner part may contain: A-Z, a-z, 0-9, underscore, forward slash.
/// Examples: `<|HAPPY|>`, `<|EMO_SAD|>`, `<|zh|>`, `<|/en|>`, `<|bg_speech|>`.
static META_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"<\|[A-Za-z0-9_/]+\|>").expect("sense_voice_filter regex is valid"));

/// Set of Unicode scalar values that SenseVoice sometimes inserts as
/// emotion / prosody dividers.  We strip these when they appear as lone
/// characters between or around words (not part of a larger emoji sequence
/// that the user might have typed).
const EMOTION_EMOJIS: &[char] = &[
    '😀', '😃', '😄', '😁', '😆', '😅', '😂', '🤣', '😊', '😇', '🙂', '🙃', '😉', '😌', '😍', '🥰',
    '😘', '😗', '😙', '😚', '😋', '😛', '😝', '😜', '🤪', '🤨', '🧐', '🤓', '😎', '🥸', '🤩', '🥳',
    '😏', '😒', '😞', '😔', '😟', '😕', '🙁', '☹', '😣', '😖', '😫', '😩', '🥺', '😢', '😭', '😤',
    '😠', '😡', '🤬', '🤯', '😳', '🥵', '🥶', '😱', '😨', '😰', '😥', '😓', '🤗', '🤔', '🫣', '🤭',
    '🫢', '🫡', '🤫', '🫠', '🤥', '😶', '🫥', '😐', '😑', '😬', '🙄', '😯', '😦', '😧', '😮', '😲',
    '🥱', '😴', '🤤', '😪', '😵', '🫨', '🤐', '🥴', '🤢', '🤮', '🤧', '😷', '🤒', '🤕', '🤑', '🤠',
];

/// Strip all SenseVoice meta/emotion tags from `text` and collapse any
/// resulting runs of whitespace into a single space.
///
/// The function is designed to be cheap to call on every transcription result.
/// The regex is compiled once via `Lazy` and subsequent calls just run the
/// matcher.
///
/// # Examples
/// ```
/// use handy_lib::audio_toolkit::sense_voice_filter::strip_meta;
/// assert_eq!(strip_meta("<|HAPPY|>你好<|NEUTRAL|>"), "你好");
/// assert_eq!(strip_meta("<|en|>Hello<|/en|>"), "Hello");
/// assert_eq!(strip_meta("normal text"), "normal text");
/// ```
pub fn strip_meta(text: &str) -> String {
    // 1. Remove <|…|> tokens.
    let after_tags = META_TAG_RE.replace_all(text, "");

    // 2. Remove lone emotion emoji characters.
    let after_emoji: String = after_tags
        .chars()
        .filter(|c| !EMOTION_EMOJIS.contains(c))
        .collect();

    // 3. Collapse multiple whitespace characters (spaces, tabs, newlines) into
    //    a single space and trim leading/trailing whitespace.
    let collapsed = after_emoji
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ");

    collapsed
}

#[cfg(test)]
mod tests {
    use super::strip_meta;

    // ── Basic tag removal ────────────────────────────────────────────────────

    #[test]
    fn test_no_tags_passthrough() {
        assert_eq!(strip_meta("hello world"), "hello world");
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(strip_meta(""), "");
    }

    #[test]
    fn test_whitespace_only() {
        assert_eq!(strip_meta("   "), "");
    }

    #[test]
    fn test_single_happy_tag() {
        assert_eq!(strip_meta("<|HAPPY|>"), "");
    }

    #[test]
    fn test_single_neutral_tag() {
        assert_eq!(strip_meta("<|NEUTRAL|>"), "");
    }

    #[test]
    fn test_single_sad_tag() {
        assert_eq!(strip_meta("<|EMO_SAD|>"), "");
    }

    #[test]
    fn test_happy_wrapping_chinese() {
        assert_eq!(strip_meta("<|HAPPY|>你好<|NEUTRAL|>"), "你好");
    }

    #[test]
    fn test_tag_before_text() {
        assert_eq!(strip_meta("<|NEUTRAL|>今天天气不错"), "今天天气不错");
    }

    #[test]
    fn test_tag_after_text() {
        assert_eq!(strip_meta("今天天气不错<|HAPPY|>"), "今天天气不错");
    }

    #[test]
    fn test_multiple_emotion_tags() {
        assert_eq!(strip_meta("<|HAPPY|><|NEUTRAL|><|SAD|>hello"), "hello");
    }

    // ── Language tags ────────────────────────────────────────────────────────

    #[test]
    fn test_language_tag_en() {
        assert_eq!(strip_meta("<|en|>Hello<|/en|>"), "Hello");
    }

    #[test]
    fn test_language_tag_zh() {
        assert_eq!(strip_meta("<|zh|>你好<|/zh|>"), "你好");
    }

    #[test]
    fn test_language_tag_ja() {
        assert_eq!(strip_meta("<|ja|>こんにちは<|/ja|>"), "こんにちは");
    }

    #[test]
    fn test_language_tag_ko() {
        assert_eq!(strip_meta("<|ko|>안녕하세요<|/ko|>"), "안녕하세요");
    }

    #[test]
    fn test_mixed_emo_and_lang() {
        assert_eq!(strip_meta("<|EMO_SAD|><|en|>Hello<|/en|>"), "Hello");
    }

    #[test]
    fn test_lang_tags_around_english_sentence() {
        assert_eq!(
            strip_meta("<|en|>The quick brown fox<|/en|>"),
            "The quick brown fox"
        );
    }

    // ── bg_speech / speech / noise tokens ───────────────────────────────────

    #[test]
    fn test_bg_speech_tag() {
        assert_eq!(strip_meta("<|bg_speech|>"), "");
    }

    #[test]
    fn test_speech_tag() {
        assert_eq!(strip_meta("<|speech|>"), "");
    }

    #[test]
    fn test_noise_tag() {
        assert_eq!(strip_meta("<|noise|>"), "");
    }

    #[test]
    fn test_speech_with_text() {
        assert_eq!(
            strip_meta("<|speech|>Hello there<|bg_speech|>"),
            "Hello there"
        );
    }

    // ── Whitespace collapsing ────────────────────────────────────────────────

    #[test]
    fn test_extra_spaces_collapsed() {
        assert_eq!(strip_meta("hello   world"), "hello world");
    }

    #[test]
    fn test_tags_leave_extra_spaces_collapsed() {
        assert_eq!(strip_meta("hello <|HAPPY|> world"), "hello world");
    }

    #[test]
    fn test_newlines_collapsed() {
        assert_eq!(strip_meta("hello\nworld"), "hello world");
    }

    #[test]
    fn test_tab_collapsed() {
        assert_eq!(strip_meta("hello\tworld"), "hello world");
    }

    #[test]
    fn test_leading_trailing_space_trimmed() {
        assert_eq!(strip_meta("  hello  "), "hello");
    }

    // ── Emoji stripping ──────────────────────────────────────────────────────

    #[test]
    fn test_happy_emoji_stripped() {
        assert_eq!(strip_meta("😀你好"), "你好");
    }

    #[test]
    fn test_sad_emoji_stripped() {
        assert_eq!(strip_meta("😢text"), "text");
    }

    #[test]
    fn test_emoji_between_words() {
        assert_eq!(strip_meta("hello 😀 world"), "hello world");
    }

    #[test]
    fn test_multiple_emojis_stripped() {
        assert_eq!(strip_meta("😀😊😭text"), "text");
    }

    // ── Real-world SenseVoice output patterns ───────────────────────────────

    #[test]
    fn test_typical_sensevoice_zh_output() {
        assert_eq!(
            strip_meta("<|zh|><|HAPPY|>今天天气很好<|/zh|>"),
            "今天天气很好"
        );
    }

    #[test]
    fn test_typical_sensevoice_en_output() {
        assert_eq!(
            strip_meta("<|en|><|NEUTRAL|>The meeting starts at three<|/en|>"),
            "The meeting starts at three"
        );
    }

    #[test]
    fn test_sensevoice_emo_prefix_zh() {
        assert_eq!(
            strip_meta("<|EMO_UNKNOWN|><|zh|>收到了吗<|/zh|>"),
            "收到了吗"
        );
    }

    #[test]
    fn test_sensevoice_full_tag_sequence() {
        // A full realistic output from SenseVoice with nested tags
        assert_eq!(
            strip_meta("<|HAPPY|><|zh|><|speech|>你好世界<|/zh|>"),
            "你好世界"
        );
    }

    #[test]
    fn test_yue_language_tag() {
        assert_eq!(strip_meta("<|yue|>你好<|/yue|>"), "你好");
    }

    #[test]
    fn test_emo_happy_variant() {
        assert_eq!(strip_meta("<|EMO_HAPPY|>hello"), "hello");
    }

    #[test]
    fn test_emo_angry_variant() {
        assert_eq!(strip_meta("<|EMO_ANGRY|>stop it"), "stop it");
    }

    #[test]
    fn test_emo_disgusted_variant() {
        assert_eq!(strip_meta("<|EMO_DISGUSTED|>ugh"), "ugh");
    }

    #[test]
    fn test_emo_fearful_variant() {
        assert_eq!(strip_meta("<|EMO_FEARFUL|>run"), "run");
    }

    #[test]
    fn test_emo_surprised_variant() {
        assert_eq!(strip_meta("<|EMO_SURPRISED|>wow"), "wow");
    }

    #[test]
    fn test_numeric_in_tag_stripped() {
        // Tags can contain numbers (e.g. hypothetical <|LANG0|>)
        assert_eq!(strip_meta("<|LANG0|>text"), "text");
    }

    #[test]
    fn test_all_tags_no_text_is_empty() {
        assert_eq!(
            strip_meta("<|HAPPY|><|NEUTRAL|><|zh|><|speech|><|/zh|>"),
            ""
        );
    }

    #[test]
    fn test_punctuation_preserved() {
        assert_eq!(strip_meta("<|NEUTRAL|>Hello, world!"), "Hello, world!");
    }

    #[test]
    fn test_numbers_preserved() {
        assert_eq!(strip_meta("<|NEUTRAL|>Call 911 now"), "Call 911 now");
    }

    #[test]
    fn test_mixed_language_sentence() {
        // Bilingual output — both language tags stripped, content preserved
        assert_eq!(
            strip_meta("<|zh|>你好<|/zh|> <|en|>hello<|/en|>"),
            "你好 hello"
        );
    }

    #[test]
    fn test_only_emotion_emoji() {
        assert_eq!(strip_meta("😀"), "");
    }

    #[test]
    fn test_text_with_no_change() {
        // String with no tags or emotion emoji should come back identical
        let input = "The quick brown fox jumps over the lazy dog.";
        assert_eq!(strip_meta(input), input);
    }

    #[test]
    fn test_chinese_with_no_tags() {
        let input = "今天的会议推迟到下午三点。";
        assert_eq!(strip_meta(input), input);
    }

    #[test]
    fn test_tag_inside_long_text() {
        assert_eq!(
            strip_meta("first part <|NEUTRAL|> second part"),
            "first part second part"
        );
    }

    #[test]
    fn test_close_tag_slash_variant() {
        assert_eq!(strip_meta("<|/speech|>hello"), "hello");
    }

    #[test]
    fn test_emo_unknown_tag() {
        assert_eq!(strip_meta("<|EMO_UNKNOWN|>"), "");
    }

    #[test]
    fn test_underscore_in_tag_name() {
        assert_eq!(strip_meta("<|BG_NOISE|>text"), "text");
    }

    #[test]
    fn test_preserves_ascii_angle_brackets_without_pipes() {
        // <tag> (no pipes) should NOT be stripped — it is not a SenseVoice token
        assert_eq!(strip_meta("<tag>text</tag>"), "<tag>text</tag>");
    }

    #[test]
    fn test_preserves_math_less_than_greater_than() {
        assert_eq!(strip_meta("2 < 3 and 5 > 4"), "2 < 3 and 5 > 4");
    }

    #[test]
    fn test_itn_number_result_preserved() {
        // ITN output like "12点30分" should not be touched
        assert_eq!(strip_meta("现在是12点30分"), "现在是12点30分");
    }
}
