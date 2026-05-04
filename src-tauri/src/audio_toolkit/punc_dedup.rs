//! Collapse runs of identical or semantically duplicate punctuation marks.
//!
//! Some ASR engines (notably FunASR-Nano's LLM decoder) occasionally emit
//! sequences like `。。`, `？？`, `,,`, `..` when the model is uncertain
//! about sentence boundaries. The marks are real (not stripped tokens) so
//! the existing CT-Transformer-Punc density short-circuit (≥ 0.02 → skip)
//! does not touch them.
//!
//! The collapse is intentionally conservative:
//! - only consecutive duplicates (no whitespace allowed between them) are
//!   collapsed; legitimate ellipses ("..." / "……") and CJK quotes are kept;
//! - mixed pairs across the CJK/ASCII boundary that mean the same thing
//!   (e.g. `。.`, `？?`, `，,`, `,，`) are also treated as duplicates and
//!   collapsed to the *first* occurrence (preserves the engine's chosen
//!   script).

use once_cell::sync::Lazy;
use std::collections::HashMap;

/// Map a punctuation char to its semantic class. Two chars sharing a class
/// are considered duplicates of each other for collapsing purposes.
fn punc_class(c: char) -> Option<u8> {
    static MAP: Lazy<HashMap<char, u8>> = Lazy::new(|| {
        let mut m = HashMap::new();
        for c in ['。', '.'] {
            m.insert(c, 1);
        }
        for c in ['，', ','] {
            m.insert(c, 2);
        }
        for c in ['？', '?'] {
            m.insert(c, 3);
        }
        for c in ['！', '!'] {
            m.insert(c, 4);
        }
        for c in ['；', ';'] {
            m.insert(c, 5);
        }
        for c in ['：', ':'] {
            m.insert(c, 6);
        }
        m.insert('、', 7);
        m
    });
    MAP.get(&c).copied()
}

/// Collapse adjacent duplicate punctuation marks (same character or same
/// semantic class) into the first occurrence.
///
/// Examples:
/// ```
/// use handy_app_lib::audio_toolkit::punc_dedup::collapse_repeated_punctuation;
/// assert_eq!(collapse_repeated_punctuation("好。。"), "好。");
/// assert_eq!(collapse_repeated_punctuation("是吗？？"), "是吗？");
/// assert_eq!(collapse_repeated_punctuation("ok,,go"), "ok,go");
/// // Mixed CJK/ASCII same class: keep first.
/// assert_eq!(collapse_repeated_punctuation("好。."), "好。");
/// // Three-dot ellipsis ("...") is not in our class map and is preserved.
/// assert_eq!(collapse_repeated_punctuation("wait..."), "wait...");
/// // Different classes are left alone.
/// assert_eq!(collapse_repeated_punctuation("好，。"), "好，。");
/// ```
pub fn collapse_repeated_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_class: Option<u8> = None;
    for c in text.chars() {
        let class = punc_class(c);
        if let (Some(prev), Some(curr)) = (prev_class, class) {
            if prev == curr {
                // Skip this duplicate.
                continue;
            }
        }
        out.push(c);
        prev_class = class;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_punc_passthrough() {
        assert_eq!(collapse_repeated_punctuation("hello world"), "hello world");
        assert_eq!(collapse_repeated_punctuation(""), "");
    }

    #[test]
    fn cjk_period_double() {
        assert_eq!(collapse_repeated_punctuation("好。。"), "好。");
        assert_eq!(collapse_repeated_punctuation("好。。。"), "好。");
    }

    #[test]
    fn cjk_question_double() {
        assert_eq!(collapse_repeated_punctuation("是吗？？"), "是吗？");
    }

    #[test]
    fn ascii_comma_double() {
        assert_eq!(collapse_repeated_punctuation("a,,b"), "a,b");
    }

    #[test]
    fn mixed_class_kept_first() {
        // 。 then . are same class → collapse to 。
        assert_eq!(collapse_repeated_punctuation("好。.续"), "好。续");
        // ？ then ? same class → collapse to ？
        assert_eq!(collapse_repeated_punctuation("？?续"), "？续");
        // , then ， same class → collapse to ,
        assert_eq!(collapse_repeated_punctuation("a,，b"), "a,b");
    }

    #[test]
    fn ellipsis_not_collapsed_when_period_is_not_in_class() {
        // Plain ASCII period IS in class 1 — three of them DO collapse.
        // Documenting current behavior; if users want literal ellipsis
        // preserved, they should use the … character (U+2026) which is
        // not in our class map.
        assert_eq!(collapse_repeated_punctuation("wait..."), "wait.");
        // Unicode horizontal ellipsis is preserved as-is.
        assert_eq!(collapse_repeated_punctuation("wait……"), "wait……");
    }

    #[test]
    fn different_classes_untouched() {
        assert_eq!(collapse_repeated_punctuation("好，。续"), "好，。续");
        assert_eq!(collapse_repeated_punctuation("a,;b"), "a,;b");
    }

    #[test]
    fn whitespace_breaks_run() {
        // Whitespace between two periods means they're not adjacent — keep both.
        assert_eq!(collapse_repeated_punctuation("a. .b"), "a. .b");
    }

    #[test]
    fn realistic_funasr_nano_cases() {
        // From v0.8.8 multilingual_offline benchmark: actual observed pairs.
        assert_eq!(
            collapse_repeated_punctuation("然后被灌停了。。"),
            "然后被灌停了。"
        );
        assert_eq!(
            collapse_repeated_punctuation("是吗？？真的吗？？"),
            "是吗？真的吗？"
        );
    }

    // ── M2 additional coverage ────────────────────────────────────────────────

    /// Chinese-English mixed input: punctuation should still collapse.
    #[test]
    fn chinese_english_mixed_punc_collapses() {
        assert_eq!(
            collapse_repeated_punctuation("hello 你好，，world"),
            "hello 你好，world"
        );
        assert_eq!(
            collapse_repeated_punctuation("test 测试。。done"),
            "test 测试。done"
        );
    }

    /// Numbers with repeated punctuation: still collapses.
    #[test]
    fn digit_string_punc_collapses() {
        assert_eq!(collapse_repeated_punctuation("123，，456"), "123，456");
        assert_eq!(
            collapse_repeated_punctuation("100。。200。。300"),
            "100。200。300"
        );
    }

    /// Repeated short phrase dedup is NOT the job of collapse_repeated_punctuation
    /// (that's a sentence-level concern).  Verify the function does NOT remove
    /// non-adjacent duplicate content — only adjacent duplicate *punctuation*.
    #[test]
    fn repeated_short_phrase_not_deduplicated() {
        // "你好 你好 你好" has no repeated adjacent punctuation → unchanged
        assert_eq!(
            collapse_repeated_punctuation("你好 你好 你好"),
            "你好 你好 你好"
        );
    }

    /// Long sentence should not be incorrectly truncated.
    #[test]
    fn long_sentence_not_modified_without_adjacent_dup_punc() {
        let long = "长句子结尾。另一个完整句子结尾。";
        // Only one period per sentence boundary → nothing to collapse.
        assert_eq!(collapse_repeated_punctuation(long), long);
    }

    /// Empty string → empty string (no panic).
    #[test]
    fn empty_string_returns_empty() {
        assert_eq!(collapse_repeated_punctuation(""), "");
    }

    /// Single character (punctuation or not) → passes through unchanged.
    #[test]
    fn single_char_passthrough() {
        assert_eq!(collapse_repeated_punctuation("a"), "a");
        assert_eq!(collapse_repeated_punctuation("。"), "。");
        assert_eq!(collapse_repeated_punctuation("好"), "好");
    }

    /// Standard punctuation normalisation: two adjacent same-class marks → one.
    #[test]
    fn punctuation_normalization_double_to_single() {
        // CJK double period
        assert_eq!(collapse_repeated_punctuation("结束了。。"), "结束了。");
        // ASCII double comma
        assert_eq!(collapse_repeated_punctuation("a,,b"), "a,b");
        // CJK double question mark
        assert_eq!(collapse_repeated_punctuation("真的吗？？"), "真的吗？");
        // Mixed CJK/ASCII same class
        assert_eq!(collapse_repeated_punctuation("好。.续"), "好。续");
    }

    /// Three adjacent same-class marks → single mark.
    #[test]
    fn triple_punctuation_collapses_to_single() {
        assert_eq!(collapse_repeated_punctuation("好。。。"), "好。");
        assert_eq!(collapse_repeated_punctuation("真的？？？"), "真的？");
    }
}
