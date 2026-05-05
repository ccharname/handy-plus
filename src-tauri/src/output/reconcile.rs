//! FD-006 M3 — Final-pass divergence reconcile.
//!
//! After chunked streaming inference finishes, the user receives a "live" partial
//! transcript that has been progressively pasted to their cursor during recording.
//! When the recording stops, a full-audio batch inference ("final pass") is run
//! to produce the authoritative transcript.  This module computes the delta
//! between the two and returns an action the caller should apply to bring the
//! on-screen text in sync.
//!
//! ## Logic
//!
//! ```text
//! streamed  = text already on screen (from chunked inference)
//! final     = authoritative result (from batch inference on full audio)
//!
//! if streamed == final:
//!     NoOp
//! elif final.starts_with(streamed):
//!     AppendTail(final[streamed.len()..])   ← clean forward extension
//! else:
//!     common = longest_common_prefix_chars(streamed, final)
//!     backspaces = len(streamed) - len(common)   (char count)
//!     retype = final[common.byte_end..]
//!     BackspaceAndRetype { backspaces, retype }
//! ```
//!
//! ## Character vs byte indexing
//!
//! All length calculations use **char** counts for backspace (each CJK
//! character = 1 keypress) but byte offsets for string slicing (Rust strings
//! are UTF-8).  `longest_common_prefix_chars` returns the common prefix as a
//! `&str` slice anchored at a char boundary so callers can use `.len()` safely.

/// The action returned by [`reconcile_streamed_with_final`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconcileAction {
    /// Both strings are identical — nothing to do.
    NoOp,
    /// `final_text` is a forward extension of `streamed`: paste the tail.
    AppendTail(String),
    /// Divergence: erase `backspaces` chars via Backspace, then type `retype`.
    BackspaceAndRetype {
        /// Number of Backspace key presses required to erase the divergent suffix.
        backspaces: usize,
        /// The text to type after backspacing (the residual part of `final_text`
        /// after the common prefix).
        retype: String,
    },
}

/// Compare `streamed` (already on screen) with `final_text` (authoritative
/// batch-inference result) and return the minimal reconcile action.
///
/// # Arguments
///
/// * `streamed`    — cumulative text already pasted to the target app.
/// * `final_text`  — authoritative transcript from the final-pass batch inference.
///
/// # Guarantees
///
/// - The returned action, when applied, makes the on-screen text equal to
///   `final_text`.
/// - Backspace count is always a **char** count, not a byte count.
/// - The `retype` / `AppendTail` string is always a valid UTF-8 substring of
///   `final_text`.
pub fn reconcile_streamed_with_final(streamed: &str, final_text: &str) -> ReconcileAction {
    // Fast path: identical.
    if streamed == final_text {
        return ReconcileAction::NoOp;
    }

    // Clean forward extension: final is a superset of streamed.
    if let Some(tail) = final_text.strip_prefix(streamed) {
        return ReconcileAction::AppendTail(tail.to_string());
    }

    // Divergence: compute longest common char-level prefix.
    let common = longest_common_prefix_chars(streamed, final_text);
    let common_byte_len = common.len(); // byte length of the shared prefix

    // Chars that must be erased = chars in streamed beyond the common prefix.
    let backspaces = streamed[common_byte_len..].chars().count();

    // Text to type after erasing = final_text beyond the common prefix.
    let retype = final_text[common_byte_len..].to_string();

    ReconcileAction::BackspaceAndRetype { backspaces, retype }
}

/// Return the longest common char-level prefix of `a` and `b` as a `&str`
/// slice into `a` (guaranteed to be at a char boundary in both strings).
///
/// # Example
///
/// ```
/// use handy_lib::output::reconcile::longest_common_prefix_chars;
/// assert_eq!(longest_common_prefix_chars("hello👋", "hello👍"), "hello");
/// assert_eq!(longest_common_prefix_chars("你好世界", "你好宇宙"), "你好");
/// assert_eq!(longest_common_prefix_chars("abc", "xyz"), "");
/// assert_eq!(longest_common_prefix_chars("", "abc"), "");
/// ```
pub fn longest_common_prefix_chars<'a>(a: &'a str, b: &str) -> &'a str {
    let mut byte_idx = 0usize;
    for (a_ch, b_ch) in a.chars().zip(b.chars()) {
        if a_ch == b_ch {
            byte_idx += a_ch.len_utf8();
        } else {
            break;
        }
    }
    &a[..byte_idx]
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Clean prefix-extension: streamed="你好", final="你好世界" → append "世界".
    #[test]
    fn reconcile_clean_prefix_extension() {
        let action = reconcile_streamed_with_final("你好", "你好世界");
        assert_eq!(action, ReconcileAction::AppendTail("世界".to_string()));
    }

    /// Near-divergence (one extra char): streamed="切换一下我", final="切换一下我们".
    /// Since final.starts_with(streamed), this is a clean extension.
    #[test]
    fn reconcile_divergence_near_append() {
        let action = reconcile_streamed_with_final("切换一下我", "切换一下我们");
        // "切换一下我们".starts_with("切换一下我") → AppendTail("们")
        assert_eq!(action, ReconcileAction::AppendTail("们".to_string()));
    }

    /// Retroactive edit: streamed="切换一下", final="切话一下".
    /// common_prefix="切", backspace 3, retype "话一下".
    #[test]
    fn reconcile_retroactive_edit() {
        let action = reconcile_streamed_with_final("切换一下", "切话一下");
        assert_eq!(
            action,
            ReconcileAction::BackspaceAndRetype {
                backspaces: 3, // "换一下" = 3 chars
                retype: "话一下".to_string(),
            }
        );
    }

    /// Empty streamed: no backspace, append all.
    #[test]
    fn reconcile_empty_streamed() {
        let action = reconcile_streamed_with_final("", "完整文本");
        assert_eq!(
            action,
            ReconcileAction::AppendTail("完整文本".to_string())
        );
    }

    /// Identical: no-op.
    #[test]
    fn reconcile_identical() {
        let action = reconcile_streamed_with_final("你好", "你好");
        assert_eq!(action, ReconcileAction::NoOp);
    }

    /// Emoji boundary: common prefix stops at the correct char boundary.
    #[test]
    fn longest_common_prefix_chars_emoji() {
        let prefix = longest_common_prefix_chars("hello👋", "hello👍");
        assert_eq!(prefix, "hello");
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// ASCII-only divergence.
    #[test]
    fn reconcile_ascii_divergence() {
        // streamed="Hello world", final="Hello earth" → common="Hello ", backspace 5, retype "earth"
        let action = reconcile_streamed_with_final("Hello world", "Hello earth");
        assert_eq!(
            action,
            ReconcileAction::BackspaceAndRetype {
                backspaces: 5, // "world" = 5 chars
                retype: "earth".to_string(),
            }
        );
    }

    /// Final is shorter: full replace.
    #[test]
    fn reconcile_final_shorter() {
        // streamed="你好世界", final="你好" → diverges (final doesn't start with streamed?
        // actually "你好".starts_with("你好世界") = false → BackspaceAndRetype
        // common = "你好", backspace 2 (world chars), retype ""
        let action = reconcile_streamed_with_final("你好世界", "你好");
        assert_eq!(
            action,
            ReconcileAction::BackspaceAndRetype {
                backspaces: 2, // "世界" = 2 chars
                retype: String::new(),
            }
        );
    }

    /// Empty both strings → no-op.
    #[test]
    fn reconcile_both_empty() {
        let action = reconcile_streamed_with_final("", "");
        assert_eq!(action, ReconcileAction::NoOp);
    }

    /// Mixed CJK + ASCII.
    #[test]
    fn reconcile_mixed_cjk_ascii() {
        // streamed="Hello你好world", final="Hello你好earth"
        // common="Hello你好", backspace 5, retype "earth"
        let action = reconcile_streamed_with_final("Hello你好world", "Hello你好earth");
        assert_eq!(
            action,
            ReconcileAction::BackspaceAndRetype {
                backspaces: 5, // "world" = 5 ASCII chars
                retype: "earth".to_string(),
            }
        );
    }

    /// longest_common_prefix_chars: empty strings.
    #[test]
    fn longest_common_prefix_chars_both_empty() {
        assert_eq!(longest_common_prefix_chars("", ""), "");
    }

    /// longest_common_prefix_chars: no common chars.
    #[test]
    fn longest_common_prefix_chars_no_common() {
        assert_eq!(longest_common_prefix_chars("abc", "xyz"), "");
    }

    /// longest_common_prefix_chars: CJK.
    #[test]
    fn longest_common_prefix_chars_cjk() {
        assert_eq!(
            longest_common_prefix_chars("你好世界", "你好宇宙"),
            "你好"
        );
    }
}
