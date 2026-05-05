//! DeltaComputer — conservative prefix-only delta extraction.
//!
//! The design principle: **never backspace, never overwrite**.  When an ASR
//! engine emits a retroactive edit (rewrites earlier characters), we skip that
//! partial and wait for the next stable one.  This makes streaming output
//! idempotent and visually stable — the cursor only moves forward.
//!
//! # Usage
//!
//! ```rust
//! use crate::output::delta::{Action, DeltaComputer};
//!
//! let mut dc = DeltaComputer::new();
//! assert!(matches!(dc.compute("你好"), Action::Append(_)));
//! assert!(matches!(dc.compute("你好世界"), Action::Append(_)));
//! assert!(matches!(dc.compute("你好"), Action::Skip));  // retroactive rewrite
//! ```

/// Maximum number of pending characters in the delta queue.
/// When a streaming engine emits too many rapid partials, we drop intermediate
/// ones and only emit the most recent delta to avoid overwhelming the target app.
const MAX_QUEUE_CHARS: usize = 20;

/// Result of a `DeltaComputer::compute()` call.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Action {
    /// A pure forward extension was detected; paste this delta immediately.
    Append(String),
    /// The new partial is not a clean prefix extension (retroactive rewrite,
    /// no change, or regression).  The caller should skip and wait.
    Skip,
    /// Final authoritative text arrived; `replace_tail_n` characters already
    /// pasted should be replaced with `with`.  When `replace_tail_n == 0`,
    /// this is a pure append finalization.
    Finalize {
        /// Number of already-pasted characters that need to be replaced.
        /// Zero means the final text is a pure forward extension.
        replace_tail_n: usize,
        /// The replacement text (may be longer or shorter than the replaced tail).
        with: String,
    },
}

/// Stateful delta computer for streaming ASR output.
///
/// Tracks the cumulative text already emitted to the target application and
/// computes the minimal forward-only delta for each new partial.
#[derive(Debug, Default)]
pub struct DeltaComputer {
    /// The cumulative text that has already been pasted to the target app.
    last_emitted: String,
    /// Internal queue depth (chars) of pending but not-yet-emitted deltas.
    /// Used for backpressure: if many rapid partials arrive before the system
    /// can paste them, older intermediate ones are dropped.
    pending_chars: usize,
}

impl DeltaComputer {
    /// Create a fresh, zeroed-out computer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a new streaming partial and compute the action to take.
    ///
    /// Conservative policy:
    /// - If `new_partial` strictly extends `last_emitted`, return `Append(delta)`.
    /// - If `new_partial == last_emitted` (no change), return `Skip`.
    /// - If `new_partial` regresses or rewrites, return `Skip`.
    ///
    /// Backpressure: if `pending_chars` exceeds [`MAX_QUEUE_CHARS`], the
    /// intermediate partial is dropped (return `Skip`) and only the next partial
    /// that makes it through the threshold will be emitted.
    pub fn compute(&mut self, new_partial: &str) -> Action {
        // Retroactive rewrite or no change — skip.
        if !new_partial.starts_with(self.last_emitted.as_str())
            || new_partial.len() <= self.last_emitted.len()
        {
            return Action::Skip;
        }

        // Extract the new delta (always a valid char boundary because starts_with
        // guarantees we're splitting on a char boundary of the prefix).
        let delta = &new_partial[self.last_emitted.len()..];

        // Backpressure: drop if we already have too many pending chars queued.
        let delta_chars = delta.chars().count();
        if self.pending_chars + delta_chars > MAX_QUEUE_CHARS {
            // Drop intermediate partial; do NOT update last_emitted.
            // The next partial that lands after the backlog clears will catch up.
            return Action::Skip;
        }

        // Commit the delta.
        self.last_emitted.push_str(delta);
        self.pending_chars += delta_chars;

        Action::Append(delta.to_string())
    }

    /// Acknowledge that `chars_consumed` characters have been successfully
    /// pasted and are no longer pending.  Call this after each successful
    /// `sink.append()` to keep the backpressure counter accurate.
    pub fn ack(&mut self, chars_consumed: usize) {
        self.pending_chars = self.pending_chars.saturating_sub(chars_consumed);
    }

    /// Produce the finalization action for the authoritative final transcript.
    ///
    /// If `final_text` is a clean extension of what we've already emitted,
    /// `replace_tail_n = 0` and `with` is the remaining suffix — a pure append.
    ///
    /// If `final_text` diverges (post-processing added punctuation, corrected a
    /// word, etc.), `replace_tail_n > 0` so the caller can issue backspaces or
    /// an equivalent to replace the tail.
    ///
    /// After `finalize()` the internal state is reset so the computer is ready
    /// for the next utterance.
    pub fn finalize(&mut self, final_text: &str) -> Action {
        let action = if final_text.starts_with(self.last_emitted.as_str()) {
            // Clean extension — the final text is a forward extension of what we pasted.
            let suffix = &final_text[self.last_emitted.len()..];
            Action::Finalize {
                replace_tail_n: 0,
                with: suffix.to_string(),
            }
        } else {
            // Divergence — find the longest common prefix to minimise backspace count.
            let common_prefix_len = longest_common_prefix_bytes(&self.last_emitted, final_text);
            let tail_already_pasted = &self.last_emitted[common_prefix_len..];
            let replace_tail_n = tail_already_pasted.chars().count();
            let new_suffix = &final_text[common_prefix_len..];
            Action::Finalize {
                replace_tail_n,
                with: new_suffix.to_string(),
            }
        };

        // Always reset after finalize.
        self.reset();
        action
    }

    /// Reset all internal state.  Call on cancel or at the start of a new
    /// recording session so stale cursor state doesn't bleed into the next run.
    pub fn reset(&mut self) {
        self.last_emitted.clear();
        self.pending_chars = 0;
    }

    /// Read-only access to the last emitted string (useful for observability).
    #[allow(dead_code)]
    pub fn last_emitted(&self) -> &str {
        &self.last_emitted
    }
}

/// Return the byte offset of the longest common prefix between `a` and `b`.
/// Always lands on a valid UTF-8 character boundary.
fn longest_common_prefix_bytes(a: &str, b: &str) -> usize {
    let mut last_char_boundary = 0;
    for ((i, ca), cb) in a.char_indices().zip(b.chars()) {
        if ca != cb {
            break;
        }
        last_char_boundary = i + ca.len_utf8();
    }
    last_char_boundary
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn append(s: &str) -> Action {
        Action::Append(s.to_string())
    }
    fn finalize_clean(with: &str) -> Action {
        Action::Finalize {
            replace_tail_n: 0,
            with: with.to_string(),
        }
    }
    fn finalize_replace(replace_tail_n: usize, with: &str) -> Action {
        Action::Finalize {
            replace_tail_n,
            with: with.to_string(),
        }
    }

    // ── Basic prefix extension ────────────────────────────────────────────────

    #[test]
    fn ascii_prefix_extension() {
        let mut dc = DeltaComputer::new();
        assert_eq!(dc.compute("Hello"), append("Hello"));
        assert_eq!(dc.compute("Hello world"), append(" world"));
    }

    #[test]
    fn cjk_prefix_extension() {
        let mut dc = DeltaComputer::new();
        assert_eq!(dc.compute("你好"), append("你好"));
        assert_eq!(dc.compute("你好世界"), append("世界"));
    }

    #[test]
    fn empty_partial_returns_skip() {
        let mut dc = DeltaComputer::new();
        assert_eq!(dc.compute(""), Action::Skip);
    }

    #[test]
    fn first_partial_from_empty_state() {
        let mut dc = DeltaComputer::new();
        assert_eq!(dc.compute("初始化"), append("初始化"));
    }

    // ── No change and regression ──────────────────────────────────────────────

    #[test]
    fn no_change_returns_skip() {
        let mut dc = DeltaComputer::new();
        dc.compute("你好");
        assert_eq!(dc.compute("你好"), Action::Skip);
    }

    #[test]
    fn regression_shorter_returns_skip() {
        let mut dc = DeltaComputer::new();
        dc.compute("你好世界");
        assert_eq!(dc.compute("你好"), Action::Skip);
    }

    #[test]
    fn retroactive_edit_first_char_returns_skip() {
        let mut dc = DeltaComputer::new();
        dc.compute("你好");
        // First character changed — retroactive rewrite.
        assert_eq!(dc.compute("他好世界"), Action::Skip);
    }

    #[test]
    fn completely_different_returns_skip() {
        let mut dc = DeltaComputer::new();
        dc.compute("Hello there");
        assert_eq!(dc.compute("Something else entirely"), Action::Skip);
    }

    // ── Mixed Chinese + English ───────────────────────────────────────────────

    #[test]
    fn mixed_cjk_ascii_extension() {
        let mut dc = DeltaComputer::new();
        assert_eq!(dc.compute("Hello 你好"), append("Hello 你好"));
        assert_eq!(dc.compute("Hello 你好 world"), append(" world"));
    }

    // ── Emoji multi-byte boundary ─────────────────────────────────────────────

    #[test]
    fn emoji_multi_byte_extension() {
        let mut dc = DeltaComputer::new();
        // "😀" is 4 bytes in UTF-8; ensure we don't slice mid-char.
        assert_eq!(dc.compute("😀"), append("😀"));
        assert_eq!(dc.compute("😀😁"), append("😁"));
    }

    #[test]
    fn emoji_mixed_with_text() {
        let mut dc = DeltaComputer::new();
        assert_eq!(dc.compute("Hi 😀"), append("Hi 😀"));
        assert_eq!(dc.compute("Hi 😀 there"), append(" there"));
    }

    // ── Repeat partial ────────────────────────────────────────────────────────

    #[test]
    fn repeated_identical_partial_skips() {
        let mut dc = DeltaComputer::new();
        dc.compute("你好");
        assert_eq!(dc.compute("你好"), Action::Skip);
        assert_eq!(dc.compute("你好"), Action::Skip);
    }

    // ── Finalize clean extension ──────────────────────────────────────────────

    #[test]
    fn finalize_clean_extension() {
        let mut dc = DeltaComputer::new();
        dc.compute("你好");
        dc.compute("你好世界");
        let act = dc.finalize("你好世界！");
        assert_eq!(act, finalize_clean("！"));
    }

    #[test]
    fn finalize_with_no_prior_partials() {
        let mut dc = DeltaComputer::new();
        let act = dc.finalize("完整句子");
        assert_eq!(act, finalize_clean("完整句子"));
    }

    // ── Finalize with tail replacement ────────────────────────────────────────

    #[test]
    fn finalize_replaces_tail_on_divergence() {
        let mut dc = DeltaComputer::new();
        // We emitted "你好 shijie" but the final ITN corrected to "你好世界"
        dc.compute("你好 shijie");
        // Common prefix = "你好", tail_pasted = " shijie" (7 chars)
        let act = dc.finalize("你好世界");
        // replace_tail_n = chars(" shijie") = 7, with = "世界"
        assert_eq!(act, finalize_replace(7, "世界"));
    }

    // ── Reset flow ────────────────────────────────────────────────────────────

    #[test]
    fn reset_clears_state() {
        let mut dc = DeltaComputer::new();
        dc.compute("你好世界");
        dc.reset();
        // After reset, starting fresh.
        assert_eq!(dc.compute("新的"), append("新的"));
    }

    #[test]
    fn finalize_resets_for_next_utterance() {
        let mut dc = DeltaComputer::new();
        dc.compute("第一句");
        dc.finalize("第一句话");
        // Computer is reset — next utterance starts fresh.
        assert_eq!(dc.compute("第二句"), append("第二句"));
    }

    // ── Backpressure ──────────────────────────────────────────────────────────

    #[test]
    fn backpressure_drops_excess_pending() {
        let mut dc = DeltaComputer::new();
        // Fill up to the limit in one shot.
        let long_delta = "a".repeat(MAX_QUEUE_CHARS);
        let act = dc.compute(&long_delta);
        assert_eq!(act, append(&long_delta));

        // Next partial would exceed the cap — should be dropped.
        let extended = format!("{}{}", long_delta, "x");
        let act2 = dc.compute(&extended);
        assert_eq!(act2, Action::Skip);
    }

    #[test]
    fn backpressure_clears_after_ack() {
        let mut dc = DeltaComputer::new();
        let long_delta = "a".repeat(MAX_QUEUE_CHARS);
        dc.compute(&long_delta);

        // Ack all consumed chars — backpressure released.
        dc.ack(MAX_QUEUE_CHARS);

        // Now a further extension should be accepted.
        let extended = format!("{}{}", long_delta, "b");
        let act = dc.compute(&extended);
        assert_eq!(act, append("b"));
    }

    // ── longest_common_prefix_bytes ───────────────────────────────────────────

    #[test]
    fn common_prefix_ascii() {
        assert_eq!(longest_common_prefix_bytes("Hello world", "Hello there"), 6);
    }

    #[test]
    fn common_prefix_cjk() {
        // "你好" = 6 bytes; diverge at "世" vs "世界".
        // Actually both start with "你好" so common = 6 bytes.
        assert_eq!(
            longest_common_prefix_bytes("你好世", "你好界"),
            "你好".len()
        );
    }

    #[test]
    fn common_prefix_empty() {
        assert_eq!(longest_common_prefix_bytes("abc", "xyz"), 0);
    }
}
