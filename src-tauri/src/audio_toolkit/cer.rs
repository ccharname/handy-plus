/// Character Error Rate (CER) computation for ASR accuracy benchmarking.
///
/// CER = Levenshtein(ref_chars, hyp_chars) / len(ref_chars)
///
/// Both reference and hypothesis are tokenised by Unicode scalar value (char),
/// so the metric works correctly for CJK text where characters are the natural
/// unit of measure and for ASCII / mixed language strings alike.
/// Compute the Levenshtein edit distance between two char sequences.
///
/// Uses the classic O(m*n) DP matrix.  Suitable for per-utterance corpus
/// entries (each utterance < 500 chars as per benchmark spec).
fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();

    // Fast paths.
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // dp[j] = edit distance between a[0..i] and b[0..j].
    // We only keep two rows (current + previous) to reduce allocations.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] // No edit needed.
            } else {
                1 + prev[j - 1].min(prev[j]).min(curr[j - 1])
                //  substitution    deletion  insertion
            };
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Compute Character Error Rate (CER).
///
/// # Arguments
/// * `reference` — The ground-truth transcript.
/// * `hypothesis` — The ASR output to evaluate.
///
/// # Returns
/// CER as a `f64` in [0.0, ∞).  Values > 1.0 are possible when the hypothesis
/// is much longer than the reference.  Returns `0.0` when both strings are
/// empty, and `1.0` when the reference is empty but the hypothesis is not.
pub fn character_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let ref_chars: Vec<char> = reference.chars().collect();
    let hyp_chars: Vec<char> = hypothesis.chars().collect();

    // Edge cases: empty reference.
    if ref_chars.is_empty() {
        return if hyp_chars.is_empty() { 0.0 } else { 1.0 };
    }

    let distance = levenshtein_chars(&ref_chars, &hyp_chars);
    distance as f64 / ref_chars.len() as f64
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_strings() {
        assert_eq!(character_error_rate("你好世界", "你好世界"), 0.0);
    }

    #[test]
    fn test_half_deleted() {
        // "你好" is deleted → 2 deletions / 4 reference chars = 0.5
        assert_eq!(character_error_rate("你好世界", "你好"), 0.5);
    }

    #[test]
    fn test_one_substitution_ascii() {
        // "abd" vs "abc": 1 substitution / 3 chars ≈ 0.333
        let cer = character_error_rate("abc", "abd");
        let expected = 1.0_f64 / 3.0_f64;
        assert!(
            (cer - expected).abs() < 1e-9,
            "expected {expected}, got {cer}"
        );
    }

    #[test]
    fn test_empty_both() {
        assert_eq!(character_error_rate("", ""), 0.0);
    }

    #[test]
    fn test_empty_reference_nonempty_hypothesis() {
        assert_eq!(character_error_rate("", "hello"), 1.0);
    }

    #[test]
    fn test_empty_hypothesis() {
        // All reference chars deleted → CER = 1.0
        assert_eq!(character_error_rate("hello", ""), 1.0);
    }

    #[test]
    fn test_cer_above_one() {
        // "ab" → "abcde": 3 insertions / 2 ref chars = 1.5
        let cer = character_error_rate("ab", "abcde");
        assert!((cer - 1.5).abs() < 1e-9, "expected 1.5, got {cer}");
    }

    #[test]
    fn test_mixed_zh_en() {
        // "ASR" substituted with "ASr" → 1/5 = 0.2
        let cer = character_error_rate("ASR测试", "ASr测试");
        assert!((cer - 0.2).abs() < 1e-9, "expected 0.2, got {cer}");
    }
}
