use natural::phonetics::soundex;
use once_cell::sync::Lazy;
use regex::Regex;
use strsim::levenshtein;

/// Builds an n-gram string by cleaning and concatenating words
///
/// Strips punctuation from each word, lowercases, and joins without spaces.
/// This allows matching "Charge B" against "ChargeBee".
fn build_ngram(words: &[&str]) -> String {
    words
        .iter()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .collect::<Vec<_>>()
        .concat()
}

/// Finds the best matching custom word for a candidate string
///
/// Uses Levenshtein distance and Soundex phonetic matching to find
/// the best match above the given threshold.
///
/// # Arguments
/// * `candidate` - The cleaned/lowercased candidate string to match
/// * `custom_words` - Original custom words (for returning the replacement)
/// * `custom_words_nospace` - Custom words with spaces removed, lowercased (for comparison)
/// * `threshold` - Maximum similarity score to accept
///
/// # Returns
/// The best matching custom word and its score, if any match was found
fn find_best_match<'a>(
    candidate: &str,
    custom_words: &'a [String],
    custom_words_nospace: &[String],
    threshold: f64,
) -> Option<(&'a String, f64)> {
    if candidate.is_empty() || candidate.len() > 50 {
        return None;
    }

    let mut best_match: Option<&String> = None;
    let mut best_score = f64::MAX;

    for (i, custom_word_nospace) in custom_words_nospace.iter().enumerate() {
        // Skip if lengths are too different (optimization + prevents over-matching)
        // Use percentage-based check: max 25% length difference (prevents n-grams from
        // matching significantly shorter custom words, e.g., "openaigpt" vs "openai")
        let len_diff = (candidate.len() as i32 - custom_word_nospace.len() as i32).abs() as f64;
        let max_len = candidate.len().max(custom_word_nospace.len()) as f64;
        let max_allowed_diff = (max_len * 0.25).max(2.0); // At least 2 chars difference allowed
        if len_diff > max_allowed_diff {
            continue;
        }

        // Calculate Levenshtein distance (normalized by length)
        let levenshtein_dist = levenshtein(candidate, custom_word_nospace);
        let max_len = candidate.len().max(custom_word_nospace.len()) as f64;
        let levenshtein_score = if max_len > 0.0 {
            levenshtein_dist as f64 / max_len
        } else {
            1.0
        };

        // Calculate phonetic similarity using Soundex
        let phonetic_match = soundex(candidate, custom_word_nospace);

        // Combine scores: favor phonetic matches, but also consider string similarity
        let combined_score = if phonetic_match {
            levenshtein_score * 0.3 // Give significant boost to phonetic matches
        } else {
            levenshtein_score
        };

        // Accept if the score is good enough (configurable threshold)
        if combined_score < threshold && combined_score < best_score {
            best_match = Some(&custom_words[i]);
            best_score = combined_score;
        }
    }

    best_match.map(|m| (m, best_score))
}

/// Applies custom word corrections to transcribed text using fuzzy matching
///
/// This function corrects words in the input text by finding the best matches
/// from a list of custom words using a combination of:
/// - Levenshtein distance for string similarity
/// - Soundex phonetic matching for pronunciation similarity
/// - N-gram matching for multi-word speech artifacts (e.g., "Charge B" -> "ChargeBee")
///
/// # Arguments
/// * `text` - The input text to correct
/// * `custom_words` - List of custom words to match against
/// * `threshold` - Maximum similarity score to accept (0.0 = exact match, 1.0 = any match)
///
/// # Returns
/// The corrected text with custom words applied
pub fn apply_custom_words(text: &str, custom_words: &[String], threshold: f64) -> String {
    if custom_words.is_empty() || text.trim().is_empty() {
        return text.to_string();
    }

    // Pre-compute lowercase versions to avoid repeated allocations
    let custom_words_lower: Vec<String> = custom_words.iter().map(|w| w.to_lowercase()).collect();

    // Pre-compute versions with spaces removed for n-gram comparison
    let custom_words_nospace: Vec<String> = custom_words_lower
        .iter()
        .map(|w| w.replace(' ', ""))
        .collect();

    // Precompute the maximum custom word length (in bytes) to skip n-grams that
    // are already too long to match any custom word. This avoids calling
    // find_best_match (which iterates all M words) for n-grams that will always
    // exceed the 25%-length-difference guard.
    let max_custom_len = custom_words_nospace
        .iter()
        .map(|w| w.len())
        .max()
        .unwrap_or(0);

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let mut matched = false;

        // Try n-grams from longest (3) to shortest (1) - greedy matching
        for n in (1..=3).rev() {
            if i + n > words.len() {
                continue;
            }

            let ngram_words = &words[i..i + n];
            let ngram = build_ngram(ngram_words);

            // Quick guard: if the ngram is much longer than the longest custom
            // word it can never match — skip the full O(M) find_best_match scan.
            // The threshold inside find_best_match uses 25% length tolerance.
            let max_tolerated = (max_custom_len as f64 * 1.35).ceil() as usize + 2;
            if ngram.len() > max_tolerated.max(51) {
                // Also handles the find_best_match internal >50 guard.
                continue;
            }

            if let Some((replacement, _score)) =
                find_best_match(&ngram, custom_words, &custom_words_nospace, threshold)
            {
                // Extract punctuation from first and last words of the n-gram
                let (prefix, _) = extract_punctuation(ngram_words[0]);
                let (_, suffix) = extract_punctuation(ngram_words[n - 1]);

                // Preserve case from first word
                let corrected = preserve_case_pattern(ngram_words[0], replacement);

                result.push(format!("{}{}{}", prefix, corrected, suffix));
                i += n;
                matched = true;
                break;
            }
        }

        if !matched {
            result.push(words[i].to_string());
            i += 1;
        }
    }

    result.join(" ")
}

/// Preserves the case pattern of the original word when applying a replacement
fn preserve_case_pattern(original: &str, replacement: &str) -> String {
    if original.chars().all(|c| c.is_uppercase()) {
        replacement.to_uppercase()
    } else if original.chars().next().is_some_and(|c| c.is_uppercase()) {
        let mut chars: Vec<char> = replacement.chars().collect();
        if let Some(first_char) = chars.get_mut(0) {
            *first_char = first_char.to_uppercase().next().unwrap_or(*first_char);
        }
        chars.into_iter().collect()
    } else {
        replacement.to_string()
    }
}

/// Extracts punctuation prefix and suffix from a word
fn extract_punctuation(word: &str) -> (&str, &str) {
    let prefix_end = word.chars().take_while(|c| !c.is_alphanumeric()).count();
    let suffix_start = word
        .char_indices()
        .rev()
        .take_while(|(_, c)| !c.is_alphanumeric())
        .count();

    let prefix = if prefix_end > 0 {
        &word[..prefix_end]
    } else {
        ""
    };

    let suffix = if suffix_start > 0 {
        &word[word.len() - suffix_start..]
    } else {
        ""
    };

    (prefix, suffix)
}

/// Apply phonetic alias substitutions: for each canonical word in `aliases`,
/// replace every occurrence of any alias in `text` with the canonical form.
/// Case-sensitive substring match (not fuzzy, not word-boundary).
/// Longer aliases are tried first to avoid prefix collisions.
pub fn apply_word_aliases(
    text: &str,
    aliases: &std::collections::HashMap<String, Vec<String>>,
) -> String {
    if aliases.is_empty() {
        return text.to_string();
    }
    // Flatten + sort by alias length DESC so "an thro pic" matches before "an"
    let mut pairs: Vec<(&str, &str)> = aliases
        .iter()
        .flat_map(|(canonical, alts)| alts.iter().map(move |a| (a.as_str(), canonical.as_str())))
        .filter(|(a, _)| !a.is_empty())
        .collect();
    pairs.sort_by_key(|(a, _)| std::cmp::Reverse(a.chars().count()));

    let mut out = text.to_string();
    for (alias, canonical) in pairs {
        out = out.replace(alias, canonical);
    }
    out
}

/// Returns filler words appropriate for the given language code.
///
/// Some words like "um" and "ha" are real words in certain languages
/// (e.g., Portuguese "um" = "a/an", Spanish "ha" = "has"), so we only
/// include them as fillers for languages where they are truly fillers.
fn get_filler_words_for_language(lang: &str) -> &'static [&'static str] {
    let base_lang = lang.split(&['-', '_'][..]).next().unwrap_or(lang);

    match base_lang {
        "en" => &[
            "uh", "um", "uhm", "umm", "uhh", "uhhh", "ah", "hmm", "hm", "mmm", "mm", "mh", "eh",
            "ehh", "ha",
        ],
        "es" => &["ehm", "mmm", "hmm", "hm"],
        "pt" => &["ahm", "hmm", "mmm", "hm"],
        "fr" => &["euh", "hmm", "hm", "mmm"],
        "de" => &["äh", "ähm", "hmm", "hm", "mmm"],
        "it" => &["ehm", "hmm", "mmm", "hm"],
        "cs" => &["ehm", "hmm", "mmm", "hm"],
        "pl" => &["hmm", "mmm", "hm"],
        "tr" => &["hmm", "mmm", "hm"],
        "ru" => &["хм", "ммм", "hmm", "mmm"],
        "uk" => &["хм", "ммм", "hmm", "mmm"],
        "ar" => &["hmm", "mmm"],
        "ja" => &["hmm", "mmm"],
        "ko" => &["hmm", "mmm"],
        "vi" => &["hmm", "mmm", "hm"],
        // Chinese fillers: include both 拼音/英文 (hmm, emm, en) and the most
        // common CJK monosyllabic interjections used in spontaneous speech.
        // Note: regex `\b` boundaries fire reliably only when a filler is
        // adjacent to ASCII / punctuation, so middle-of-sentence single-char
        // fillers may slip through. That is acceptable given the alternative
        // (no filtering at all). Words like 那个/这个 are intentionally NOT
        // included here because they often carry meaning ("that one").
        "zh" => &[
            "hmm", "mmm", "emm", "uhm", "umm", "en", "嗯", "啊", "呃", "呢", "哦", "诶", "唉", "哎",
        ],
        // Conservative universal fallback (no "um", "eh", "ha")
        _ => &[
            "uh", "uhm", "umm", "uhh", "uhhh", "ah", "hmm", "hm", "mmm", "mm", "mh", "ehh",
        ],
    }
}

static MULTI_SPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

/// Collapses CJK n-gram repetitions (3+ consecutive identical 1-3 char
/// blocks) to a single instance. Handles ASR repetition hallucinations
/// in CJK-only text where there's no whitespace to split on.
///
/// Examples:
///   "扛着扛着扛着忙了两天" → "扛着忙了两天"   (bigram "扛着" × 3)
///   "我我我我吗"               → "我吗"           (unigram "我" × 4)
///   "走走停停"                 → "走走停停"       (only 2 reps each, kept)
///   "看看"                     → "看看"           (verb reduplication, 2 reps kept)
///   "你好世界"                 → "你好世界"       (no repetition)
///
/// Skips ASCII codepoints — those are handled by `collapse_stutters` which
/// is whitespace-aware. Algorithm: scan with shrinking n-gram window
/// (largest first: 3, 2, 1). At each position try to find the longest
/// n-gram that repeats ≥ 3 times immediately, collapse to 1 instance,
/// advance the cursor past the run.
fn collapse_cjk_repetitions(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len());
    let mut i = 0;

    while i < chars.len() {
        // Anchor must be a CJK codepoint to attempt repetition collapse.
        // ASCII / latin / digits are left alone so collapse_stutters can
        // handle them via whitespace tokenisation.
        if !is_cjk_char(chars[i]) {
            result.push(chars[i]);
            i += 1;
            continue;
        }

        let mut collapsed = false;
        // Try n-gram lengths 3, 2, 1 (longest first wins).
        for ngram_len in (1..=3).rev() {
            if i + ngram_len * 3 > chars.len() {
                continue;
            }
            // The candidate n-gram must be all CJK to anchor a repetition.
            let ngram: &[char] = &chars[i..i + ngram_len];
            if !ngram.iter().all(|&c| is_cjk_char(c)) {
                continue;
            }
            // Count consecutive matches (including the anchor).
            let mut count = 1;
            while i + ngram_len * (count + 1) <= chars.len()
                && chars[i + ngram_len * count..i + ngram_len * (count + 1)] == *ngram
            {
                count += 1;
            }
            if count >= 3 {
                // Emit one instance, skip the rest of the run.
                for &c in ngram {
                    result.push(c);
                }
                i += ngram_len * count;
                collapsed = true;
                break;
            }
        }

        if !collapsed {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// True for CJK Unified Ideographs (U+4E00..U+9FFF) and the major
/// extension blocks. Excludes punctuation, kana, hangul.
#[inline]
fn is_cjk_char(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF |    // CJK Unified Ideographs
        0x3400..=0x4DBF |    // CJK Extension A
        0x20000..=0x2A6DF |  // CJK Extension B
        0xF900..=0xFAFF      // CJK Compatibility Ideographs
    )
}

/// Collapses repeated words (3+ repetitions) to a single instance.
/// E.g., "wh wh wh wh" -> "wh", "I I I I" -> "I"
fn collapse_stutters(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        if word_lower.chars().all(|c| c.is_alphabetic()) {
            // Count consecutive repetitions (case-insensitive)
            let mut count = 1;
            while i + count < words.len() && words[i + count].to_lowercase() == word_lower {
                count += 1;
            }

            // If 3+ repetitions, collapse to single instance
            if count >= 3 {
                result.push(word);
                i += count;
            } else {
                result.push(word);
                i += 1;
            }
        } else {
            result.push(word);
            i += 1;
        }
    }

    result.join(" ")
}

/// Filters transcription output by removing filler words and stutter artifacts.
///
/// This function cleans up raw transcription text by:
/// 1. Removing filler words based on the app language (or custom list)
/// 2. Collapsing repeated word stutters (e.g., "wh wh wh" -> "wh")
/// 3. Cleaning up excess whitespace
///
/// # Arguments
/// * `text` - The raw transcription text to filter
/// * `lang` - The app language code (e.g., "en", "pt-BR") used to select filler words
/// * `custom_filler_words` - Optional user-provided filler word list. `Some(vec)` overrides
///   language defaults; `Some(empty vec)` disables filtering; `None` uses language defaults.
///
/// # Returns
/// The filtered text with filler words and stutters removed
pub fn filter_transcription_output(
    text: &str,
    lang: &str,
    custom_filler_words: &Option<Vec<String>>,
) -> String {
    let mut filtered = text.to_string();

    // Source list (custom override or language default).
    let filler_words: Vec<String> = match custom_filler_words {
        Some(words) => words.clone(),
        None => get_filler_words_for_language(lang)
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };

    // Split into ASCII fillers (Latin / Cyrillic etc — \b regex works) and
    // CJK fillers (single CJK char fillers like 嗯 / 呃 — \b never fires in
    // CJK runs so we strip them by direct char substitution instead).
    let mut ascii_patterns: Vec<Regex> = Vec::new();
    let mut cjk_fillers: Vec<&str> = Vec::new();
    for word in &filler_words {
        let is_cjk_filler = !word.is_empty()
            && word.chars().all(is_cjk_char);
        if is_cjk_filler {
            cjk_fillers.push(word.as_str());
        } else if let Ok(rx) = Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))) {
            ascii_patterns.push(rx);
        }
    }

    // Remove ASCII fillers via word-boundary regex.
    for pattern in &ascii_patterns {
        filtered = pattern.replace_all(&filtered, "").to_string();
    }

    // Remove CJK fillers via raw substitution. Order matters slightly — strip
    // longer fillers first so e.g. a hypothetical "啊呀" pattern would beat
    // single "啊". Current default zh list is all single-char, so stable.
    let mut cjk_sorted = cjk_fillers.clone();
    cjk_sorted.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    for word in &cjk_sorted {
        // Also swallow a trailing comma/period after the filler so we don't
        // leave "，，" doubles after stripping a "嗯，" pattern.
        let with_comma = format!("{},", word);
        let with_zh_comma = format!("{}，", word);
        let with_period = format!("{}.", word);
        let with_zh_period = format!("{}。", word);
        filtered = filtered
            .replace(&with_comma, "")
            .replace(&with_zh_comma, "")
            .replace(&with_period, "")
            .replace(&with_zh_period, "")
            .replace(word, "");
    }

    // Collapse repeated 1-2 letter words (stutter artifacts like "wh wh wh wh")
    filtered = collapse_stutters(&filtered);

    // Collapse CJK n-gram repetitions (e.g. "扛着扛着扛着" → "扛着") —
    // common Qwen3-ASR-0.6B-8bit hallucination pattern not caught by the
    // whitespace-based collapse_stutters above.
    filtered = collapse_cjk_repetitions(&filtered);

    // Clean up multiple spaces to single space
    filtered = MULTI_SPACE_PATTERN.replace_all(&filtered, " ").to_string();

    // Trim leading/trailing whitespace
    let trimmed = filtered.trim().to_string();

    // Apply Chinese ITN for zh* languages (zh, zh-CN, zh-TW, …)
    let base_lang = lang.split(&['-', '_'][..]).next().unwrap_or(lang);
    if base_lang == "zh" {
        crate::audio_toolkit::itn_zh::normalize(&trimmed)
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_custom_words_exact_match() {
        let text = "hello world";
        let custom_words = vec!["Hello".to_string(), "World".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_apply_custom_words_fuzzy_match() {
        let text = "helo wrold";
        let custom_words = vec!["hello".to_string(), "world".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preserve_case_pattern() {
        assert_eq!(preserve_case_pattern("HELLO", "world"), "WORLD");
        assert_eq!(preserve_case_pattern("Hello", "world"), "World");
        assert_eq!(preserve_case_pattern("hello", "WORLD"), "WORLD");
    }

    #[test]
    fn test_extract_punctuation() {
        assert_eq!(extract_punctuation("hello"), ("", ""));
        assert_eq!(extract_punctuation("!hello?"), ("!", "?"));
        assert_eq!(extract_punctuation("...hello..."), ("...", "..."));
    }

    #[test]
    fn test_empty_custom_words() {
        let text = "hello world";
        let custom_words = vec![];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_filter_filler_words() {
        let text = "So uhm I was thinking uh about this";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "So I was thinking about this");
    }

    #[test]
    fn test_filter_filler_words_case_insensitive() {
        let text = "UHM this is UH a test";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "this is a test");
    }

    #[test]
    fn test_filter_filler_words_with_punctuation() {
        let text = "Well, uhm, I think, uh. that's right";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Well, I think, that's right");
    }

    #[test]
    fn test_filter_cleans_whitespace() {
        let text = "Hello    world   test";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Hello world test");
    }

    #[test]
    fn test_filter_trims() {
        let text = "  Hello world  ";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_filter_combined() {
        let text = "  Uhm, so I was, uh, thinking about this  ";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "so I was, thinking about this");
    }

    #[test]
    fn test_filter_preserves_valid_text() {
        let text = "This is a completely normal sentence.";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "This is a completely normal sentence.");
    }

    #[test]
    fn test_filter_stutter_collapse() {
        let text = "w wh wh wh wh wh wh wh wh wh why";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "w wh why");
    }

    #[test]
    fn test_filter_stutter_short_words() {
        let text = "I I I I think so so so so";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "I think so");
    }

    #[test]
    fn test_filter_stutter_longer_words() {
        let text = "Check data doc doc doc doc documentation.";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "Check data doc documentation.");
    }

    #[test]
    fn test_filter_stutter_mixed_case() {
        let text = "No NO no NO no";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "No");
    }

    #[test]
    fn test_filter_stutter_preserves_two_repetitions() {
        let text = "no no is fine";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "no no is fine");
    }

    #[test]
    fn test_filter_english_removes_um() {
        let text = "um I think um this is good";
        let result = filter_transcription_output(text, "en", &None);
        assert_eq!(result, "I think this is good");
    }

    #[test]
    fn test_filter_portuguese_preserves_um() {
        // "um" means "a/an" in Portuguese
        let text = "um gato bonito";
        let result = filter_transcription_output(text, "pt", &None);
        assert_eq!(result, "um gato bonito");
    }

    #[test]
    fn test_filter_spanish_preserves_ha() {
        // "ha" means "has" in Spanish
        let text = "ha sido un buen día";
        let result = filter_transcription_output(text, "es", &None);
        assert_eq!(result, "ha sido un buen día");
    }

    #[test]
    fn test_filter_language_code_with_region() {
        // "pt-BR" should normalize to "pt"
        let text = "um gato bonito";
        let result = filter_transcription_output(text, "pt-BR", &None);
        assert_eq!(result, "um gato bonito");
    }

    #[test]
    fn test_filter_custom_filler_words_override() {
        let custom = Some(vec!["okay".to_string(), "right".to_string()]);
        let text = "okay so I think right this works";
        let result = filter_transcription_output(text, "en", &custom);
        assert_eq!(result, "so I think this works");
    }

    #[test]
    fn test_filter_custom_filler_words_empty_disables() {
        let custom = Some(vec![]);
        let text = "So uhm I was thinking uh about this";
        let result = filter_transcription_output(text, "en", &custom);
        // No filler words removed since custom list is empty
        assert_eq!(result, "So uhm I was thinking uh about this");
    }

    #[test]
    fn test_filter_unknown_language_uses_fallback() {
        let text = "uh I think uhm this works";
        let result = filter_transcription_output(text, "xx", &None);
        assert_eq!(result, "I think this works");
    }

    #[test]
    fn test_filter_fallback_does_not_remove_um() {
        // Fallback (unknown language) should not remove "um" since it's a real word in some languages
        let text = "um I think this works";
        let result = filter_transcription_output(text, "xx", &None);
        assert_eq!(result, "um I think this works");
    }

    #[test]
    fn test_apply_custom_words_ngram_two_words() {
        let text = "il cui nome è Charge B, che permette";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChargeBee,"));
        assert!(!result.contains("Charge B"));
    }

    #[test]
    fn test_apply_custom_words_ngram_three_words() {
        let text = "use Chat G P T for this";
        let custom_words = vec!["ChatGPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("ChatGPT"));
    }

    #[test]
    fn test_apply_custom_words_prefers_longer_ngram() {
        let text = "Open AI GPT model";
        let custom_words = vec!["OpenAI".to_string(), "GPT".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert_eq!(result, "OpenAI GPT model");
    }

    #[test]
    fn test_apply_custom_words_ngram_preserves_case() {
        let text = "CHARGE B is great";
        let custom_words = vec!["ChargeBee".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("CHARGEBEE"));
    }

    #[test]
    fn test_apply_custom_words_ngram_with_spaces_in_custom() {
        // Custom word with space should also match against split words
        let text = "using Mac Book Pro";
        let custom_words = vec!["MacBook Pro".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        assert!(result.contains("MacBook"));
    }

    // ── apply_word_aliases tests ──────────────────────────────────────────────

    #[test]
    fn test_apply_word_aliases_empty_map() {
        let aliases = std::collections::HashMap::new();
        assert_eq!(
            apply_word_aliases("aobic is cool", &aliases),
            "aobic is cool"
        );
    }

    #[test]
    fn test_apply_word_aliases_basic_and_longer_first() {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert(
            "Anthropic".to_string(),
            vec!["an thro pic".to_string(), "an".to_string()],
        );
        // "an thro pic" must win over the shorter "an" prefix match.
        let result = apply_word_aliases("I work at an thro pic", &aliases);
        assert_eq!(result, "I work at Anthropic");
        // The shorter alias "an" should still fire when it's the only match.
        let result2 = apply_word_aliases("an apple", &aliases);
        assert_eq!(result2, "Anthropic apple");
    }

    #[test]
    fn test_apply_word_aliases_chinese_alias() {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("Obsidian".to_string(), vec!["op店".to_string()]);
        let result = apply_word_aliases("我用op店记笔记", &aliases);
        assert_eq!(result, "我用Obsidian记笔记");
    }

    #[test]
    fn test_apply_word_aliases_case_sensitive() {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("Anthropic".to_string(), vec!["aobic".to_string()]);
        // Exact case match replaces; different case does NOT.
        let result = apply_word_aliases("aobic is great", &aliases);
        assert_eq!(result, "Anthropic is great");
        let result2 = apply_word_aliases("Aobic is great", &aliases);
        // "Aobic" ≠ "aobic" — should be unchanged
        assert_eq!(result2, "Aobic is great");
    }

    #[test]
    fn test_apply_custom_words_trailing_number_not_doubled() {
        // Verify that trailing non-alpha chars (like numbers) aren't double-counted
        // between build_ngram stripping them and extract_punctuation capturing them
        let text = "use GPT4 for this";
        let custom_words = vec!["GPT-4".to_string()];
        let result = apply_custom_words(text, &custom_words, 0.5);
        // Should NOT produce "GPT-44" (double-counting the trailing 4)
        assert!(
            !result.contains("GPT-44"),
            "got double-counted result: {}",
            result
        );
    }

    // ── collapse_cjk_repetitions tests ───────────────────────────────────────

    #[test]
    fn cjk_collapse_bigram_three_reps() {
        // The canonical zheng-reported case: "扛着扛着扛着" → "扛着".
        assert_eq!(
            collapse_cjk_repetitions("扛着扛着扛着忙了两天"),
            "扛着忙了两天"
        );
    }

    #[test]
    fn cjk_collapse_unigram_four_reps() {
        assert_eq!(collapse_cjk_repetitions("我我我我吗"), "我吗");
    }

    #[test]
    fn cjk_keep_two_reps_verb_reduplication() {
        // Chinese AAB-form verb reduplication ("看看", "试试", "想想") must
        // be preserved — only ≥ 3 consecutive reps collapse.
        assert_eq!(collapse_cjk_repetitions("我看看就走"), "我看看就走");
        assert_eq!(collapse_cjk_repetitions("试试这个"), "试试这个");
    }

    #[test]
    fn cjk_keep_two_reps_bigram_phrase() {
        // "走走停停" is two bigrams reduplicated once each, both kept.
        assert_eq!(collapse_cjk_repetitions("走走停停"), "走走停停");
    }

    #[test]
    fn cjk_no_repetition_unchanged() {
        assert_eq!(collapse_cjk_repetitions("你好世界"), "你好世界");
        assert_eq!(collapse_cjk_repetitions("今天天气真好"), "今天天气真好");
    }

    #[test]
    fn cjk_collapse_trigram_three_reps() {
        // Trigram repetition is rarer but the algorithm must catch it.
        assert_eq!(
            collapse_cjk_repetitions("不行了不行了不行了"),
            "不行了"
        );
    }

    #[test]
    fn cjk_mixed_with_ascii_only_cjk_collapsed() {
        // ASCII/Latin runs are skipped — handled by collapse_stutters via
        // whitespace tokenisation. Only CJK n-grams collapse.
        let input = "use Claude Claude Claude 来扛着扛着扛着活";
        let out = collapse_cjk_repetitions(input);
        // ASCII "Claude" repetitions left alone (no whitespace inside this fn);
        // CJK "扛着" × 3 collapses to one.
        assert!(out.contains("Claude Claude Claude"), "ASCII untouched: {}", out);
        assert!(out.contains("扛着活") && !out.contains("扛着扛着"), "CJK collapsed: {}", out);
    }

    #[test]
    fn cjk_short_text_unchanged() {
        assert_eq!(collapse_cjk_repetitions(""), "");
        assert_eq!(collapse_cjk_repetitions("你"), "你");
        assert_eq!(collapse_cjk_repetitions("你好"), "你好");
    }

    #[test]
    fn cjk_punctuation_does_not_anchor() {
        // Chinese punctuation between repetitions must break the run.
        // "好。好。好。" should NOT collapse — the 。 is a sentence boundary.
        let result = collapse_cjk_repetitions("好。好。好。");
        // The unigram "好" appears 3× but not consecutively (broken by 。)
        // → kept as-is.
        assert_eq!(result, "好。好。好。");
    }

    #[test]
    fn cjk_filter_pipeline_integration() {
        // End-to-end through filter_transcription_output for zh.
        let result = filter_transcription_output(
            "扛着扛着扛着两天",
            "zh",
            &Some(vec![]),  // empty custom_filler_words → no filler removal
        );
        assert!(
            result.contains("扛着两天") && !result.contains("扛着扛着"),
            "expected one-instance after pipeline, got: {}",
            result
        );
    }

    // ── CJK filler removal tests (M0 finding 2026-05-07) ─────────────────────

    #[test]
    fn cjk_filler_removed_mid_sentence() {
        // The post-deploy bug: middle-of-sentence "嗯" survives \b regex.
        // Char-substitution path strips it.
        let result = filter_transcription_output(
            "提供了连接跟断开的嗯方式",
            "zh",
            &None,  // None → use default zh filler list (includes 嗯)
        );
        assert!(
            !result.contains('嗯'),
            "expected mid-sentence '嗯' stripped, got: {}",
            result
        );
        assert!(
            result.contains("方式"),
            "must preserve real content: {}",
            result
        );
    }

    #[test]
    fn cjk_filler_removed_with_trailing_comma() {
        // "嗯，" pattern — strip the filler AND the trailing comma so we
        // don't leave a sentence-leading "，".
        let result = filter_transcription_output(
            "嗯，需要在这个快捷方式里面",
            "zh",
            &None,
        );
        assert!(!result.contains('嗯'), "filler should be gone: {}", result);
        // The leading "，" is also gone (post-trim) so the sentence is clean.
        assert!(
            !result.starts_with('，'),
            "leading '，' should be cleaned: {}",
            result
        );
    }

    #[test]
    fn cjk_filler_e_removed() {
        // "呃" same path.
        let result = filter_transcription_output(
            "需要在这个，呃，快捷方式",
            "zh",
            &None,
        );
        assert!(!result.contains('呃'), "filler '呃' should be gone: {}", result);
    }

    #[test]
    fn cjk_filler_preserves_non_filler_chars() {
        // "啊" is a default zh filler. Make sure stripping it doesn't break
        // unrelated text. (Edge case: "第一啊" loses the emphasis — accepted
        // tradeoff per FD-009 default zh list.)
        let result = filter_transcription_output("你好世界", "zh", &None);
        assert_eq!(result, "你好世界");
    }

    #[test]
    fn cjk_filler_list_empty_skips_strip() {
        // Explicit empty custom list = "no filtering" override.
        let result = filter_transcription_output(
            "嗯方式",
            "zh",
            &Some(vec![]),
        );
        assert!(
            result.contains('嗯'),
            "empty custom list must NOT strip: {}",
            result
        );
    }
}
