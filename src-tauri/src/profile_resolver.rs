use crate::foreground::{current_foreground_app, ForegroundApp};
use crate::settings::{AppProfile, AppSettings, PasteMethod, ProfileMatcher};

/// The fully resolved settings that the transcription pipeline should use
/// for one recording session. All Option fields from AppSettings are
/// materialised to their concrete types here.
#[derive(Debug, Clone)]
pub struct EffectiveSettings {
    pub selected_language: String,
    /// Global custom_words merged with profile custom_words_extra (deduped).
    pub custom_words: Vec<String>,
    pub post_process_provider_id: String,
    pub post_process_selected_prompt_id: Option<String>,
    pub paste_method: PasteMethod,
    pub append_trailing_space: bool,
    pub auto_submit: bool,
    /// Non-None when power mode matched a profile.
    pub matched_profile_id: Option<String>,
    pub matched_profile_name: Option<String>,
}

/// Resolve the effective settings for the current recording session.
///
/// If power mode is disabled or no foreground app can be determined, the
/// global AppSettings values are returned unchanged.
pub fn resolve_effective_settings(settings: &AppSettings) -> EffectiveSettings {
    let baseline = baseline_from_settings(settings);

    if !settings.power_mode_enabled {
        return baseline;
    }

    let Some(fg) = current_foreground_app() else {
        return baseline;
    };

    // Find the first enabled profile whose matchers match the foreground app.
    if let Some(profile) = settings
        .app_profiles
        .iter()
        .filter(|p| p.enabled)
        .find(|p| profile_matches(p, &fg))
    {
        apply_profile(baseline, profile)
    } else {
        baseline
    }
}

fn profile_matches(profile: &AppProfile, fg: &ForegroundApp) -> bool {
    profile.matchers.iter().any(|m| matcher_hits(m, fg))
}

fn matcher_hits(matcher: &ProfileMatcher, fg: &ForegroundApp) -> bool {
    match matcher {
        ProfileMatcher::Disabled => false,

        ProfileMatcher::BundleId { value } => {
            // On Linux bundle_id is typically None; fall back to process name.
            if let Some(bid) = &fg.bundle_id {
                if value.ends_with('*') {
                    let prefix = &value[..value.len() - 1];
                    bid.starts_with(prefix)
                } else {
                    bid == value
                }
            } else {
                // No bundle id — try process name fallback (Windows / Linux).
                if let Some(pname) = &fg.process_name {
                    // Strip the known ".exe" suffix for Windows comparisons.
                    let pname_base = pname.trim_end_matches(".exe");
                    let val_base = value.trim_end_matches('*');
                    pname_base.eq_ignore_ascii_case(val_base)
                } else {
                    false
                }
            }
        }

        ProfileMatcher::ProcessName { value } => {
            if let Some(pname) = &fg.process_name {
                let pname_base = pname.trim_end_matches(".exe");
                pname_base.eq_ignore_ascii_case(value)
            } else {
                false
            }
        }

        ProfileMatcher::WindowTitleSubstring { value } => {
            if let Some(title) = &fg.window_title {
                title
                    .to_ascii_lowercase()
                    .contains(&value.to_ascii_lowercase())
            } else {
                // Window title is None when Screen Recording permission is absent.
                // Gracefully skip rather than panic.
                false
            }
        }
    }
}

fn apply_profile(mut base: EffectiveSettings, profile: &AppProfile) -> EffectiveSettings {
    base.matched_profile_id = Some(profile.id.clone());
    base.matched_profile_name = Some(profile.name.clone());

    if let Some(lang) = &profile.selected_language {
        base.selected_language = lang.clone();
    }

    // Merge custom_words_extra into the existing list, deduped.
    for word in &profile.custom_words_extra {
        if !base.custom_words.contains(word) {
            base.custom_words.push(word.clone());
        }
    }

    if let Some(provider_id) = &profile.post_process_provider_id {
        base.post_process_provider_id = provider_id.clone();
    }

    if profile.post_process_selected_prompt_id.is_some() {
        base.post_process_selected_prompt_id = profile.post_process_selected_prompt_id.clone();
    }

    if let Some(pm) = profile.paste_method {
        base.paste_method = pm;
    }

    if let Some(ats) = profile.append_trailing_space {
        base.append_trailing_space = ats;
    }

    if let Some(as_) = profile.auto_submit {
        base.auto_submit = as_;
    }

    base
}

fn baseline_from_settings(settings: &AppSettings) -> EffectiveSettings {
    EffectiveSettings {
        selected_language: settings.selected_language.clone(),
        custom_words: settings.custom_words.clone(),
        post_process_provider_id: settings.post_process_provider_id.clone(),
        post_process_selected_prompt_id: settings.post_process_selected_prompt_id.clone(),
        paste_method: settings.paste_method,
        append_trailing_space: settings.append_trailing_space,
        auto_submit: settings.auto_submit,
        matched_profile_id: None,
        matched_profile_name: None,
    }
}
