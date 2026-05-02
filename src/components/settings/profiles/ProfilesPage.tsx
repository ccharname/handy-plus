import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown, ChevronUp, Plus, Trash2, Copy } from "lucide-react";
import { commands } from "@/bindings";
import type {
  AppProfile,
  AsrPreset,
  ForegroundApp,
  PasteMethod,
  ProfileMatcher,
} from "@/bindings";
import { useSettings } from "../../../hooks/useSettings";
import { useModelStore } from "../../../stores/modelStore";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";
import { Dropdown } from "../../ui/Dropdown";

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

function fgAppLabel(fg: ForegroundApp | null): string {
  if (!fg) return "";
  const parts: string[] = [];
  if (fg.bundle_id) parts.push(fg.bundle_id);
  else if (fg.process_name) parts.push(fg.process_name);
  if (fg.window_title) parts.push(`"${fg.window_title}"`);
  return parts.join(" · ") || "unknown";
}

type TriState = "inherit" | "on" | "off";

function boolToTriState(v: boolean | null | undefined): TriState {
  if (v === null || v === undefined) return "inherit";
  return v ? "on" : "off";
}

function triStateToBool(t: TriState): boolean | null {
  if (t === "inherit") return null;
  return t === "on";
}

// ─────────────────────────────────────────────────────────────────────────────
// MatcherRow
// ─────────────────────────────────────────────────────────────────────────────

interface MatcherRowProps {
  matcher: ProfileMatcher;
  onChange: (m: ProfileMatcher) => void;
  onRemove: () => void;
}

const matcherKindOptions = [
  { value: "bundle_id", label: "Bundle ID (macOS)" },
  { value: "process_name", label: "Process name" },
  { value: "window_title_substring", label: "Window title contains" },
  { value: "disabled", label: "Disabled" },
];

const MatcherRow: React.FC<MatcherRowProps> = ({
  matcher,
  onChange,
  onRemove,
}) => {
  const { t } = useTranslation();
  const kind = matcher.kind;
  const value = "value" in matcher ? matcher.value : "";

  const handleKindChange = (newKind: string) => {
    if (newKind === "disabled") {
      onChange({ kind: "disabled" });
    } else if (newKind === "bundle_id") {
      onChange({ kind: "bundle_id", value: value || "" });
    } else if (newKind === "process_name") {
      onChange({ kind: "process_name", value: value || "" });
    } else if (newKind === "window_title_substring") {
      onChange({ kind: "window_title_substring", value: value || "" });
    }
  };

  const handleValueChange = (newValue: string) => {
    if (kind === "disabled") return;
    onChange({ ...matcher, value: newValue } as ProfileMatcher);
  };

  return (
    <div className="flex gap-2 items-center">
      <div className="shrink-0">
        <Dropdown
          options={matcherKindOptions}
          selectedValue={kind}
          onSelect={handleKindChange}
        />
      </div>
      {kind !== "disabled" && (
        <Input
          type="text"
          value={value}
          onChange={(e) => handleValueChange(e.target.value)}
          placeholder={
            kind === "bundle_id"
              ? "com.example.App"
              : kind === "process_name"
                ? "AppName"
                : "keyword"
          }
          className="flex-1 text-xs"
        />
      )}
      <Button
        variant="danger-ghost"
        size="sm"
        onClick={onRemove}
        title={t("settings.profiles.removeMatcher")}
      >
        <Trash2 size={14} />
      </Button>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// ProfileCard
// ─────────────────────────────────────────────────────────────────────────────

interface ProfileCardProps {
  profile: AppProfile;
  onChange: (updated: AppProfile) => void;
  onDelete: () => void;
  onDuplicate: () => void;
  providers: Array<{ id: string; label: string }>;
  prompts: Array<{ id: string; name: string }>;
  models: Array<{ id: string; name: string }>;
  presets: Array<{ id: string; label: string }>;
}

const ProfileCard: React.FC<ProfileCardProps> = ({
  profile,
  onChange,
  onDelete,
  onDuplicate,
  providers,
  prompts,
  models,
  presets,
}) => {
  const { t } = useTranslation();
  const [overridesOpen, setOverridesOpen] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const update = (partial: Partial<AppProfile>) =>
    onChange({ ...profile, ...partial });

  const handleMatcherChange = (idx: number, m: ProfileMatcher) => {
    const matchers = [...profile.matchers];
    matchers[idx] = m;
    update({ matchers });
  };

  const handleMatcherRemove = (idx: number) => {
    const matchers = profile.matchers.filter((_, i) => i !== idx);
    update({ matchers });
  };

  const handleAddMatcher = () => {
    update({
      matchers: [...profile.matchers, { kind: "bundle_id", value: "" }],
    });
  };

  const handleCustomWordAdd = (word: string) => {
    const trimmed = word.trim();
    if (trimmed && !profile.custom_words_extra.includes(trimmed)) {
      update({ custom_words_extra: [...profile.custom_words_extra, trimmed] });
    }
  };

  const handleCustomWordRemove = (word: string) => {
    update({
      custom_words_extra: profile.custom_words_extra.filter((w) => w !== word),
    });
  };

  const pasteMethodOptions = [
    { value: "__inherit__", label: t("settings.profiles.inheritGlobal") },
    { value: "ctrl_v", label: "Clipboard (Ctrl/Cmd+V)" },
    { value: "direct", label: "Direct" },
    { value: "none", label: "None" },
    { value: "shift_insert", label: "Clipboard (Shift+Insert)" },
    { value: "ctrl_shift_v", label: "Clipboard (Ctrl+Shift+V)" },
  ];

  const triStateOptions = [
    { value: "inherit", label: t("settings.profiles.tristate.inherit") },
    { value: "on", label: t("settings.profiles.tristate.on") },
    { value: "off", label: t("settings.profiles.tristate.off") },
  ];

  const providerOptions = [
    { value: "__inherit__", label: t("settings.profiles.inheritGlobal") },
    ...providers.map((p) => ({ value: p.id, label: p.label })),
  ];

  const promptOptions = [
    { value: "__inherit__", label: t("settings.profiles.inheritGlobal") },
    ...prompts.map((p) => ({ value: p.id, label: p.name })),
  ];

  const modelOptions = [
    { value: "__inherit__", label: t("settings.profiles.inheritGlobal") },
    ...models.map((m) => ({ value: m.id, label: m.name })),
  ];

  const presetOptions = [
    { value: "__inherit__", label: t("settings.profiles.inheritGlobal") },
    ...presets.map((p) => ({ value: p.id, label: p.label })),
  ];

  const [wordInput, setWordInput] = useState("");

  return (
    <div className="rounded-lg border border-mid-gray/20 bg-mid-gray/5 overflow-hidden">
      {/* Card header */}
      <div className="flex items-center gap-3 px-4 py-3">
        {/* Enable toggle */}
        <input
          type="checkbox"
          checked={profile.enabled}
          onChange={(e) => update({ enabled: e.target.checked })}
          className="w-4 h-4 accent-logo-primary"
          title={t("settings.profiles.profileEnabled")}
        />
        {/* Name */}
        <input
          type="text"
          value={profile.name}
          onChange={(e) => update({ name: e.target.value })}
          className="flex-1 bg-transparent text-sm font-medium focus:outline-none border-b border-transparent focus:border-mid-gray/40 pb-0.5"
          placeholder={t("settings.profiles.profileName")}
        />
        {/* Duplicate / Delete buttons */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onDuplicate}
          title={t("settings.profiles.duplicate")}
        >
          <Copy size={14} />
        </Button>
        {confirmingDelete ? (
          <div className="flex gap-1 items-center">
            <span className="text-xs text-red-400">
              {t("settings.profiles.confirmDelete")}
            </span>
            <Button variant="danger" size="sm" onClick={onDelete}>
              {t("common.yes")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setConfirmingDelete(false)}
            >
              {t("common.no")}
            </Button>
          </div>
        ) : (
          <Button
            variant="danger-ghost"
            size="sm"
            onClick={() => setConfirmingDelete(true)}
            title={t("settings.profiles.delete")}
          >
            <Trash2 size={14} />
          </Button>
        )}
      </div>

      {/* Override summary chips — display only fields that are overridden */}
      {(() => {
        const chips: Array<{ icon: string; label: string }> = [];
        if (profile.active_preset_id) {
          const preset = presets.find((p) => p.id === profile.active_preset_id);
          chips.push({
            icon: "🎯",
            label: preset?.label ?? profile.active_preset_id,
          });
        }
        if (profile.selected_model) {
          const model = models.find((m) => m.id === profile.selected_model);
          chips.push({
            icon: "🧠",
            label: model?.name ?? profile.selected_model,
          });
        }
        if (profile.selected_language) {
          chips.push({ icon: "🌐", label: profile.selected_language });
        }
        if (profile.paste_method) {
          chips.push({ icon: "⚡", label: profile.paste_method });
        }
        if (profile.punc_zh_enabled !== null && profile.punc_zh_enabled !== undefined) {
          chips.push({
            icon: "・",
            label: profile.punc_zh_enabled ? "punc on" : "punc off",
          });
        }
        if (profile.append_trailing_space === true) {
          chips.push({ icon: "␣", label: "trailing space" });
        }
        if (profile.auto_submit === true) {
          chips.push({ icon: "↵", label: "auto submit" });
        }
        if (profile.custom_words_extra && profile.custom_words_extra.length > 0) {
          chips.push({
            icon: "⊕",
            label: t("settings.profiles.customWordsCount", {
              count: profile.custom_words_extra.length,
            }),
          });
        }
        if (
          profile.post_process_chain &&
          profile.post_process_chain.length > 0
        ) {
          chips.push({
            icon: "⛓",
            label: t("settings.profiles.chainStepsCount", {
              count: profile.post_process_chain.length,
            }),
          });
        }
        if (chips.length === 0) {
          return (
            <div className="px-4 pb-2 text-xs text-mid-gray/50 italic">
              {t("settings.profiles.noOverrides")}
            </div>
          );
        }
        return (
          <div className="px-4 pb-2 flex flex-wrap gap-1.5">
            {chips.map((c, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-logo-primary/10 text-logo-primary text-[11px]"
              >
                <span>{c.icon}</span>
                <span>{c.label}</span>
              </span>
            ))}
          </div>
        );
      })()}

      {/* Matchers section */}
      <div className="px-4 pb-3 border-t border-mid-gray/10">
        <p className="text-xs font-semibold text-mid-gray/70 mt-2 mb-2 uppercase tracking-wide">
          {t("settings.profiles.matchers")}
        </p>
        <div className="space-y-2">
          {profile.matchers.map((m, idx) => (
            <MatcherRow
              key={idx}
              matcher={m}
              onChange={(updated) => handleMatcherChange(idx, updated)}
              onRemove={() => handleMatcherRemove(idx)}
            />
          ))}
        </div>
        <Button
          variant="secondary"
          size="sm"
          className="mt-2"
          onClick={handleAddMatcher}
        >
          <Plus size={12} className="mr-1" />
          {t("settings.profiles.addMatcher")}
        </Button>
      </div>

      {/* Overrides collapsible */}
      <div className="border-t border-mid-gray/10">
        <button
          className="w-full flex items-center justify-between px-4 py-2 text-xs font-semibold text-mid-gray/70 uppercase tracking-wide hover:bg-mid-gray/10 transition-colors"
          onClick={() => setOverridesOpen(!overridesOpen)}
        >
          {t("settings.profiles.overrides")}
          {overridesOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {overridesOpen && (
          <div className="px-4 pb-4 space-y-3">
            {/* ASR model (requires profile_hot_swap_engine global toggle) */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.selectedModel")}
              </label>
              <Dropdown
                options={modelOptions}
                selectedValue={profile.selected_model ?? "__inherit__"}
                onSelect={(v) =>
                  update({
                    selected_model: v === "__inherit__" ? null : v,
                  })
                }
              />
              <p className="text-[10px] text-mid-gray/50 mt-1">
                {t("settings.profiles.selectedModelHint")}
              </p>
            </div>

            {/* ASR preset override */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.selectedPreset")}
              </label>
              <Dropdown
                options={presetOptions}
                selectedValue={profile.active_preset_id ?? "__inherit__"}
                onSelect={(v) =>
                  update({
                    active_preset_id: v === "__inherit__" ? null : v,
                  })
                }
              />
              <p className="text-[10px] text-mid-gray/50 mt-1">
                {t("settings.profiles.selectedPresetHint")}
              </p>
            </div>

            {/* Chinese punctuation (CT-Punc) override */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.puncZhEnabled")}
              </label>
              <Dropdown
                options={triStateOptions}
                selectedValue={boolToTriState(profile.punc_zh_enabled)}
                onSelect={(v) =>
                  update({
                    punc_zh_enabled: triStateToBool(v as TriState),
                  })
                }
              />
            </div>

            {/* Post-process chain override (read-only chip + hint) */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.postProcessChain")}
              </label>
              <div className="flex items-center gap-2 flex-wrap">
                {profile.post_process_chain == null ? (
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-mid-gray/10 text-xs text-mid-gray/60">
                    {t("settings.profiles.inheritGlobal")}
                  </span>
                ) : profile.post_process_chain.length === 0 ? (
                  <span className="inline-flex items-center px-2 py-0.5 rounded-full bg-yellow-500/20 text-xs text-yellow-400">
                    {t("settings.profiles.postProcessChainDisabled")}
                  </span>
                ) : (
                  profile.post_process_chain.map((id) => (
                    <span
                      key={id}
                      className="inline-flex items-center px-2 py-0.5 rounded-full bg-logo-primary/20 text-xs"
                    >
                      {id}
                    </span>
                  ))
                )}
                {profile.post_process_chain != null && (
                  <button
                    onClick={() => update({ post_process_chain: null })}
                    className="text-xs text-mid-gray/50 hover:text-red-400 transition-colors"
                  >
                    {t("settings.profiles.postProcessChainReset")}
                  </button>
                )}
              </div>
              <p className="text-[10px] text-mid-gray/50 mt-1">
                {t("settings.profiles.postProcessChainHint")}
              </p>
            </div>

            {/* Paste method */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.pasteMethod")}
              </label>
              <Dropdown
                options={pasteMethodOptions}
                selectedValue={profile.paste_method ?? "__inherit__"}
                onSelect={(v) =>
                  update({
                    paste_method:
                      v === "__inherit__" ? null : (v as PasteMethod),
                  })
                }
              />
            </div>

            {/* Append trailing space */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.appendTrailingSpace")}
              </label>
              <Dropdown
                options={triStateOptions}
                selectedValue={boolToTriState(profile.append_trailing_space)}
                onSelect={(v) =>
                  update({
                    append_trailing_space: triStateToBool(v as TriState),
                  })
                }
              />
            </div>

            {/* Auto submit */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.autoSubmit")}
              </label>
              <Dropdown
                options={triStateOptions}
                selectedValue={boolToTriState(profile.auto_submit)}
                onSelect={(v) =>
                  update({ auto_submit: triStateToBool(v as TriState) })
                }
              />
            </div>

            {/* Post-process provider */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.postProcessProvider")}
              </label>
              <Dropdown
                options={providerOptions}
                selectedValue={
                  profile.post_process_provider_id ?? "__inherit__"
                }
                onSelect={(v) =>
                  update({
                    post_process_provider_id: v === "__inherit__" ? null : v,
                  })
                }
              />
            </div>

            {/* Post-process prompt */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.postProcessPrompt")}
              </label>
              <Dropdown
                options={promptOptions}
                selectedValue={
                  profile.post_process_selected_prompt_id ?? "__inherit__"
                }
                onSelect={(v) =>
                  update({
                    post_process_selected_prompt_id:
                      v === "__inherit__" ? null : v,
                  })
                }
              />
            </div>

            {/* Custom words extra */}
            <div>
              <label className="text-xs text-mid-gray/60 mb-1 block">
                {t("settings.profiles.customWordsExtra")}
              </label>
              <div className="flex gap-2 mb-2">
                <Input
                  type="text"
                  value={wordInput}
                  onChange={(e) => setWordInput(e.target.value)}
                  placeholder={t(
                    "settings.profiles.customWordsExtraPlaceholder",
                  )}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleCustomWordAdd(wordInput);
                      setWordInput("");
                    }
                  }}
                  className="flex-1 text-xs"
                />
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    handleCustomWordAdd(wordInput);
                    setWordInput("");
                  }}
                >
                  {t("common.add")}
                </Button>
              </div>
              <div className="flex flex-wrap gap-1">
                {profile.custom_words_extra.map((word) => (
                  <span
                    key={word}
                    className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-logo-primary/20 text-xs"
                  >
                    {word}
                    <button
                      onClick={() => handleCustomWordRemove(word)}
                      className="text-mid-gray/60 hover:text-red-400 transition-colors"
                    >
                      &times;
                    </button>
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// ProfilesPage (main export)
// ─────────────────────────────────────────────────────────────────────────────

export const ProfilesPage: React.FC = () => {
  const { t } = useTranslation();
  const { settings, updateSetting } = useSettings();

  const [foreground, setForeground] = useState<ForegroundApp | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [asrPresets, setAsrPresets] = useState<AsrPreset[]>([]);

  // Poll foreground app every second
  useEffect(() => {
    const poll = async () => {
      try {
        const fg = await commands.getForegroundApp();
        setForeground(fg);
      } catch {
        // ignore — permission not granted or not supported
      }
    };
    poll();
    intervalRef.current = setInterval(poll, 1000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  // Fetch ASR presets once on mount
  useEffect(() => {
    commands.listAsrPresets().then((result) => {
      if (result.status === "ok") {
        setAsrPresets(result.data);
      }
    });
  }, []);

  const profiles: AppProfile[] = settings?.app_profiles ?? [];
  const powerModeEnabled = settings?.power_mode_enabled ?? false;

  const providers = (settings?.post_process_providers ?? []).map((p) => ({
    id: p.id,
    label: p.label,
  }));
  const prompts = (settings?.post_process_prompts ?? []).map((p) => ({
    id: p.id,
    name: p.name,
  }));
  const allModels = useModelStore((s) => s.models);
  const models = allModels
    .filter((m) => m.is_downloaded)
    .map((m) => ({ id: m.id, name: m.name }));

  const presets = asrPresets.map((p) => ({
    id: p.id,
    label: `${p.icon} ${p.name}`,
  }));

  const updateProfiles = useCallback(
    (newProfiles: AppProfile[]) => {
      updateSetting("app_profiles", newProfiles);
    },
    [updateSetting],
  );

  const handleProfileChange = (idx: number, updated: AppProfile) => {
    const newProfiles = [...profiles];
    newProfiles[idx] = updated;
    updateProfiles(newProfiles);
  };

  const handleDelete = async (profileId: string) => {
    try {
      const result = await commands.deleteAppProfile(profileId);
      if (result.status === "ok") {
        updateProfiles(profiles.filter((p) => p.id !== profileId));
      }
    } catch (e) {
      console.error("Failed to delete profile:", e);
    }
  };

  const handleDuplicate = async (profileId: string) => {
    try {
      const result = await commands.duplicateAppProfile(profileId);
      if (result.status === "ok") {
        updateProfiles([...profiles, result.data]);
      }
    } catch (e) {
      console.error("Failed to duplicate profile:", e);
    }
  };

  const handleAdd = async () => {
    try {
      const result = await commands.addAppProfile("New Profile");
      if (result.status === "ok") {
        updateProfiles([...profiles, result.data]);
      }
    } catch (e) {
      console.error("Failed to add profile:", e);
    }
  };

  // Determine which profile is currently matched (for display)
  const matchedProfileName =
    powerModeEnabled && foreground
      ? (() => {
          // Simple JS-side matching for display purposes only
          // (the real matching is authoritative on the Rust side)
          for (const profile of profiles) {
            if (!profile.enabled) continue;
            for (const matcher of profile.matchers) {
              if (matcher.kind === "disabled") continue;
              if (matcher.kind === "bundle_id" && foreground.bundle_id) {
                const v = matcher.value;
                const bid = foreground.bundle_id;
                if (
                  v.endsWith("*") ? bid.startsWith(v.slice(0, -1)) : bid === v
                ) {
                  return profile.name;
                }
              }
              if (matcher.kind === "process_name" && foreground.process_name) {
                if (
                  foreground.process_name
                    .replace(/\.exe$/i, "")
                    .toLowerCase() === matcher.value.toLowerCase()
                ) {
                  return profile.name;
                }
              }
              if (
                matcher.kind === "window_title_substring" &&
                foreground.window_title
              ) {
                if (
                  foreground.window_title
                    .toLowerCase()
                    .includes(matcher.value.toLowerCase())
                ) {
                  return profile.name;
                }
              }
            }
          }
          return null;
        })()
      : null;

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.profiles.title")}>
        {/* Power Mode master toggle */}
        <div className="flex items-center justify-between px-4 py-3">
          <div>
            <p className="text-sm font-medium">
              {t("settings.profiles.enabledLabel")}
            </p>
            <p className="text-xs text-mid-gray/60 mt-0.5 max-w-md">
              {t("settings.profiles.enabledDescription")}
            </p>
          </div>
          <label className="cursor-pointer">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={powerModeEnabled}
              onChange={(e) =>
                updateSetting("power_mode_enabled", e.target.checked)
              }
            />
            <div className="relative w-11 h-6 bg-mid-gray/20 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-logo-primary rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-background-ui"></div>
          </label>
        </div>

        {/* Hot-swap engine master toggle (off by default; costs 1-3s of model load) */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-mid-gray/10">
          <div>
            <p className="text-sm font-medium">
              {t("settings.profiles.hotSwapEngineLabel")}
            </p>
            <p className="text-xs text-mid-gray/60 mt-0.5 max-w-md">
              {t("settings.profiles.hotSwapEngineDescription")}
            </p>
          </div>
          <label className="cursor-pointer">
            <input
              type="checkbox"
              className="sr-only peer"
              checked={settings?.profile_hot_swap_engine ?? false}
              onChange={(e) =>
                updateSetting("profile_hot_swap_engine", e.target.checked)
              }
            />
            <div className="relative w-11 h-6 bg-mid-gray/20 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-logo-primary rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-background-ui"></div>
          </label>
        </div>

        {/* How it works hint banner */}
        <div className="mx-4 mb-2 px-3 py-2 rounded-md bg-logo-primary/10 text-xs text-mid-gray/80">
          {t("settings.profiles.howItWorksHint")}
        </div>

        {/* Foreground app detection status */}
        <div className="px-4 pb-3 flex items-center gap-2 text-xs text-mid-gray/60">
          <span
            className={`w-2 h-2 rounded-full inline-block ${powerModeEnabled ? "bg-green-500" : "bg-mid-gray/30"}`}
          />
          {foreground
            ? t("settings.profiles.detectedNow", {
                app: fgAppLabel(foreground),
              })
            : t("settings.profiles.noForeground")}
          {matchedProfileName && (
            <span className="ml-2 px-2 py-0.5 rounded-full bg-logo-primary/20 text-logo-primary text-xs">
              {t("settings.profiles.matchedNow", { name: matchedProfileName })}
            </span>
          )}
        </div>
      </SettingsGroup>

      {/* Profile list */}
      <div className="space-y-4">
        {profiles.map((profile, idx) => (
          <ProfileCard
            key={profile.id}
            profile={profile}
            onChange={(updated) => handleProfileChange(idx, updated)}
            onDelete={() => handleDelete(profile.id)}
            onDuplicate={() => handleDuplicate(profile.id)}
            providers={providers}
            prompts={prompts}
            models={models}
            presets={presets}
          />
        ))}

        <Button variant="secondary" className="w-full" onClick={handleAdd}>
          <Plus size={14} className="mr-2" />
          {t("settings.profiles.add")}
        </Button>
      </div>
    </div>
  );
};
