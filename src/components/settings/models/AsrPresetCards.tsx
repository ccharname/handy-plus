import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { platform } from "@tauri-apps/plugin-os";
import { commands } from "@/bindings";
import type { AsrPreset } from "@/bindings";
import { useSettingsStore } from "@/stores/settingsStore";
import { useModelStore } from "@/stores/modelStore";

export const AsrPresetCards: React.FC = () => {
  const { t } = useTranslation();
  const [presets, setPresets] = useState<AsrPreset[]>([]);
  const [loadingPresets, setLoadingPresets] = useState(true);
  const [switchingPresetId, setSwitchingPresetId] = useState<string | null>(
    null,
  );

  const settings = useSettingsStore((s) => s.settings);
  const applyAsrPreset = useSettingsStore((s) => s.applyAsrPreset);
  const activePresetId = settings?.active_preset_id ?? null;

  const { models, downloadModel } = useModelStore();

  // Determine current platform synchronously (plugin-os platform() is sync)
  const currentPlatform = platform();
  const isMacOS = currentPlatform === "macos";

  useEffect(() => {
    const load = async () => {
      try {
        const result = await commands.listAsrPresets();
        if (result.status === "ok") {
          // Filter out apple_native on non-macOS platforms
          const filtered = result.data.filter(
            (p) => p.id !== "apple_native" || isMacOS,
          );
          setPresets(filtered);
        } else {
          console.error("Failed to load ASR presets:", result.error);
        }
      } catch (err) {
        console.error("Failed to load ASR presets:", err);
      } finally {
        setLoadingPresets(false);
      }
    };
    load();
  }, [isMacOS]);

  const isModelDownloaded = (modelId: string): boolean => {
    const model = models.find((m) => m.id === modelId);
    return model?.is_downloaded ?? false;
  };

  const getPresetName = (preset: AsrPreset): string => {
    const key = `settings.asrPresets.${toCamelId(preset.id)}.name`;
    const translated = t(key);
    // fallback to preset.name if key not translated
    return translated === key ? preset.name : translated;
  };

  const getPresetDescription = (preset: AsrPreset): string => {
    const key = `settings.asrPresets.${toCamelId(preset.id)}.description`;
    const translated = t(key);
    return translated === key ? preset.description : translated;
  };

  const toCamelId = (id: string): string => {
    // "chinese_balanced" -> "chineseBalanced"
    return id.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());
  };

  const handleApply = async (preset: AsrPreset) => {
    if (switchingPresetId !== null) return;
    setSwitchingPresetId(preset.id);
    try {
      if (!isModelDownloaded(preset.model_id)) {
        // Download model first
        await downloadModel(preset.model_id);
      }
      await applyAsrPreset(preset.id);
    } catch (err) {
      console.error("Failed to apply ASR preset:", err);
    } finally {
      setSwitchingPresetId(null);
    }
  };

  if (loadingPresets) {
    return (
      <div className="flex items-center justify-center py-6">
        <div className="w-6 h-6 border-2 border-logo-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (presets.length === 0) {
    return null;
  }

  const isSwitching = switchingPresetId !== null;

  return (
    <div className="mb-6">
      <div className="mb-3">
        <h2 className="text-sm font-medium text-text/60">
          {t("settings.asrPresets.title")}
        </h2>
        <p className="text-xs text-text/40 mt-0.5">
          {t("settings.asrPresets.description")}
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {presets.map((preset) => {
          const isActive = activePresetId === preset.id;
          const isThisSwitching = switchingPresetId === preset.id;
          const downloaded = isModelDownloaded(preset.model_id);

          return (
            <div
              key={preset.id}
              className={`relative flex flex-col p-4 rounded-xl border transition-all ${
                isActive
                  ? "border-logo-primary bg-logo-primary/5"
                  : "border-mid-gray/40 bg-mid-gray/5 hover:border-mid-gray/60"
              }`}
            >
              {/* Active badge */}
              {isActive && (
                <span className="absolute top-2 right-2 text-xs font-semibold text-logo-primary bg-logo-primary/15 px-2 py-0.5 rounded-full">
                  {t("settings.asrPresets.active")}
                </span>
              )}

              {/* Icon + name */}
              <div className="flex items-center gap-2 mb-2">
                <span className="text-2xl leading-none">{preset.icon}</span>
                <span className="text-sm font-semibold truncate">
                  {getPresetName(preset)}
                </span>
              </div>

              {/* Description */}
              <p className="text-xs text-text/50 mb-3 flex-1 leading-relaxed">
                {getPresetDescription(preset)}
              </p>

              {/* Apply button */}
              <button
                type="button"
                disabled={isSwitching || isActive}
                onClick={() => handleApply(preset)}
                className={`w-full py-1.5 px-3 rounded-lg text-xs font-medium transition-colors ${
                  isActive
                    ? "bg-logo-primary/20 text-logo-primary cursor-default"
                    : isSwitching
                      ? "bg-mid-gray/20 text-text/40 cursor-not-allowed"
                      : "bg-logo-primary text-white hover:bg-logo-primary/90 active:scale-95"
                }`}
              >
                {isThisSwitching ? (
                  <span className="flex items-center justify-center gap-1.5">
                    <span className="w-3 h-3 border-2 border-white/60 border-t-transparent rounded-full animate-spin inline-block" />
                    {t("settings.asrPresets.switching")}
                  </span>
                ) : isActive ? (
                  t("settings.asrPresets.active")
                ) : downloaded ? (
                  t("settings.asrPresets.apply")
                ) : (
                  t("settings.asrPresets.applyAndDownload")
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Show "customized" hint when preset was detached */}
      {activePresetId === null && (
        <p className="mt-2 text-xs text-text/40 italic">
          {t("settings.asrPresets.customized")}
        </p>
      )}
    </div>
  );
};
