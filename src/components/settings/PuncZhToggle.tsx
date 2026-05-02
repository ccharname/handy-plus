import React, { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { listen } from "@tauri-apps/api/event";
import { CheckCircle2, AlertCircle, XCircle } from "lucide-react";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { Button } from "../ui/Button";
import { useSettings } from "../../hooks/useSettings";
import { commands } from "../../bindings";

interface PuncZhToggleProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

type PuncModelState =
  | { kind: "checking" }
  | { kind: "ready" }
  | { kind: "missing" }
  | {
      kind: "downloading";
      downloaded: number;
      total: number;
      percentage: number;
    }
  | { kind: "failed"; error: string };

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export const PuncZhToggle: React.FC<PuncZhToggleProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("punc_zh_enabled") ?? true;

    const [modelState, setModelState] = useState<PuncModelState>({
      kind: "checking",
    });

    const checkDownloaded = useCallback(async () => {
      try {
        const downloaded = await commands.isPuncDownloaded();
        setModelState({ kind: downloaded ? "ready" : "missing" });
      } catch {
        setModelState({ kind: "missing" });
      }
    }, []);

    // Check on mount
    useEffect(() => {
      checkDownloaded();
    }, [checkDownloaded]);

    // Listen for download events
    useEffect(() => {
      const progressPromise = listen<{
        downloaded: number;
        total: number;
        percentage: number;
      }>("punc-download-progress", (e) => {
        const { downloaded, total, percentage } = e.payload;
        setModelState({ kind: "downloading", downloaded, total, percentage });
      });

      const failedPromise = listen<{ error: string }>(
        "punc-download-failed",
        (e) => {
          setModelState({ kind: "failed", error: e.payload.error });
        },
      );

      return () => {
        progressPromise.then((fn) => fn());
        failedPromise.then((fn) => fn());
      };
    }, []);

    const handleDownload = useCallback(async () => {
      setModelState({
        kind: "downloading",
        downloaded: 0,
        total: 0,
        percentage: 0,
      });
      try {
        const result = await commands.downloadPuncModel();
        if (result.status === "ok") {
          // Re-verify after download completes
          await checkDownloaded();
        } else {
          setModelState({ kind: "failed", error: result.error });
        }
      } catch (err) {
        setModelState({
          kind: "failed",
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }, [checkDownloaded]);

    return (
      <div>
        <ToggleSwitch
          checked={enabled}
          onChange={(value) => updateSetting("punc_zh_enabled", value)}
          isUpdating={isUpdating("punc_zh_enabled")}
          label={t("settings.advanced.puncZh.label")}
          description={t("settings.advanced.puncZh.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        />

        {enabled && (
          <div className="px-4 pb-2 pt-1">
            {modelState.kind === "checking" && (
              <p className="text-xs text-text/50">
                {t("common.loading")}
              </p>
            )}

            {modelState.kind === "ready" && (
              <div className="flex items-center gap-1.5 text-xs text-green-500">
                <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
                <span>{t("settings.advanced.puncZh.modelReady")}</span>
              </div>
            )}

            {modelState.kind === "missing" && (
              <div className="flex items-center gap-3 flex-wrap">
                <div className="flex items-center gap-1.5 text-xs text-yellow-500">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>{t("settings.advanced.puncZh.modelMissing")}</span>
                </div>
                <Button size="sm" variant="primary-soft" onClick={handleDownload}>
                  {t("settings.advanced.puncZh.download")}
                </Button>
              </div>
            )}

            {modelState.kind === "downloading" && (
              <div className="space-y-1">
                <p className="text-xs text-text/60">
                  {t("settings.advanced.puncZh.modelDownloading")}
                </p>
                <div className="flex items-center gap-3">
                  <progress
                    value={modelState.percentage}
                    max={100}
                    className="w-40 h-1.5 [&::-webkit-progress-bar]:rounded-full [&::-webkit-progress-bar]:bg-mid-gray/20 [&::-webkit-progress-value]:rounded-full [&::-webkit-progress-value]:bg-logo-primary"
                  />
                  <span className="text-xs text-text/50 tabular-nums">
                    {t("settings.advanced.puncZh.progressText", {
                      percentage: Math.round(modelState.percentage),
                      downloaded: formatBytes(modelState.downloaded),
                      total:
                        modelState.total > 0
                          ? formatBytes(modelState.total)
                          : "…",
                    })}
                  </span>
                </div>
              </div>
            )}

            {modelState.kind === "failed" && (
              <div className="flex items-center gap-3 flex-wrap">
                <div className="flex items-center gap-1.5 text-xs text-red-500">
                  <XCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>
                    {t("settings.advanced.puncZh.modelFailed", {
                      error: modelState.error,
                    })}
                  </span>
                </div>
                <Button size="sm" variant="secondary" onClick={handleDownload}>
                  {t("settings.advanced.puncZh.retry")}
                </Button>
              </div>
            )}
          </div>
        )}
      </div>
    );
  },
);
