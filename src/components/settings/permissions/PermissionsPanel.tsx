import { useEffect, useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import { platform } from "@tauri-apps/plugin-os";
import { relaunch } from "@tauri-apps/plugin-process";
import {
  checkAccessibilityPermission,
  requestAccessibilityPermission,
  checkMicrophonePermission,
  requestMicrophonePermission,
} from "tauri-plugin-macos-permissions-api";
import { Check, X, RefreshCw, AlertTriangle } from "lucide-react";
import { commands } from "@/bindings";
import { SettingsGroup } from "../../ui/SettingsGroup";

type PermissionStatus = "unknown" | "granted" | "denied";

interface PermissionsState {
  accessibility: PermissionStatus;
  microphone: PermissionStatus;
  /** true if accessibility is reported granted but enigo init failed (macOS TCC cache stale) */
  enigoFailed: boolean;
}

const StatusIcon: React.FC<{ status: PermissionStatus }> = ({ status }) => {
  if (status === "granted") {
    return <Check className="w-4 h-4 text-emerald-400 shrink-0" />;
  }
  if (status === "denied") {
    return <X className="w-4 h-4 text-red-400 shrink-0" />;
  }
  return <span className="w-4 h-4 rounded-full bg-mid-gray/30 shrink-0 inline-block" />;
};

export const PermissionsPanel: React.FC = () => {
  const { t } = useTranslation();
  const [currentPlatform, setCurrentPlatform] = useState<string | null>(null);
  const [permissions, setPermissions] = useState<PermissionsState>({
    accessibility: "unknown",
    microphone: "unknown",
    enigoFailed: false,
  });
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkPermissions = useCallback(async () => {
    const plat = platform();
    setCurrentPlatform(plat);

    if (plat !== "macos") {
      return;
    }

    try {
      const [accessibilityGranted, microphoneGranted] = await Promise.all([
        checkAccessibilityPermission(),
        checkMicrophonePermission(),
      ]);

      let enigoFailed = false;
      if (accessibilityGranted) {
        try {
          await Promise.all([
            commands.initializeEnigo(),
            commands.initializeShortcuts(),
          ]);
        } catch {
          enigoFailed = true;
        }
      }

      setPermissions({
        accessibility: accessibilityGranted ? "granted" : "denied",
        microphone: microphoneGranted ? "granted" : "denied",
        enigoFailed,
      });
    } catch (e) {
      console.warn("PermissionsPanel: failed to check permissions", e);
    }
  }, []);

  useEffect(() => {
    checkPermissions();
  }, [checkPermissions]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await checkPermissions();
    setIsRefreshing(false);
  };

  const handleRequestAccessibility = async () => {
    try {
      await requestAccessibilityPermission();
    } catch (e) {
      console.warn("Failed to request accessibility permission:", e);
    }
    // Re-check after user interaction
    await checkPermissions();
  };

  const handleRequestMicrophone = async () => {
    try {
      await requestMicrophonePermission();
    } catch (e) {
      console.warn("Failed to request microphone permission:", e);
    }
    await checkPermissions();
  };

  const handleRelaunch = async () => {
    try {
      await relaunch();
    } catch (e) {
      console.error("Failed to relaunch:", e);
    }
  };

  // Only render on macOS
  if (currentPlatform !== null && currentPlatform !== "macos") {
    return null;
  }

  return (
    <SettingsGroup title={t("settings.permissions.title")}>
      {/* Stale TCC cache warning */}
      {permissions.enigoFailed && (
        <div className="px-4 py-3 flex items-start gap-3 bg-yellow-500/10 border-b border-mid-gray/20">
          <AlertTriangle className="w-4 h-4 text-yellow-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-text/80">
              {t("settings.permissions.staleCacheWarning")}
            </p>
            <button
              onClick={handleRelaunch}
              className="mt-2 px-3 py-1.5 text-xs font-medium rounded-md bg-yellow-500/20 hover:bg-yellow-500/30 text-yellow-300 transition-colors"
            >
              {t("settings.permissions.relaunch")}
            </button>
          </div>
        </div>
      )}

      {/* Accessibility row */}
      <div className="px-4 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <StatusIcon status={permissions.accessibility} />
          <div className="min-w-0">
            <p className="text-sm font-medium text-text">
              {t("settings.permissions.accessibility")}
            </p>
            <p className="text-xs text-mid-gray">
              {permissions.accessibility === "granted"
                ? t("settings.permissions.statusGranted")
                : t("settings.permissions.statusDenied")}
            </p>
          </div>
        </div>
        {permissions.accessibility !== "granted" && (
          <button
            onClick={handleRequestAccessibility}
            className="px-3 py-1.5 text-xs font-medium rounded-md bg-logo-primary hover:bg-logo-primary/90 text-white transition-colors shrink-0"
          >
            {t("settings.permissions.authorize")}
          </button>
        )}
      </div>

      {/* Microphone row */}
      <div className="px-4 py-3 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <StatusIcon status={permissions.microphone} />
          <div className="min-w-0">
            <p className="text-sm font-medium text-text">
              {t("settings.permissions.microphone")}
            </p>
            <p className="text-xs text-mid-gray">
              {permissions.microphone === "granted"
                ? t("settings.permissions.statusGranted")
                : t("settings.permissions.statusDenied")}
            </p>
          </div>
        </div>
        {permissions.microphone !== "granted" && (
          <button
            onClick={handleRequestMicrophone}
            className="px-3 py-1.5 text-xs font-medium rounded-md bg-logo-primary hover:bg-logo-primary/90 text-white transition-colors shrink-0"
          >
            {t("settings.permissions.authorize")}
          </button>
        )}
      </div>

      {/* Refresh row */}
      <div className="px-4 py-2 flex justify-end">
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-1.5 text-xs text-mid-gray hover:text-text transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3 h-3 ${isRefreshing ? "animate-spin" : ""}`} />
          {t("settings.permissions.refresh")}
        </button>
      </div>
    </SettingsGroup>
  );
};
