import { listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import React, { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import "./RecordingOverlay.css";
import i18n, { syncLanguageFromSettings } from "@/i18n";
import { getLanguageDirection } from "@/lib/utils/rtl";

type OverlayState = "recording" | "transcribing" | "processing";

const RecordingOverlay: React.FC = () => {
  const { t } = useTranslation();
  const [isVisible, setIsVisible] = useState(false);
  const [state, setState] = useState<OverlayState>("recording");
  const [levels, setLevels] = useState<number[]>(Array(16).fill(0));
  const [partialText, setPartialText] = useState<string>("");
  // The macOS overlay panel does not reliably honor `prefers-color-scheme`,
  // so we read the OS theme via Tauri and drive a `data-theme` attribute.
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const smoothedLevelsRef = useRef<number[]>(Array(16).fill(0));
  const direction = getLanguageDirection(i18n.language);

  useEffect(() => {
    const overlayWindow = getCurrentWebviewWindow();
    let unlistenTheme: (() => void) | undefined;

    overlayWindow.theme().then((t) => {
      if (t === "dark" || t === "light") setTheme(t);
    });

    overlayWindow
      .onThemeChanged(({ payload }) => {
        if (payload === "dark" || payload === "light") setTheme(payload);
      })
      .then((unlisten) => {
        unlistenTheme = unlisten;
      });

    return () => {
      if (unlistenTheme) unlistenTheme();
    };
  }, []);

  useEffect(() => {
    const setupEventListeners = async () => {
      const unlistenShow = await listen("show-overlay", async (event) => {
        await syncLanguageFromSettings();
        const overlayState = event.payload as OverlayState;
        setState(overlayState);
        setIsVisible(true);
      });

      const unlistenHide = await listen("hide-overlay", () => {
        setIsVisible(false);
      });

      const unlistenLevel = await listen<number[]>("mic-level", (event) => {
        const newLevels = event.payload as number[];
        const smoothed = smoothedLevelsRef.current.map((prev, i) => {
          const target = newLevels[i] || 0;
          return prev * 0.7 + target * 0.3;
        });
        smoothedLevelsRef.current = smoothed;
        setLevels(smoothed.slice(0, 9));
      });

      const unlistenPartial = await listen<{ text: string }>(
        "transcription-partial",
        (event) => {
          const raw = event.payload.text;
          setPartialText(raw.length > 80 ? raw.slice(raw.length - 80) : raw);
        },
      );

      const unlistenPartialClear = await listen(
        "transcription-partial-clear",
        () => {
          setPartialText("");
        },
      );

      return () => {
        unlistenShow();
        unlistenHide();
        unlistenLevel();
        unlistenPartial();
        unlistenPartialClear();
      };
    };

    setupEventListeners();
  }, []);

  return (
    <div
      dir={direction}
      data-theme={theme}
      className={`recording-overlay ${isVisible ? "fade-in" : ""}`}
    >
      {state === "recording" && !partialText && (
        <div className="bars-container">
          {levels.map((v, i) => (
            <div
              key={i}
              className="bar"
              style={{
                height: `${Math.min(14, 3 + Math.pow(v, 0.7) * 11)}px`,
                transition: "height 60ms ease-out, opacity 120ms ease-out",
                opacity: Math.max(0.25, v * 1.7),
              }}
            />
          ))}
        </div>
      )}
      {state === "recording" && partialText && (
        <div className="partial-text">{partialText}</div>
      )}
      {state === "transcribing" && (
        <div className={partialText ? "partial-text" : "transcribing-text"}>
          {partialText || t("overlay.transcribing")}
        </div>
      )}
      {state === "processing" && (
        <div className="transcribing-text">{t("overlay.processing")}</div>
      )}
    </div>
  );
};

export default RecordingOverlay;
