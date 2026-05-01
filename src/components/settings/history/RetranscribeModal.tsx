import React, { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands, type ModelInfo } from "@/bindings";
import { LANGUAGES } from "@/lib/constants/languages";
import { getTranslatedModelName } from "@/lib/utils/modelTranslation";
import { Button } from "../../ui/Button";

interface RetranscribeModalProps {
  entryId: number;
  onClose: () => void;
  onRetranscribe: (
    id: number,
    overrideModel: string | null,
    overrideLanguage: string | null,
  ) => Promise<void>;
}

export const RetranscribeModal: React.FC<RetranscribeModalProps> = ({
  entryId,
  onClose,
  onRetranscribe,
}) => {
  const { t } = useTranslation();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedLanguage, setSelectedLanguage] = useState<string>("auto");
  const [isRunning, setIsRunning] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    commands.getAvailableModels().then((result) => {
      if (result.status === "ok") {
        setModels(result.data.filter((m) => m.is_downloaded));
      }
    });
  }, []);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isRunning) {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isRunning, onClose]);

  const handleOverlayClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === overlayRef.current && !isRunning) {
        onClose();
      }
    },
    [isRunning, onClose],
  );

  const handleRun = async () => {
    setIsRunning(true);
    try {
      const modelOverride = selectedModel || null;
      const langOverride = selectedLanguage === "auto" ? null : selectedLanguage;
      await onRetranscribe(entryId, modelOverride, langOverride);
      onClose();
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={handleOverlayClick}
    >
      <div className="bg-background border border-mid-gray/30 rounded-lg shadow-xl p-5 w-80 flex flex-col gap-4">
        <h3 className="text-sm font-semibold text-text">
          {t("settings.history.retranscribeModal.title")}
        </h3>

        {/* Model selector */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-text/60 uppercase tracking-wide">
            {t("settings.history.retranscribeModal.modelLabel")}
          </label>
          <select
            className="px-2 py-1.5 text-sm bg-mid-gray/10 border border-mid-gray/60 rounded focus:outline-none focus:ring-1 focus:ring-logo-primary focus:border-logo-primary text-text"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isRunning}
          >
            <option value="">
              {t("settings.history.retranscribeModal.modelPlaceholder")}
            </option>
            {models.length === 0 ? (
              <option disabled>
                {t("settings.history.retranscribeModal.noDownloadedModels")}
              </option>
            ) : (
              models.map((m) => (
                <option key={m.id} value={m.id}>
                  {getTranslatedModelName(m, t)}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Language selector */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-medium text-text/60 uppercase tracking-wide">
            {t("settings.history.retranscribeModal.languageLabel")}
          </label>
          <select
            className="px-2 py-1.5 text-sm bg-mid-gray/10 border border-mid-gray/60 rounded focus:outline-none focus:ring-1 focus:ring-logo-primary focus:border-logo-primary text-text"
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            disabled={isRunning}
          >
            {LANGUAGES.map((lang) => (
              <option key={lang.value} value={lang.value}>
                {lang.value === "auto"
                  ? t("settings.general.language.auto")
                  : lang.label}
              </option>
            ))}
          </select>
        </div>

        {/* Action buttons */}
        <div className="flex justify-end gap-2 pt-1">
          <Button
            variant="secondary"
            size="sm"
            onClick={onClose}
            disabled={isRunning}
          >
            {t("settings.history.retranscribeModal.cancelButton")}
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleRun}
            disabled={isRunning}
          >
            {isRunning
              ? t("settings.history.transcribing")
              : t("settings.history.retranscribeModal.runButton")}
          </Button>
        </div>
      </div>
    </div>
  );
};
