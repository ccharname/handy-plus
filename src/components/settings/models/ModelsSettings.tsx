import React from "react";
import { useTranslation } from "react-i18next";
import { AsrPresetCards } from "./AsrPresetCards";

export const ModelsSettings: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="max-w-3xl w-full mx-auto space-y-4">
      <div className="mb-4">
        <h1 className="text-xl font-semibold mb-2">
          {t("settings.models.title")}
        </h1>
        <p className="text-sm text-text/60">
          {t("settings.models.description")}
        </p>
      </div>

      {/* ASR Preset Cards — quick-switch section */}
      <AsrPresetCards />
    </div>
  );
};
