import React from "react";
import { useTranslation } from "react-i18next";
import { CustomWords } from "../CustomWords";
import { SettingsGroup } from "../../ui/SettingsGroup";

export const VocabularySettings: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="max-w-3xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.vocabulary.title")}>
        <CustomWords descriptionMode="tooltip" grouped={true} />
      </SettingsGroup>
    </div>
  );
};
