import React from "react";
import { useTranslation } from "react-i18next";
import { ToggleSwitch } from "../ui/ToggleSwitch";
import { useSettings } from "../../hooks/useSettings";

interface PuncZhToggleProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

export const PuncZhToggle: React.FC<PuncZhToggleProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();

    const enabled = getSetting("punc_zh_enabled") ?? true;

    return (
      <ToggleSwitch
        checked={enabled}
        onChange={(value) => updateSetting("punc_zh_enabled", value)}
        isUpdating={isUpdating("punc_zh_enabled")}
        label={t("settings.advanced.puncZh.label")}
        description={t("settings.advanced.puncZh.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
      />
    );
  },
);
