import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { open } from "@tauri-apps/plugin-dialog";
import { useSettings } from "../../../hooks/useSettings";
import { SettingsGroup } from "../../ui/SettingsGroup";
import { Button } from "../../ui/Button";
import { Input } from "../../ui/Input";

export const DiarySettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const diaryDir: string = getSetting("diary_dir") ?? "";
  const diaryKeywords: string[] = getSetting("diary_keywords") ?? [];

  const [newKeyword, setNewKeyword] = useState("");

  const handleBrowse = async () => {
    const selected = await open({
      directory: true,
      multiple: false,
      title: t("settings.general.diary.dirLabel"),
    });
    if (selected && typeof selected === "string") {
      await updateSetting("diary_dir", selected);
    }
  };

  const handleDirChange = async (value: string) => {
    await updateSetting("diary_dir", value || null);
  };

  const handleAddKeyword = async () => {
    const kw = newKeyword.trim();
    if (!kw) return;
    if (diaryKeywords.includes(kw)) {
      setNewKeyword("");
      return;
    }
    await updateSetting("diary_keywords", [...diaryKeywords, kw]);
    setNewKeyword("");
  };

  const handleRemoveKeyword = async (keyword: string) => {
    await updateSetting(
      "diary_keywords",
      diaryKeywords.filter((k) => k !== keyword),
    );
  };

  return (
    <SettingsGroup title={t("settings.general.diary.title")}>
      {/* Directory row */}
      <div className="px-4 py-3 space-y-1">
        <p className="text-xs text-mid-gray">
          {t("settings.general.diary.dirDescription")}
        </p>
        <div className="flex gap-2 items-center">
          <Input
            className="flex-1"
            value={diaryDir}
            placeholder={t("settings.general.diary.dirPlaceholder")}
            disabled={isUpdating("diary_dir")}
            onChange={(e) => handleDirChange(e.target.value)}
          />
          <Button
            variant="secondary"
            size="sm"
            onClick={handleBrowse}
            disabled={isUpdating("diary_dir")}
          >
            {t("settings.general.diary.browse")}
          </Button>
        </div>
      </div>

      {/* Keywords row */}
      <div className="px-4 py-3 space-y-2">
        <div>
          <p className="text-sm font-medium">
            {t("settings.general.diary.keywordsLabel")}
          </p>
          <p className="text-xs text-mid-gray">
            {t("settings.general.diary.keywordsDescription")}
          </p>
        </div>

        {/* Chip list */}
        <div className="flex flex-wrap gap-1">
          {diaryKeywords.map((kw) => (
            <span
              key={kw}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-logo-primary/20 text-text"
            >
              {kw}
              <button
                type="button"
                aria-label={t("settings.general.diary.removeKeyword")}
                className="ml-0.5 hover:text-red-400 transition-colors"
                disabled={isUpdating("diary_keywords")}
                onClick={() => handleRemoveKeyword(kw)}
              >
                &times;
              </button>
            </span>
          ))}
        </div>

        {/* Add keyword input */}
        <div className="flex gap-2">
          <Input
            className="flex-1"
            value={newKeyword}
            placeholder={t("settings.general.diary.keywordPlaceholder")}
            disabled={isUpdating("diary_keywords")}
            onChange={(e) => setNewKeyword(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAddKeyword();
              }
            }}
          />
          <Button
            variant="secondary"
            size="sm"
            disabled={!newKeyword.trim() || isUpdating("diary_keywords")}
            onClick={handleAddKeyword}
          >
            {t("settings.general.diary.addKeyword")}
          </Button>
        </div>
      </div>
    </SettingsGroup>
  );
};
