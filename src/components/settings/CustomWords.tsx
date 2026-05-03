import React, { useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";
import { useSettings } from "../../hooks/useSettings";
import { Button } from "../ui/Button";
import { Input } from "../ui/Input";
import { SettingContainer } from "../ui/SettingContainer";
import { Textarea } from "../ui/Textarea";

interface CustomWordsProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
}

// ---------- Parsing helpers ----------

// Strip only characters that would break HTML rendering / hotwords file format.
// Keep `'` and `"` so brand names like O'Reilly and "AI Lab" survive.
const ILLEGAL_CHARS = /[<>&]/g;
const MAX_WORD_LEN = 80;

function parseRawText(raw: string): string[] {
  return raw
    .split(/[\n,，、；;]+/)
    .map((w) => w.trim().replace(ILLEGAL_CHARS, ""))
    .filter((w) => w.length > 0 && w.length <= MAX_WORD_LEN);
}

function parseInput(raw: string): string[] {
  const trimmed = raw.trim();
  // Try JSON array fallback first
  if (trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      if (
        Array.isArray(parsed) &&
        parsed.every((item) => typeof item === "string")
      ) {
        return parsed
          .map((w: string) => w.trim().replace(ILLEGAL_CHARS, ""))
          .filter((w: string) => w.length > 0 && w.length <= MAX_WORD_LEN);
      }
    } catch {
      // fall through to text parsing
    }
  }
  return parseRawText(raw);
}

interface PreviewResult {
  toAdd: string[];
  skipCount: number;
  invalidCount: number;
}

function computePreview(
  raw: string,
  existing: string[],
  originalInvalidCount?: number,
): PreviewResult {
  const trimmed = raw.trim();
  if (!trimmed) return { toAdd: [], skipCount: 0, invalidCount: 0 };

  // Count raw tokens to derive invalid count
  const allRawTokens = trimmed.startsWith("[")
    ? (() => {
        try {
          const parsed = JSON.parse(trimmed);
          if (Array.isArray(parsed)) return parsed as string[];
        } catch {
          // fall through
        }
        return trimmed.split(/[\n,，、]+/).filter((w) => w.trim());
      })()
    : trimmed.split(/[\n,，、]+/).filter((w) => w.trim());

  const valid = parseInput(raw);
  const invalidCount = Math.max(
    0,
    allRawTokens.length - valid.length + (originalInvalidCount ?? 0),
  );

  const existingSet = new Set(existing);
  const seen = new Set<string>();
  const toAdd: string[] = [];
  let skipCount = 0;

  for (const w of valid) {
    if (existingSet.has(w) || seen.has(w)) {
      skipCount++;
    } else {
      seen.add(w);
      toAdd.push(w);
    }
  }

  return { toAdd, skipCount, invalidCount };
}

// ---------- Component ----------

export const CustomWords: React.FC<CustomWordsProps> = React.memo(
  ({ descriptionMode = "tooltip", grouped = false }) => {
    const { t } = useTranslation();
    const { getSetting, updateSetting, isUpdating } = useSettings();
    const [newWord, setNewWord] = useState("");
    const [showImportModal, setShowImportModal] = useState(false);
    const [importText, setImportText] = useState("");
    const fileInputRef = useRef<HTMLInputElement>(null);

    const customWords: string[] = getSetting("custom_words") || [];

    // ---------- Single-word handlers ----------

    const handleAddWord = () => {
      const trimmedWord = newWord.trim();
      const sanitizedWord = trimmedWord.replace(ILLEGAL_CHARS, "");
      if (sanitizedWord && sanitizedWord.length <= MAX_WORD_LEN) {
        if (customWords.includes(sanitizedWord)) {
          toast.error(
            t("settings.advanced.customWords.duplicate", {
              word: sanitizedWord,
            }),
          );
          return;
        }
        updateSetting("custom_words", [...customWords, sanitizedWord]);
        setNewWord("");
      }
    };

    const handleRemoveWord = (wordToRemove: string) => {
      updateSetting(
        "custom_words",
        customWords.filter((word) => word !== wordToRemove),
      );
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        handleAddWord();
      }
    };

    // ---------- Export ----------

    const handleExport = () => {
      const content = customWords.join("\n");
      const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const date = new Date().toISOString().slice(0, 10).replace(/-/g, "");
      const a = document.createElement("a");
      a.href = url;
      a.download = `handy-custom-words-${date}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };

    // ---------- Import modal ----------

    const preview = computePreview(importText, customWords);

    const handleFileLoad = (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result;
        if (typeof text === "string") {
          setImportText(text);
        }
      };
      reader.readAsText(file, "utf-8");
      // reset so the same file can be reloaded
      e.target.value = "";
    };

    const handleConfirmImport = () => {
      if (preview.toAdd.length === 0) return;
      updateSetting("custom_words", [...customWords, ...preview.toAdd]);
      setImportText("");
      setShowImportModal(false);
    };

    const handleCancelImport = () => {
      setImportText("");
      setShowImportModal(false);
    };

    return (
      <>
        <SettingContainer
          title={t("settings.advanced.customWords.title")}
          description={t("settings.advanced.customWords.description")}
          descriptionMode={descriptionMode}
          grouped={grouped}
        >
          <div className="flex items-center gap-1.5 flex-wrap">
            <Input
              type="text"
              className="max-w-44"
              value={newWord}
              onChange={(e) => setNewWord(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={t("settings.advanced.customWords.placeholder")}
              variant="compact"
              disabled={isUpdating("custom_words")}
            />
            <Button
              onClick={handleAddWord}
              disabled={
                !newWord.trim() ||
                newWord.trim().length > MAX_WORD_LEN ||
                isUpdating("custom_words")
              }
              variant="primary"
              size="sm"
            >
              {t("settings.advanced.customWords.add")}
            </Button>
            <button
              type="button"
              onClick={() => setShowImportModal(true)}
              disabled={isUpdating("custom_words")}
              title={t("settings.advanced.customWords.import")}
              aria-label={t("settings.advanced.customWords.import")}
              className="ml-1 p-1.5 rounded text-text/50 hover:text-text hover:bg-mid-gray/15 disabled:opacity-40 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 3v12m0 0l-4-4m4 4l4-4M5 21h14" />
              </svg>
            </button>
            <button
              type="button"
              onClick={handleExport}
              disabled={customWords.length === 0 || isUpdating("custom_words")}
              title={t("settings.advanced.customWords.export")}
              aria-label={t("settings.advanced.customWords.export")}
              className="p-1.5 rounded text-text/50 hover:text-text hover:bg-mid-gray/15 disabled:opacity-40 transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 21V9m0 0l-4 4m4-4l4 4M5 3h14" />
              </svg>
            </button>
          </div>
        </SettingContainer>

        {/* Word chips */}
        {customWords.length > 0 && (
          <div
            className={`px-4 p-2 ${grouped ? "" : "rounded-lg border border-mid-gray/20"} flex flex-wrap gap-1`}
          >
            {customWords.map((word) => (
              <Button
                key={word}
                onClick={() => handleRemoveWord(word)}
                disabled={isUpdating("custom_words")}
                variant="secondary"
                size="sm"
                className="inline-flex items-center gap-1 cursor-pointer"
                aria-label={t("settings.advanced.customWords.remove", { word })}
              >
                <span>{word}</span>
                <svg
                  className="w-3 h-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </Button>
            ))}
          </div>
        )}

        {/* Import modal */}
        {showImportModal && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
            onClick={(e) => {
              if (e.target === e.currentTarget) handleCancelImport();
            }}
          >
            <div className="bg-background rounded-xl border border-mid-gray/30 shadow-xl w-full max-w-md mx-4 p-5 flex flex-col gap-4">
              <h2 className="text-sm font-semibold">
                {t("settings.advanced.customWords.bulkImportTitle")}
              </h2>

              {/* Textarea */}
              <Textarea
                className="w-full min-h-[120px]"
                placeholder={t(
                  "settings.advanced.customWords.bulkImportPlaceholder",
                )}
                value={importText}
                onChange={(e) => setImportText(e.target.value)}
                autoFocus
              />

              {/* File upload */}
              <div className="flex items-center gap-2">
                <span className="text-text/60 text-xs">
                  {t("settings.advanced.customWords.fromFile")}
                </span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                >
                  {t("settings.advanced.customWords.fileTypes")}
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.csv,.json"
                  className="hidden"
                  onChange={handleFileLoad}
                />
              </div>

              {/* Preview */}
              {importText.trim() && (
                <p className="text-text/60 text-xs">
                  {t("settings.advanced.customWords.preview", {
                    add: preview.toAdd.length,
                    skip: preview.skipCount,
                    invalid: preview.invalidCount,
                  })}
                </p>
              )}

              {/* Action buttons */}
              <div className="flex justify-end gap-2">
                <Button
                  variant="secondary"
                  size="md"
                  onClick={handleCancelImport}
                >
                  {t("settings.advanced.customWords.cancel")}
                </Button>
                <Button
                  variant="primary"
                  size="md"
                  disabled={preview.toAdd.length === 0}
                  onClick={handleConfirmImport}
                >
                  {t("settings.advanced.customWords.confirmAdd")}
                </Button>
              </div>
            </div>
          </div>
        )}
      </>
    );
  },
);
