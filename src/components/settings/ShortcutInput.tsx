import React from "react";
import { platform } from "@tauri-apps/plugin-os";
import { GlobalShortcutInput } from "./GlobalShortcutInput";
import { HandyKeysShortcutInput } from "./HandyKeysShortcutInput";

interface ShortcutInputProps {
  descriptionMode?: "inline" | "tooltip";
  grouped?: boolean;
  shortcutId: string;
  disabled?: boolean;
}

/**
 * Selects the appropriate shortcut input implementation based on platform:
 * - Linux: GlobalShortcutInput (Tauri built-in global-shortcut)
 * - macOS / Windows: HandyKeysShortcutInput
 */
export const ShortcutInput: React.FC<ShortcutInputProps> = (props) => {
  const isLinux = platform() === "linux";

  if (isLinux) {
    return <GlobalShortcutInput {...props} />;
  }

  return <HandyKeysShortcutInput {...props} />;
};
