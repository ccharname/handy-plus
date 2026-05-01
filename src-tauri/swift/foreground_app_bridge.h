#ifndef foreground_app_bridge_h
#define foreground_app_bridge_h

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ForegroundAppInfo {
    char* bundle_id;      // nullable: Bundle Identifier (macOS) or exe basename (Windows)
    char* process_name;   // nullable: localized app name or process name
    char* window_title;   // nullable: frontmost window title (requires Screen Recording on macOS)
} ForegroundAppInfo;

// Returns a heap-allocated ForegroundAppInfo for the current frontmost application.
// The caller MUST call free_foreground_app_info() on the returned pointer.
// Returns NULL only on catastrophic allocation failure; fields may individually be NULL.
ForegroundAppInfo* get_foreground_app(void);

// Free a ForegroundAppInfo previously returned by get_foreground_app().
void free_foreground_app_info(ForegroundAppInfo* info);

#ifdef __cplusplus
}
#endif

#endif /* foreground_app_bridge_h */
