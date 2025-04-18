import SwiftUI

private struct ThemeKey: EnvironmentKey {
    static let defaultValue: ThemeColors = ThemeManager.shared.getThemeColors(for: AppTheme.anysphere.key)
}

extension EnvironmentValues {
    var themeColors: ThemeColors {
        get { self[ThemeKey.self] }
        set { self[ThemeKey.self] = newValue }
    }
}

class ThemeEnvironment: ObservableObject {
    @Published var currentTheme: ThemeColors
    @AppStorage("appTheme") private var currentThemeKey: String = AppTheme.anysphere.key
    
    // For observing system appearance changes
    @Environment(\.colorScheme) private var colorScheme
    private var appearanceObserver: NSObjectProtocol?
    
    init() {
        self.currentTheme = ThemeManager.shared.getThemeColors(for: AppTheme.anysphere.key)
        self.currentTheme = ThemeManager.shared.getThemeColors(for: currentThemeKey)
        
        setupAppearanceObserver()
    }
    
    deinit {
        if let observer = appearanceObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
    
    func updateTheme(to themeKey: String) {
        currentThemeKey = themeKey
        currentTheme = ThemeManager.shared.getThemeColors(for: themeKey)
    }
    
    private func setupAppearanceObserver() {
        appearanceObserver = NotificationCenter.default.addObserver(
            forName: NSNotification.Name("AppleInterfaceThemeChangedNotification"),
            object: nil,
            queue: .main
        ) { [weak self] _ in
            if self?.currentThemeKey == "system" {
                DispatchQueue.main.async {
                    self?.refreshSystemTheme()
                }
            }
        }
    }
    
    func refreshSystemTheme() {
        if currentThemeKey == "system" {
            currentTheme = ThemeManager.shared.getThemeColors(for: "system")
            objectWillChange.send()
        }
    }
}

struct ThemeApplier: ViewModifier {
    @ObservedObject var themeEnvironment: ThemeEnvironment
    @Environment(\.colorScheme) var colorScheme
    
    func body(content: Content) -> some View {
        content
            .environment(\.themeColors, themeEnvironment.currentTheme)
            .environmentObject(themeEnvironment)
            .onChange(of: colorScheme) { _, _ in
                if UserDefaults.standard.string(forKey: "appTheme") == "system" {
                    themeEnvironment.refreshSystemTheme()
                }
            }
    }
}

extension View {
    func withTheme(_ themeEnvironment: ThemeEnvironment) -> some View {
        self.modifier(ThemeApplier(themeEnvironment: themeEnvironment))
    }
} 