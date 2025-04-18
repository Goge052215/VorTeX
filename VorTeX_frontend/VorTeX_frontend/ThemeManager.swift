import SwiftUI

// Theme structure
struct ThemeColors {
    let background: Color
    let text: Color
    let button: Color
    let buttonText: Color
    let buttonHover: Color
    let inputBackground: Color
    let border: Color
    let title: Color
}

class ThemeManager {
    static let shared = ThemeManager()
    
    // Theme dictionary
    private let themes: [String: ThemeColors] = [
        "tokyo_night": ThemeColors(
            background: Color(hex: "#1a1b26"),
            text: Color(hex: "#a9b1d6"),
            button: Color(hex: "#3d59a1"),
            buttonText: Color(hex: "#ffffff"),
            buttonHover: Color(hex: "#2ac3de"),
            inputBackground: Color(hex: "#24283b"),
            border: Color(hex: "#414868"),
            title: Color(hex: "#ffffff")
        ),
        "aura": ThemeColors(
            background: Color(hex: "#15141b"),
            text: Color(hex: "#edecee"),
            button: Color(hex: "#a277ff"),
            buttonText: Color(hex: "#edecee"),
            buttonHover: Color(hex: "#61ffca"),
            inputBackground: Color(hex: "#15141b"),
            border: Color(hex: "#6d6d6d"),
            title: Color(hex: "#edecee")
        ),
        "light": ThemeColors(
            background: Color(hex: "#f0f0f0"),
            text: Color(hex: "#333333"),
            button: Color(hex: "#4a90e2"),
            buttonText: Color(hex: "#ffffff"),
            buttonHover: Color(hex: "#357abd"),
            inputBackground: Color(hex: "#ffffff"),
            border: Color(hex: "#cccccc"),
            title: Color(hex: "#333333")
        ),
        "anysphere": ThemeColors(
            background: Color(hex: "#0e1116"),
            text: Color(hex: "#e6edf3"),
            button: Color(hex: "#E6C895"),
            buttonText: Color(hex: "#ffffff"),
            buttonHover: Color(hex: "#E6C895"),
            inputBackground: Color(hex: "#0e1116"),
            border: Color(hex: "#30363d"),
            title: Color(hex: "#ffffff")
        ),
        "system": ThemeColors(
            background: Color(NSColor.windowBackgroundColor),
            text: Color(NSColor.textColor),
            button: Color(NSColor.controlAccentColor),
            buttonText: Color(NSColor.textBackgroundColor),
            buttonHover: Color(NSColor.controlAccentColor).opacity(0.8),
            inputBackground: Color(NSColor.textBackgroundColor),
            border: Color(NSColor.separatorColor),
            title: Color(NSColor.labelColor)
        )
    ]
    
    func getThemeColors(for themeKey: String) -> ThemeColors {
        return themes[themeKey] ?? themes["anysphere"]!
    }
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
} 