// SettingsView.swift

import SwiftUI

// Settings Categories
enum SettingsCategory: String, CaseIterable, Identifiable, Hashable {
    case appearance = "Appearance"
    case font = "Font Settings"
    case matlab = "MATLAB License"
    case logs = "Debug Logs"
    case about = "About"
    var id: String { self.rawValue }

    var iconName: String {
        switch self {
        case .appearance: return "paintbrush"
        case .font: return "textformat.size"
        case .matlab: return "terminal" 
        case .logs: return "doc.text"
        case .about: return "info.circle"
        }
    }
}

// Enum for Themes (matching Python themes)
enum AppTheme: String, CaseIterable, Identifiable {
    case anysphere
    case tokyoNight = "Tokyo night"
    case aura = "Aura"
    case light = "Light"
    case system = "System Default"

    var id: String { self.rawValue }

    var key: String {
        switch self {
        case .anysphere: return "anysphere"
        case .tokyoNight: return "tokyo_night"
        case .aura: return "aura"
        case .light: return "light"
        case .system: return "system"
        }
    }
}


struct SettingsView: View {
    @State private var selectedCategory: SettingsCategory? = .appearance

    @AppStorage("appTheme") private var currentThemeKey: String = AppTheme.anysphere.key
    @AppStorage("mathFontFamily") private var mathFontFamily: String = "Menlo"
    @AppStorage("mathFontSize") private var mathFontSize: Int = 14
    
    @StateObject private var themeEnvironment = ThemeEnvironment()

    let availableFontFamilies = ["Menlo", "Monaco", "Courier New", "Consolas", "Monaspace Neon"]
    let availableFontSizes = [10, 11, 12, 13, 14, 16, 18]

    @State private var selectedFontFamily: String
    @State private var selectedFontSize: Int

    @State private var licenseStatus: String = "Not Checked"

    @Environment(\.dismiss) var dismiss

    init() {
        _selectedFontFamily = State(initialValue: UserDefaults.standard.string(forKey: "mathFontFamily") ?? "Menlo")
        _selectedFontSize = State(initialValue: UserDefaults.standard.integer(forKey: "mathFontSize") != 0 ? UserDefaults.standard.integer(forKey: "mathFontSize") : 14)
    }


    var body: some View {
        NavigationSplitView {
            List(SettingsCategory.allCases, selection: $selectedCategory) { category in
                Label {
                    Text(category.rawValue)
                        .font(.system(.body, weight: .medium))
                        .padding(.vertical, 4)
                } icon: {
                    Image(systemName: category.iconName)
                        .foregroundColor(.accentColor)
                        .frame(width: 26, height: 26)
                }
                .tag(category)
                .padding(.vertical, 4)
            }
            .listStyle(.sidebar) 
            .navigationTitle("Settings")
            #if os(macOS)
            .navigationSplitViewColumnWidth(min: 200, ideal: 220)
            #endif

        } detail: {
            VStack(alignment: .leading) {
                if let category = selectedCategory {
                    switch category {
                    case .appearance:
                        AppearanceSettingsView(themeEnvironment: themeEnvironment)
                            .navigationTitle("Appearance")
                    case .font:
                         FontSettingsView(
                            selectedFontFamily: $selectedFontFamily,
                            selectedFontSize: $selectedFontSize,
                            availableFontFamilies: availableFontFamilies,
                            availableFontSizes: availableFontSizes,
                            applyAction: applyFontSettings
                         )
                         .navigationTitle("Font Settings")
                    case .matlab:
                         MATLABLicenseView(status: $licenseStatus, checkAction: checkMatlabLicense)
                         .navigationTitle("MATLAB License")
                    case .logs:
                         DebugLogsView(clearAction: clearLogs, exportAction: exportLogs)
                         .navigationTitle("Debug Logs")
                    case .about:
                         AboutView()
                         .navigationTitle("About")
                    }
                } else {
                    Text("Select a category")
                        .foregroundColor(.secondary)
                }

                Spacer()

                HStack {
                    Spacer()
                    Button("Done") {
                        dismiss()
                    }
                    .keyboardShortcut(.defaultAction)
                }
                .padding()

            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            #if os(macOS)
            .navigationSplitViewColumnWidth(min: 400, ideal: 420)
            #endif
            .background(themeEnvironment.currentTheme.background)

        }
         .frame(width: 600, height: 400)
         .onAppear {
             selectedFontFamily = mathFontFamily
             selectedFontSize = mathFontSize
             themeEnvironment.updateTheme(to: currentThemeKey)
         }
         .onChange(of: currentThemeKey) { _, newValue in
             themeEnvironment.updateTheme(to: newValue)
         }
         .withTheme(themeEnvironment)
    }

    // --- Actions ---

    func applyFontSettings() {
        print("Applying font settings: \(selectedFontFamily) \(selectedFontSize)pt")
        mathFontFamily = selectedFontFamily
        mathFontSize = selectedFontSize
    }

    func checkMatlabLicense() {
        print("Checking MATLAB license...")
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            let isValid = Bool.random()
            licenseStatus = isValid ? "Valid" : "Invalid/Unavailable"
        }
    }

    func clearLogs() {
        print("Clearing logs...")
    }

    func exportLogs() {
        print("Exporting logs...")
    }
}

// --- Subviews for each settings category ---
// Make content more compact but still readable

struct AppearanceSettingsView: View {
    @ObservedObject var themeEnvironment: ThemeEnvironment
    @AppStorage("appTheme") private var currentThemeKey: String = AppTheme.anysphere.key

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Choose your preferred visual theme for the calculator interface.")
                     .foregroundColor(themeEnvironment.currentTheme.text)
                     .font(.subheadline)

                Picker("Theme", selection: $currentThemeKey) {
                    ForEach(AppTheme.allCases) { theme in
                        Text(theme.rawValue).tag(theme.key)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 200)
                .tint(themeEnvironment.currentTheme.button)
                
                // Theme preview
                VStack(alignment: .leading, spacing: 12) {
                    Text("Theme Preview")
                        .font(.headline)
                        .foregroundColor(themeEnvironment.currentTheme.title)
                    
                    HStack(spacing: 10) {
                        RoundedRectangle(cornerRadius: 5)
                            .fill(themeEnvironment.currentTheme.background)
                            .overlay(
                                Text("Background")
                                    .font(.caption)
                                    .foregroundColor(themeEnvironment.currentTheme.text)
                            )
                            .frame(width: 100, height: 40)
                        
                        RoundedRectangle(cornerRadius: 5)
                            .fill(themeEnvironment.currentTheme.inputBackground)
                            .overlay(
                                Text("Input")
                                    .font(.caption)
                                    .foregroundColor(themeEnvironment.currentTheme.text)
                            )
                            .frame(width: 100, height: 40)
                            .overlay(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                            )
                    }
                    
                    HStack(spacing: 10) {
                        Button(action: {}) {
                            Text("Button")
                                .foregroundColor(themeEnvironment.currentTheme.buttonText)
                        }
                        .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                        
                        Text("Text Sample")
                            .foregroundColor(themeEnvironment.currentTheme.text)
                        
                        Text("Title")
                            .foregroundColor(themeEnvironment.currentTheme.title)
                            .fontWeight(.bold)
                    }
                }
                .padding()
                .background(themeEnvironment.currentTheme.background)
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                )

                Spacer()
            }
            .padding(25)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(themeEnvironment.currentTheme.background)
    }
}

// Button style that uses theme colors
struct ThemedButtonStyle: ButtonStyle {
    let theme: ThemeColors
    
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.clear)
            .foregroundColor(theme.buttonText)
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(configuration.isPressed ? theme.buttonHover : theme.button, lineWidth: 1)
            )
            .scaleEffect(configuration.isPressed ? 0.98 : 1.0)
    }
}

struct FontSettingsView: View {
    @Binding var selectedFontFamily: String
    @Binding var selectedFontSize: Int
    let availableFontFamilies: [String]
    let availableFontSizes: [Int]
    let applyAction: () -> Void
    
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Customize the fonts used throughout the calculator interface.")
                    .foregroundColor(themeEnvironment.currentTheme.text)
                    .font(.subheadline)

                Group {
                    HStack {
                        Text("Math Input Font:")
                             .foregroundColor(themeEnvironment.currentTheme.text)
                             .frame(minWidth: 120, alignment: .leading)
                        Picker("Font Family", selection: $selectedFontFamily) {
                            ForEach(availableFontFamilies, id: \.self) { family in
                                Text(family).font(.custom(family, size: 14)).tag(family)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: 250)
                        .tint(themeEnvironment.currentTheme.button)
                    }

                    HStack {
                        Text("Font Size:")
                             .foregroundColor(themeEnvironment.currentTheme.text)
                             .frame(minWidth: 120, alignment: .leading)
                        Picker("Font Size", selection: $selectedFontSize) {
                            ForEach(availableFontSizes, id: \.self) { size in
                                Text("\(size) pt").tag(size)
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: 100)
                        .tint(themeEnvironment.currentTheme.button)
                    }
                }
                .padding(.bottom, 5)

                Text("Math Input Preview:")
                    .foregroundColor(themeEnvironment.currentTheme.title)
                TextEditor(text: .constant("\\int_{0}^{\\pi} \\sin(x) dx = 2\n\\frac{d}{dx}[x^2] = 2x\n\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}"))
                    .font(.custom(selectedFontFamily, size: CGFloat(selectedFontSize)))
                    .frame(height: 100)
                    .background(themeEnvironment.currentTheme.inputBackground)
                    .foregroundColor(themeEnvironment.currentTheme.text)
                    .cornerRadius(6)
                    .disabled(true)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                    )

                Button("Apply Font Settings", action: applyAction)
                    .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                    .padding(.top, 5)

                Spacer()
            }
            .padding(25)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(themeEnvironment.currentTheme.background)
    }
}

struct MATLABLicenseView: View {
    @Binding var status: String
    let checkAction: () -> Void
    
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment

    var body: some View {
         ScrollView {
            VStack(alignment: .leading, spacing: 15) {
                 Text("A valid MATLAB license is required for MATLAB engine computations. Without a valid license, the calculator will operate in SymPy-only mode with limited functionality.")
                    .foregroundColor(themeEnvironment.currentTheme.text)
                    .font(.subheadline)

                HStack {
                    Text("License Status:")
                        .foregroundColor(themeEnvironment.currentTheme.text)
                    Text(status)
                        .fontWeight(status == "Valid" ? .bold : .regular)
                        .foregroundColor(status == "Valid" ? .green : (status == "Not Checked" ? .secondary : .red))
                }
                .padding(.top, 5)

                Button("Check License", action: checkAction)
                    .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))

                Spacer()
            }
            .padding(25)
         }
         .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
         .background(themeEnvironment.currentTheme.background)
    }
}

struct DebugLogsView: View {
    let clearAction: () -> Void
    let exportAction: () -> Void
    @State private var logContent: String = "Loading logs..." // Placeholder
    
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            TextEditor(text: $logContent)
                 .font(.system(size: 12, design: .monospaced))
                 .background(themeEnvironment.currentTheme.inputBackground)
                 .foregroundColor(themeEnvironment.currentTheme.text)
                 .cornerRadius(6)
                 .overlay(
                     RoundedRectangle(cornerRadius: 6)
                         .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                 )
                 .frame(maxHeight: .infinity)

            HStack {
                Button("Clear Logs", action: clearAction)
                    .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                Button("Export Logs", action: exportAction)
                    .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                Spacer()
            }
        }
        .padding(25)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(themeEnvironment.currentTheme.background)
        .onAppear(perform: loadLogs)
    }

    func loadLogs() {
        logContent = "2025-04-19 ... Log entry 1\n2025-04-19 ... Log entry 2\n..."
    }
}

struct AboutView: View {
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 15) {
                Group {
                    Text("VorTeX Calculator")
                        .font(.headline)
                        .foregroundColor(themeEnvironment.currentTheme.title)
                    Text("Version 1.0.3")
                        .foregroundColor(themeEnvironment.currentTheme.text)
                    Text("A LaTeX-based calculator with MATLAB/Sympy integration.")
                        .foregroundColor(themeEnvironment.currentTheme.text)
                    Text("© 2025 VorTeX Team")
                        .foregroundColor(themeEnvironment.currentTheme.text)
                }
                .padding(.vertical, 2)

                Spacer()
            }
            .padding(25)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(themeEnvironment.currentTheme.background)
    }
}


#Preview {
    SettingsView()
        .environmentObject(ThemeEnvironment())
}