//
//  ContentView.swift
//  VorTeX_frontend
//
//  Created by Goge on 2025/4/18.
//

import SwiftUI

// The calculation modes
enum CalculationMode: String, CaseIterable, Identifiable {
    case latex = "LaTeX"
    case matlab = "MATLAB"
    case sympy = "SymPy"
    case matrix = "Matrix"
    var id: String { self.rawValue }
}

// Angle Modes
enum AngleMode: String, CaseIterable, Identifiable {
    case degree = "Degree"
    case radian = "Radian"
    var id: String { self.rawValue }
}

// Matrix Operations
enum MatrixOperation: String, CaseIterable, Identifiable {
    case determinant = "Determinant"
    case inverse = "Inverse"
    case eigenvalues = "Eigenvalues"
    case rank = "Rank"
    case multiply = "Multiply"
    case add = "Add"
    case subtract = "Subtract"
    case divide = "Divide"
    case differentiate = "Differentiate"
    var id: String { self.rawValue }
}

struct ContentView: View {
    // State variables
    @State private var selectedMode: CalculationMode = .latex
    @State private var selectedAngleMode: AngleMode = .degree
    @State private var formulaInput: String = ""
    @State private var matrixInput: String = ""
    @State private var resultOutput: String = "Result will appear here."
    @State private var selectedMatrixOperation: MatrixOperation = .determinant

    @State private var showingSettings = false
    @State private var showingLegend = false
    
    // Access the theme environment
    @EnvironmentObject private var themeEnvironment: ThemeEnvironment

    private var showVisualizeButton: Bool {
        switch selectedMode {
        case .latex, .matlab, .sympy:
            return true
        case .matrix:
            return false
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 15) {
                VStack(alignment: .leading) {
                    Text("Mode").font(.caption).foregroundColor(.secondary)
                    Picker("Mode", selection: $selectedMode) {
                        ForEach(CalculationMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                .frame(maxWidth: 300)

                VStack(alignment: .leading) {
                    Text("Angle").font(.caption).foregroundColor(.secondary)
                    Picker("Angle", selection: $selectedAngleMode) {
                        ForEach(AngleMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                }
                .frame(maxWidth: 150)

                Spacer()

                HStack {
                    Button {
                        showingSettings = true
                    } label: {
                        Label("Settings", systemImage: "gear")
                    }
                    .buttonStyle(.bordered)
                    .padding(.horizontal, 5)

                    Button {
                        showingLegend = true
                    } label: {
                         Label("Legend", systemImage: "list.bullet.rectangle")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            .background(themeEnvironment.currentTheme.background.opacity(0.9))

            VStack(alignment: .leading, spacing: 20) {
                if selectedMode == .matrix {
                    Text("Matrix Input:")
                        .font(.headline)
                        .foregroundColor(themeEnvironment.currentTheme.title)
                    TextEditor(text: $matrixInput)
                        .frame(height: 150)
                        .background(themeEnvironment.currentTheme.inputBackground)
                        .foregroundColor(themeEnvironment.currentTheme.text)
                        .font(.system(.body, design: .monospaced))
                        .autocorrectionDisabled()
                        .cornerRadius(6)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.1), radius: 3, x: 0, y: 2)

                    HStack {
                         Picker("Operation", selection: $selectedMatrixOperation) {
                            ForEach(MatrixOperation.allCases) { op in
                                Text(op.rawValue).tag(op)
                            }
                        }
                         .pickerStyle(.menu)
                         .frame(maxWidth: 180)

                        Spacer()

                        HStack {
                            Button("Store") { /* Add action */ }
                                .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                            Button("Recall") { /* Add action */ }
                                .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                        }
                    }

                } else {
                    Text("Expression Input:")
                        .font(.headline)
                        .foregroundColor(themeEnvironment.currentTheme.title)
                    TextEditor(text: $formulaInput)
                        .frame(height: 150)
                        .background(themeEnvironment.currentTheme.inputBackground)
                        .foregroundColor(themeEnvironment.currentTheme.text)
                        .font(.system(.body, design: .monospaced))
                        .autocorrectionDisabled()
                        .cornerRadius(6)
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.1), radius: 3, x: 0, y: 2)
                }

                Text("Result:")
                    .font(.headline)
                    .foregroundColor(themeEnvironment.currentTheme.title)
                ScrollView {
                    Text(resultOutput)
                        .foregroundColor(themeEnvironment.currentTheme.text)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .textSelection(.enabled)
                }
                .frame(height: 100)
                .background(themeEnvironment.currentTheme.inputBackground)
                .font(.system(.body, design: .monospaced))
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(themeEnvironment.currentTheme.border, lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.1), radius: 3, x: 0, y: 2)

            }
            .padding(20)

            Spacer()

            HStack {
                Spacer()

                if showVisualizeButton {
                    Button("Visualize") {
                        print("Visualize button tapped")
                    }
                    .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                }

                 Button(selectedMode == .matrix ? "Calculate Matrix" : "Calculate") {
                    print("Calculate button tapped for mode: \(selectedMode.rawValue)")
                 }
                 .buttonStyle(ThemedButtonStyle(theme: themeEnvironment.currentTheme))
                 .keyboardShortcut(.defaultAction)
                 .shadow(color: .black.opacity(0.15), radius: 4, x: 0, y: 3)
            }
             .padding(.horizontal, 20)
             .padding(.bottom, 15)

        }
        .background(themeEnvironment.currentTheme.background)
        .frame(width: 700, height: 500)
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
        .sheet(isPresented: $showingLegend) {
            Text("Legend View Placeholder")
                .frame(minWidth: 500, minHeight: 400)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ThemeEnvironment())
}
