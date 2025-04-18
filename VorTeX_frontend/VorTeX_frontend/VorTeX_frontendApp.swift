//
//  VorTeX_frontendApp.swift
//  VorTeX_frontend
//
//  Created by Goge on 2025/4/18.
//

import SwiftUI

@main
struct VorTeX_frontendApp: App {
    // Create a theme environment that will be shared across the app
    @StateObject private var themeEnvironment = ThemeEnvironment()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(width: 700, height: 500)
                .withTheme(themeEnvironment)
                .onAppear {
                    // Initialize theme on app start
                    if let savedTheme = UserDefaults.standard.string(forKey: "appTheme") {
                        themeEnvironment.updateTheme(to: savedTheme)
                    }
                }
        }
    }
}
